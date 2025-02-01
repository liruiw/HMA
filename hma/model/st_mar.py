# https://github.com/LTH14/mar/tree/main/
from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from einops import rearrange
import mup
from hma.config import DiffusionGenieConfig

from .diffloss import DiffLoss
from .st_mask_git import STMaskGIT
from transformers.utils import ModelOutput


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking, dim=-1, index=order[:, : mask_len.long()], src=torch.ones(bsz, seq_len).cuda()
    ).bool()
    return masking


class FixedMuReadout(mup.MuReadout):
    def forward(self, x):
        """
        Using `return super(mup.MuReadout, self).forward(self.output_mult * x / self.width_mult())` with `torch.compile`
        results in two divisions by `self.width_mult()` for some reason
        """
        # return F.linear(self.output_mult * x / self.width_mult(), self.weight, self.bias)  # equivalent
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())


class STMAR(STMaskGIT):
    """Spatial-Time MAR with VisionTransformer backbone"""

    def __init__(self, config: DiffusionGenieConfig):
        self.diffloss_w = config.diffloss_w
        self.diffloss_d = config.diffloss_d
        self.num_sampling_steps = config.num_sampling_steps
        self.grad_checkpointing = config.grad_checkpointing

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.patch_size = config.patch_size
        self.vae_stride = config.vae_stride
        self.buffer_size = config.buffer_size
        self.vae_embed_dim = config.vae_embed_dim
        self.maskgit_steps = config.maskgit_steps
        super().__init__(config)

        # --------------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.vae_embed_dim))
        self.token_embed = nn.Linear(
            config.vae_embed_dim * self.config.patch_size**2, config.d_model, bias=False
        )  # hard coded
        cls = FixedMuReadout if config.use_mup else nn.Linear  # (Fixed)MuReadout might slow dow down compiled training?
        self.out_x_proj = cls(config.d_model, config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.z_proj_ln = nn.LayerNorm(config.d_model, eps=1e-6)
        self.seq_len = config.S // (self.config.patch_size**2)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len * config.T, config.d_model))

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=config.vae_embed_dim * self.config.patch_size**2,
            z_channels=config.d_model,
            width=config.diffloss_w,
            depth=config.diffloss_d,
            num_sampling_steps=config.num_sampling_steps,
            grad_checkpointing=config.grad_checkpointing,
        )

        self.diffusion_batch_mul = config.diffusion_batch_mul
        self.initialize_weights()

    def init_action_projectors(
        self,
        domains: list[str],
        d_actions: list[int],
        action_stats: list[list[list[float]]],
        action_network: str = "mlp",
    ):
        super().init_action_projectors(domains, d_actions, action_stats, action_network, use_diffusion=True)
        self.action_diff_losses = nn.ModuleDict()

        # action heads are heterogeneous
        for domain, d_action in zip(self.config.action_domains, self.config.d_actions):
            self.action_diff_losses[domain] = DiffLoss(
                target_channels=d_action,
                z_channels=self.config.d_model,
                width=self.diffloss_w,
                depth=self.diffloss_d,
                num_sampling_steps=self.num_sampling_steps,
                grad_checkpointing=self.grad_checkpointing,
            )

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm parameters
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        self.init_weights()

    def set_mup_shapes(self, rescale_params=False):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256  # currently hardcoding to this shape
        base_model = STMAR(base_config)
        if hasattr(self, "action_preprocessor"):
            for base_layer, layer in zip(base_model.decoder.layers, self.decoder.layers):
                base_layer.action_projectors = layer.action_projectors
            base_model.action_preprocessor = self.action_preprocessor

        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    def compute_action_loss_and_acc(self, z, target, domain, mask=None):
        bsz, seq_len, *_ = target.shape
        # not so sure if this repeated is needed
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        if mask is not None:
            mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.action_diff_losses[domain[0]](z=z, target=target, mask=mask)  #

        acc = torch.zeros_like(loss)
        return loss, acc

    def compute_video_loss_and_acc(self, z, target, mask=None):
        z = rearrange(z, "B C T H W -> B (T H W) C").float()

        target = rearrange(target, "B T H W C  -> B (T H W) C").float()
        bsz, seq_len, *_ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)

        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        if mask is not None:
            mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)  # no need for

        acc = torch.zeros_like(loss)
        return loss, acc

    def compute_latents(self, x_THW, action_ids: torch.Tensor = None, domain=None, action_mask=None, **kwargs):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        pos_embed_TSC = self.pos_embed_TSC
        diffusion_pos_embed_learned = self.diffusion_pos_embed_learned
        b, t, h, w, c = x_THW.shape
        x_TSC = rearrange(x_THW, "B T H W C -> B T (H W) C").float()
        x_TSC = self.token_embed(x_TSC)
        T = x_TSC.shape[1]

        if action_ids is not None:
            # currently, action_preprocessor just normalizes the actions
            skip_normalization = kwargs.get("skip_normalization", False)
            if not skip_normalization:
                action_ids = self.action_preprocessor[domain[0]](action_ids)
            action_ids = self.action_mlp[domain[0]](action_ids)  # [B, T, D]

            if "concat" in self.config.action_network:
                # randomly dropped the conditioning
                if self.config.action_network == "resampler_concat":
                    action_condition = self.action_projectors[domain[0]](action_ids[:, :T])
                else:
                    action_condition = action_ids[:, :T, None].repeat(
                        1, 1, self.config.action_token_size, 1
                    )  # [B, T, S, C]

                # we add masked tokens between 0 (fully unmasked as in video pred) and 1 (fully masked as in policies) for training losses
                # if we have actions and are trying to predict actions
                x_TSC = torch.concat((x_TSC, action_condition), dim=2)  # [B, T, S, C]

        elif self.config.jointly_predict_actions:
            # all masked when predicting actions and there is no input actions
            action_condition = self.action_mask_tokens[:, :T].repeat(1, 1, self.config.action_token_size, 1)
            x_TSC = torch.concat((x_TSC, action_condition), dim=2)  # [B, T, S, C]

        x_TSC = self.z_proj_ln(x_TSC + pos_embed_TSC[:, : x_TSC.shape[1], : x_TSC.shape[2]])

        # additive position embeddings, using the same vocab space
        domain = domain[0] if domain is not None else None
        x_TSC = self.decoder(x_TSC, action_ids=action_ids, domain=domain)

        # dummy if are not used
        decoded_states = rearrange(diffusion_pos_embed_learned, "B (T H W) C -> B C T H W", T=self.config.T, H=h, W=w)
        decoded_actions = None
        if self.config.jointly_predict_actions:
            decoded_actions = x_TSC[:, :, -self.config.action_token_size :].mean(dim=2)  # pool all tokens

        x_TSC = x_TSC[:, :, : h * w]  # remove action tokens for states
        x_next_TSC = self.decoder_norm(self.out_x_proj(x_TSC))
        x_next_TSC = x_next_TSC + diffusion_pos_embed_learned.view(1, self.config.T, h * w, self.config.d_model)[:, :T]
        decoded_states = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=h, W=w)

        return decoded_states, decoded_actions

    def patchify(self, x):
        bsz, t, h, w, c = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, t, h_, p, w_, p, c)
        x = torch.einsum("nthpwqc->nthwpqc", x)
        x = x.reshape(bsz, t, h_, w_, c * p**2)
        return x

    def unpatchify(self, x):
        # input: B T H W C
        p = self.patch_size
        bsz, t, h, w, _ = x.shape
        c = self.vae_embed_dim
        x = x.reshape(bsz, t, h, w, p, p, c)
        x = torch.einsum("nthwpqc->nthpwqc", x)
        x = x.reshape(bsz, t, h * p, w * p, c)
        return x

    def forward(self, input_ids, labels, action_ids=None, domain="default", **kwargs):
        assert "masked_tokens_indicator" in kwargs
        relevant_mask = kwargs["masked_tokens_indicator"]
        # class embed
        T, H, W = self.config.T, self.h, self.w
        if "h" in kwargs:
            H = kwargs["h"][0]
        if "w" in kwargs:
            W = kwargs["w"][0]

        x_THW = rearrange(input_ids, "B (T H W) C -> B T H W C", T=T, H=H, W=W)
        action_mask = None

        if action_ids is not None and self.config.jointly_predict_actions:
            action_labels = action_ids.clone()
            action_mask = torch.zeros(len(action_ids), T, 1)
            random_timesteps = torch.randint(0, T, (len(action_ids), 1), device=action_ids.device)

            # Set all timesteps from the sampled t to T to 1
            for i, t in enumerate(random_timesteps):
                action_mask[i, t:] = 1

            # Move the mask to the same device and dtype as x_THW if needed
            action_mask = action_mask.unsqueeze(-1).cuda().to(x_THW.dtype)

        # change masked token id -> masked token latents
        x_THW[relevant_mask] = self.mask_token
        x_THW = self.patchify(x_THW)
        latents_CTHW, action_outputs = self.compute_latents(
            x_THW, action_ids=action_ids, domain=domain, action_mask=action_mask, **kwargs
        )

        labels = rearrange(labels, "B (T H W) C -> B T H W C", T=T, H=H, W=W)
        labels = self.patchify(labels)

        relevant_loss = torch.zeros(1).to(x_THW.device)
        relevant_acc = torch.zeros(1).to(x_THW.device)
        relevant_mask = self.patchify(relevant_mask[..., None]).sum(-1) > 0  # as long as it's not no mask

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        if self.config.jointly_predict_states:
            # could also get mask of corrupted tokens by uncommenting line in `get_maskgit_collator`
            relevant_loss, relevant_acc = self.compute_video_loss_and_acc(
                latents_CTHW, labels, relevant_mask
            )  # relevant_mask

        # compute the action losses
        if action_outputs is not None:
            action_loss, _ = self.compute_action_loss_and_acc(action_outputs, action_labels, domain, action_mask)
            return ModelOutput(
                loss=relevant_loss,
                acc=relevant_acc,
                logits=latents_CTHW,
                action_loss=action_loss,
                actions=action_outputs,
            )
        return ModelOutput(loss=relevant_loss, acc=relevant_acc, logits=latents_CTHW)

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        min_new_tokens: int = None,
        return_logits: int = False,
        return_with_actions: bool = False,
        temperature: float = 1.0,
        action_ids: torch.Tensor = None,
        domain: str = "default",
        action_only: bool = False,
        state_only: bool = False,
        **kwargs
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Args designed to match the format of Llama.
        We ignore `attention_mask`, and use `max_new_tokens` to determine the number of frames to generate.

        Returns: `(sample_THW, factored_logits)` if `return_logits` else `sample_THW`
            sample_THW: size (B, num_new_frames * H * W) corresponding to autoregressively generated
                unfactorized token ids for future frames.
            Optionally, factored_logits: size (B, factored_vocab_size, num_factored_vocabs, num_new_frames, H, W).
        """
        assert min_new_tokens in (
            None,
            max_new_tokens,
        ), "Expecting `min_new_tokens`, if specified, to match `max_new_tokens`."

        # assert max_new_tokens % self.config.S == 0, "Expecting `max_new_tokens` to be a multiple of `self.config.S`."
        h, w, c = self.h, self.w, self.vae_embed_dim
        if "h" in kwargs:
            h = kwargs["h"][0]
        if "w" in kwargs:
            w = kwargs["w"][0]
            S = h * w

        num_new_frames = max_new_tokens // S
        inputs_THW = rearrange(input_ids.clone(), "b (t h w) c-> b t h w c", h=h, w=w)
        inputs_masked_THW = torch.cat(
            [inputs_THW, self.mask_token[None, None].repeat(inputs_THW.size(0), num_new_frames, h, w, 1)], dim=1
        )

        all_factored_logits = []
        for timestep in range(inputs_THW.size(1), inputs_THW.size(1) + num_new_frames):
            # could change sampling hparams
            sample_HW, factored_logits, actions = self.maskgit_generate(
                inputs_masked_THW,
                timestep,
                maskgit_steps=self.maskgit_steps,
                temperature=temperature,
                action_ids=action_ids,
                domain=domain,
                action_only=action_only,
                state_only=state_only,
                **kwargs
            )
            inputs_masked_THW[:, timestep] = sample_HW
            all_factored_logits.append(factored_logits)

        predicted_tokens = rearrange(inputs_masked_THW, "B T H W C -> B (T H W) C")
        if return_with_actions:
            # unnormalize actions
            actions = self.action_preprocessor[domain[0]].unnormalize(actions)
            return predicted_tokens, actions
        elif return_logits:
            return predicted_tokens, torch.stack(all_factored_logits, dim=3)  # (b, c, num_new_frames, h, w)
        else:
            return predicted_tokens

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW,
        out_t: int,
        unmask_mode: str = "random",
        action_ids=None,
        domain="default",
        maskgit_steps=8,
        cfg=1.0,
        temperature=1.0,
        cfg_schedule="linear",
        action_only: bool = False,
        state_only: bool = False,
        **kwargs
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        # init and sample generation orders
        assert out_t, "maskgit_generate requires out_t > 0"
        prompt_THW = self.patchify(prompt_THW)
        bs, t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        S = h * w
        orders = self.sample_orders(bs)  # random order
        sampled_action_token_latent = None

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)

        # patchify the prompt
        latents_CTHW, action_outputs = self.compute_latents(prompt_THW, action_ids=action_ids, domain=domain, **kwargs)
        latents_CHW = latents_CTHW[:, :, out_t]
        orig_latents_CHW = latents_CHW.clone()
        # Return these original logits, not logits after partially sampling.

        for step in range(maskgit_steps):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                latents_CHW, action_outputs = self.compute_latents(
                    prompt_THW, action_ids=action_ids, domain=domain, **kwargs
                )
                latents_CHW = latents_CHW[:, :, out_t]

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / maskgit_steps)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(), torch.minimum(torch.sum(~unmasked, dim=-1, keepdims=True) - 1, mask_len)
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bs, self.seq_len)
            mask = ~unmasked

            if step >= maskgit_steps - 1:
                mask_to_pred = mask[:bs].bool()  # last step
            else:
                mask_to_pred = torch.logical_xor(mask[:bs].bool(), mask_next.bool())
            mask = mask_next

            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            latents_CHW = rearrange(latents_CHW, "b c h w -> b (h w) c")
            latents_CHW = latents_CHW[mask_to_pred.nonzero(as_tuple=True)]

            # copy previously unmasked values from prompt input into sample
            total_mask_len = unmasked.shape[1]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (total_mask_len - unmasked.sum()) / total_mask_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            # need to reshape back
            sampled_token_latent = self.diffloss.sample(
                latents_CHW.contiguous(), temperature, cfg_iter, clip_denoised=True
            )
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            if action_outputs is not None and self.config.jointly_predict_actions:
                sampled_action_token_latent = self.action_diff_losses[domain[0]].sample(
                    action_outputs.view(-1, action_outputs.shape[-1]), temperature, cfg_iter, clip_denoised=True
                )
                if not cfg == 1.0:
                    sampled_action_token_latent, _ = sampled_action_token_latent.chunk(2, dim=0)

            prompt_THW_reshape = rearrange(prompt_THW, "B T H W C -> B T (H W) C")
            prompt_THW_reshape[:, out_t][mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            prompt_THW = rearrange(prompt_THW_reshape, "B T (H W) C -> B T H W C", H=h, W=w)

        # Return the final sample and logits
        prompt_THW = self.unpatchify(prompt_THW)
        return prompt_THW[:, out_t], orig_latents_CHW, sampled_action_token_latent

    @torch.no_grad()
    def maskgit_generate_horizon(
        self,
        prompt_THW,
        out_t_min: int,
        out_t_max: int,
        unmask_mode: str = "random",
        action_ids=None,
        domain="default",
        maskgit_steps=8,
        cfg=1.0,
        temperature=1.0,
        cfg_schedule="linear",
        **kwargs
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        # init and sample generation orders

        prompt_THW = self.patchify(prompt_THW)
        bs, t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        S = h * w
        orders = self.sample_orders(bs)  # random order

        # this will be modified in place on each iteration of this loop
        horizon = out_t_max - out_t_min
        unmasked = self.init_mask(prompt_THW, t=horizon)

        # patchify the prompt
        latents_CTHW, latents_actions = self.compute_latents(prompt_THW, action_ids=action_ids, domain=domain, **kwargs)

        latents_CHW = latents_CTHW[:, :, out_t_min:out_t_max]
        orig_latents_CHW = latents_CHW.clone()
        # Return these original logits, not logits after partially sampling.

        seq_len = horizon * self.seq_len

        for step in range(maskgit_steps):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                latents_CHW, latents_actions = self.compute_latents(
                    prompt_THW, action_ids=action_ids, domain=domain, **kwargs
                )
                latents_CHW = latents_CHW[:, :, out_t_min:out_t_max]

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / maskgit_steps)
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(), torch.minimum(torch.sum(~unmasked, dim=-1, keepdims=True) - 1, mask_len)
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bs, seq_len)
            mask = ~unmasked

            if step >= maskgit_steps - 1:
                mask_to_pred = mask[:bs].bool()  # last step
            else:
                mask_to_pred = torch.logical_xor(mask[:bs].bool(), mask_next.bool())
            mask = mask_next

            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            latents_CHW = rearrange(latents_CHW, "b c t h w -> b (t h w) c")
            latents_CHW = latents_CHW[mask_to_pred.nonzero(as_tuple=True)]

            # copy previously unmasked values from prompt input into sample
            # cfg schedule follow Muse
            total_mask_len = unmasked.shape[1]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (total_mask_len - unmasked.sum()) / total_mask_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            # need to reshape back
            sampled_token_latent = self.diffloss.sample(latents_CHW.contiguous(), temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            if latents_actions is not None and self.config.jointly_predict_actions:
                action_outputs = self.action_diff_losses[domain[0]].sample(
                    latents_actions.view(-1, latents_actions.shape[-1]), temperature, cfg_iter
                )
                if not cfg == 1.0:
                    action_outputs, _ = action_outputs.chunk(2, dim=0)

            # need to reshape backout_t_max - out_t_min_latent.chunk(2, dim=0)
            prompt_THW_reshape = rearrange(prompt_THW[:, out_t_min:out_t_max], "B T H W C -> B (T H W) C")
            prompt_THW_reshape[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            prompt_THW[:, out_t_min:out_t_max] = rearrange(
                prompt_THW_reshape.clone(), "B (T H W) C -> B T H W C", T=horizon, H=h, W=w
            )

        # Return the final sample and logits
        prompt_THW = self.unpatchify(prompt_THW)
        return prompt_THW[:, out_t_min:out_t_max], orig_latents_CHW, action_outputs
