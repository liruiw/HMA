import cv2
import spaces

import torch
import numpy as np
import einops
import skimage
import time

from hma.model.st_mask_git import STMaskGIT
from hma.model.st_mar import STMAR
from datasets.utils import get_image_encoder
from hma.data import DATA_FREQ_TABLE
from hma.train_multi import SVD_SCALE

from typing import Optional, Tuple, Callable, Dict


class Simulator:
    def set_initial_state(self, state):
        """
        the initial state of the simulated scene
        e.g.
        1. in robomimic, it's the scene state vector
        2. in genie, it's the initial frames to prompt the model
        """
        raise NotImplementedError

    @torch.inference_mode()
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def dt(self):
        raise NotImplementedError


class PhysicsSimulator(Simulator):
    def __init__(self):
        super().__init__()

    # physics engine should be able to update dt
    def set_dt(self, dt):
        raise NotImplementedError

    # physics engine should be able to get scene state
    # e.g., robot joint positions, object positions, etc.
    def get_raw_state(self, port: Optional[str] = None):
        raise NotImplementedError

    @property
    def action_dimension(self):
        raise NotImplementedError


class LearnedSimulator(Simulator):
    def __init__(self):
        super().__init__()


# data replayed respect physics, so we inherit from PhysicsSimulator
# it can be considered as a special case of PhysicsSimulator
class ReplaySimulator(PhysicsSimulator):
    def __init__(self, frames, prompt_horizon: int = 0, dt: Optional[float] = None):
        super().__init__()
        self.frames = frames
        self.frame_idx = prompt_horizon
        assert self.frame_idx < len(self.frames)
        self._dt = dt
        self.prompt_horizon = prompt_horizon

    def __len__(self):
        return len(self.frames) - self.prompt_horizon

    def step(self, action):
        frame = self.frames[self.frame_idx]
        assert self.frame_idx < len(self.frames)
        self.frame_idx = self.frame_idx + 1
        return {"pred_next_frame": frame}

    def reset(self):  # return current frame = last frame of prompt
        self.frame_idx = self.prompt_horizon
        return self.prompt()[-1]

    def prompt(self):
        return self.frames[: self.prompt_horizon]

    @property
    def dt(self):
        return self._dt


@spaces.GPU
class GenieSimulator(LearnedSimulator):

    average_delta_psnr_over = 5

    def __init__(
        self,
        # image preprocessing
        max_image_resolution: int = 1024,
        resize_image: bool = True,
        resize_image_resolution: int = 256,
        # tokenizer setting
        image_encoder_type: str = "temporalvae",
        image_encoder_ckpt: str = "stabilityai/stable-video-diffusion-img2vid",
        quantize: bool = False,
        quantization_slice_size: int = 16,
        # dynamics backbone setting
        backbone_type: str = "stmar",
        backbone_ckpt: str = "data/mar_ckpt/robomimic",
        prompt_horizon: int = 11,
        inference_iterations: Optional[int] = None,
        sampling_temperature: float = 0.0,
        action_stride: Optional[int] = None,
        domain: str = "robomimic",
        genie_frequency: int = 2,
        # misc
        measure_step_time: bool = False,
        compute_psnr: bool = False,
        compute_delta_psnr: bool = False,  # act as a signal for controlability
        gaussian_action_perturbation_scale: Optional[float] = None,
        device: str = "cuda",
        physics_simulator: Optional[PhysicsSimulator] = None,
        physics_simulator_teacher_force: Optional[int] = None,
        post_processor: Optional[Callable] = None,  # on the predicted image, e.g., add action
        allow_external_prompt: bool = False,
    ):
        super().__init__()

        assert quantize == (
            image_encoder_type == "magvit"
        ), "Currently quantization if and only if magvit is the image encoder."
        assert image_encoder_type in [
            "magvit",
            "temporalvae",
        ], "Image encoder type must be either 'magvit' or 'temporalvae'."
        assert (
            not quantize or image_encoder_type == "magvit"
        ), "If quantize is enabled, image encoder type must be 'magvit'."
        assert backbone_type in ["stmaskgit", "stmar"], "Backbone type must be either 'stmaskgit' or 'stmar'."
        if physics_simulator is None:
            assert (
                physics_simulator_teacher_force is None
            ), "Physics simulator teacher force is only available when physics simulator is provided."
            assert compute_psnr is False, "PSNR computation is only available when physics simulator is provided."
            assert (
                compute_delta_psnr is False
            ), "Delta PSNR computation is only available when physics simulator is provided."

        if action_stride is None:
            action_stride = DATA_FREQ_TABLE[domain] // genie_frequency
        if compute_delta_psnr:
            compute_psnr = True  # to compute delta psnr, psnr must be computed
        if inference_iterations is None:
            if backbone_type == "stmaskgit":
                inference_iterations = 2
            elif backbone_type == "stmar":
                inference_iterations = 2

        # misc
        self.device = torch.device(device)
        self.measure_step_time = measure_step_time
        self.compute_psnr = compute_psnr
        self.compute_delta_psnr = compute_delta_psnr
        self.allow_external_prompt = allow_external_prompt

        # image preprocessing
        self.max_image_resolution = max_image_resolution
        self.resize_image = resize_image
        self.resize_image_resolution = resize_image_resolution

        # load image encoder
        self.image_encoding_dtype = torch.bfloat16
        self.quantize = quantize
        self.quant_slice_size = quantization_slice_size
        self.image_encoder_type = image_encoder_type
        self.image_encoder = (
            get_image_encoder(image_encoder_type, image_encoder_ckpt)
            .to(device=self.device, dtype=self.image_encoding_dtype)
            .eval()
        )

        # load STMaskGIT model (STMAR is inherited from STMaskGIT)
        self.prompt_horizon = prompt_horizon
        self.domain = domain
        self.genie_frequency = genie_frequency
        self.inference_iterations = inference_iterations
        self.sampling_temperature = sampling_temperature
        self.action_stride = action_stride
        self.gauss_act_perturb_scale = gaussian_action_perturbation_scale
        self.backbone_type = backbone_type
        if backbone_type == "stmaskgit":
            self.backbone = STMaskGIT.from_pretrained(backbone_ckpt)
        else:
            self.backbone = STMAR.from_pretrained(backbone_ckpt)
        self.backbone = self.backbone.to(device=self.device).eval()

        self.post_processor = post_processor

        # load physics simulator if available
        # the phys sim to get ground truth image,
        # assume the phys sim has aligned prompt frames
        self.gt_phys_sim = physics_simulator
        self.gt_teacher_force = physics_simulator_teacher_force

        # history buffer, i.e., the input to the model
        self.cached_actions = None  # (prompt_horizon, action_stride, A)
        self.cached_latent_frames = None  # (prompt_horizon, ...)
        self.init_prompt = None  # (prompt_frames, prompt_actions)

        self.step_count = 0

        # report model size
        print(
            "================ Model Size Report ================\n"
            f"    encoder size: {sum(p.numel() for p in self.image_encoder.parameters()) / 1e6:.3f}M \n"
            f"    backbone size: {sum(p.numel() for p in self.backbone.parameters()) / 1e6:.3f}M\n"
            "==================================================="
        )

    def set_initial_state(self, state: Tuple[np.ndarray, np.ndarray]):
        if not self.allow_external_prompt and self.gt_phys_sim is not None:
            raise NotImplementedError("Initial state is set by the physics simulator.")
        self.init_prompt = state

    @torch.inference_mode()
    def step(self, action: np.ndarray) -> Dict:
        # action: (action_stride, A) OR (A,)
        # return: (H, W, 3)
        assert (
            self.cached_latent_frames is not None and self.cached_actions is not None
        ), "Model is not prompted yet. Please call `set_initial_state` first."

        if action.ndim == 1:
            action = np.tile(action, (self.action_stride, 1))

        # perturb action
        if self.gauss_act_perturb_scale is not None:
            action = np.random.normal(action, self.gauss_act_perturb_scale)

        # encoding
        input_latent_states = (
            torch.cat(
                [
                    self.cached_latent_frames,
                    torch.zeros_like(self.cached_latent_frames[[0]]),
                ]
            )
            .unsqueeze(0)
            .to(torch.float32)
        )

        input_latent_states = input_latent_states[:, : self.prompt_horizon + 1]

        # dtype conversion and mask token
        if self.backbone_type == "stmaskgit":
            input_latent_states = input_latent_states.long()
            input_latent_states[:, -1] = self.backbone.mask_token_id
        elif self.backbone_type == "stmar":
            input_latent_states[:, -1] = self.backbone.mask_token

        # dynamics rollout
        action = torch.from_numpy(action).to(device=self.device)
        input_actions = (
            torch.cat(
                [  # (1, prompt_horizon + 1, action_stride * A)
                    self.cached_actions,
                    action.unsqueeze(0),
                    action.unsqueeze(0),  # the last action is not used, but we need a_{t-1}, s_{t-1} to predict s_t
                ]
            )
            .view(1, -1, action.shape[-1])
            .to(torch.float32)
        )  #  + 1
        input_actions = input_actions[:, : self.prompt_horizon + 1]

        if self.measure_step_time:
            start_time = time.time()
        pred_next_latent_state = self.backbone.maskgit_generate(
            input_latent_states,
            out_t=input_latent_states.shape[1] - 1,
            maskgit_steps=self.inference_iterations,
            temperature=self.sampling_temperature,
            action_ids=input_actions,
            domain=[self.domain],
        )[0].squeeze(0)

        # decoding
        pred_next_frame = self._decode_image(pred_next_latent_state)

        # timing
        if self.measure_step_time:
            end_time = time.time()

        step_result = {
            "pred_next_frame": pred_next_frame,
        }
        if self.measure_step_time:
            step_result["step_time"] = end_time - start_time

        # physics simulation
        if self.gt_phys_sim is not None:
            for a in action.cpu().numpy():
                gt_result = self.gt_phys_sim.step(a)
            gt_next_frame = cv2.resize(gt_result["pred_next_frame"], pred_next_frame.shape[:2])
            step_result["gt_next_frame"] = gt_next_frame
            gt_result.pop("pred_next_frame")
            step_result.update(gt_result)

            # gt state observation
            try:
                raw_state = self.gt_phys_sim.get_raw_state()
                step_result.update(raw_state)
            except NotImplementedError:
                pass

            # compute PSNR against ground truth
            if self.compute_psnr:
                psnr = skimage.metrics.peak_signal_noise_ratio(
                    image_true=gt_next_frame / 255.0, image_test=pred_next_frame / 255.0, data_range=1.0
                )
                step_result["psnr"] = psnr

            # controlability metric
            if self.compute_delta_psnr:
                delta_psnr = 0.0
                for _ in range(self.average_delta_psnr_over):
                    # re-mask the input latent states for masked prediction
                    if self.backbone_type == "stmaskgit":
                        input_latent_states = input_latent_states.long()
                        input_latent_states[:, self.prompt_horizon] = self.backbone.mask_token_id
                    elif self.backbone_type == "stmar":
                        input_latent_states[:, self.prompt_horizon] = self.backbone.mask_token
                    # sample random action from N(0, 1)
                    random_input_actions = torch.randn_like(input_actions)
                    random_pred_next_latent_state = self.backbone.maskgit_generate(
                        input_latent_states,
                        out_t=self.prompt_horizon,
                        maskgit_steps=self.inference_iterations,
                        temperature=self.sampling_temperature,
                        action_ids=random_input_actions,
                        domain=[self.domain],
                        skip_normalization=True,
                    )[0].squeeze(0)
                    random_pred_next_frame = self._decode_image(random_pred_next_latent_state)
                    this_delta_psnr = step_result["psnr"] - skimage.metrics.peak_signal_noise_ratio(
                        image_true=gt_next_frame / 255.0, image_test=random_pred_next_frame / 255.0, data_range=1.0
                    )
                    delta_psnr += this_delta_psnr / self.average_delta_psnr_over
                step_result["delta_psnr"] = delta_psnr

            if self.gt_teacher_force is not None and self.step_count % self.gt_teacher_force == 0:
                pred_next_latent_state = self._encode_image(gt_next_frame)

        # update history buffer
        self.cached_latent_frames = torch.cat([self.cached_latent_frames[1:], pred_next_latent_state.unsqueeze(0)])
        self.cached_actions = torch.cat([self.cached_actions[1:], action.unsqueeze(0)])

        # post processing
        if self.post_processor is not None:
            pred_next_frame = self.post_processor(pred_next_frame, action)

        self.step_count += 1

        return step_result

    @torch.inference_mode()
    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        # (H, W, 3)
        image = (
            torch.from_numpy(self._normalize_image(image).transpose(2, 0, 1))
            .to(device=self.device, dtype=self.image_encoding_dtype)
            .unsqueeze(0)
        )
        H, W = image.shape[-2:]

        if self.quantize:
            H //= self.quant_slice_size
            W //= self.quant_slice_size
            _, _, indices, _ = self.image_encoder.encode(image, flip=True)
            indices = einops.rearrange(indices, "(h w) -> h w", h=H, w=W)
            indices = indices.to(torch.int32)
            return indices

        else:
            if self.image_encoder_type == "magvit":
                latent = self.image_encoder.encode_without_quantize(image)
            elif self.image_encoder_type == "temporalvae":
                latent_dist = self.image_encoder.encode(image).latent_dist
                latent = latent_dist.mean
                latent *= SVD_SCALE
                latent = einops.rearrange(latent, "b c h w -> b h w c")
            else:
                pass
            latent = latent.squeeze(0).to(torch.float32)
            return latent

    @torch.inference_mode()
    def _decode_image(self, latent: torch.Tensor) -> np.ndarray:
        # latent can be either quantized indices or raw latent
        # return (H, W, 3)

        latent = latent.to(device=self.device).unsqueeze(0)

        if self.quantize:
            latent = self.image_encoder.quantize.get_codebook_entry(
                einops.rearrange(latent, "b h w -> b (h w)"),
                bhwc=(*latent.shape, self.image_encoder.quantize.codebook_dim),
            ).flip(1)

        latent = latent.to(device=self.device, dtype=self.image_encoding_dtype)
        if self.image_encoder_type == "magvit":
            decoded_image = self.image_encoder.decode(latent)
        elif self.image_encoder_type == "temporalvae":
            latent = einops.rearrange(latent, "b h w c -> b c h w")
            latent /= SVD_SCALE
            # HACK: clip for less visual artifacts
            latent = torch.clamp(latent, -25, 25)
            decoded_image = self.image_encoder.decode(latent, num_frames=1).sample
        decoded_image = decoded_image.squeeze(0).to(torch.float32).detach().cpu().numpy()
        decoded_image = self._unnormalize_image(decoded_image).transpose(1, 2, 0)
        return decoded_image

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        # (H, W, 3) normalized to [-1, 1]
        # if `resize`, resize the shorter side to `resized_res`
        #   and then do a center crop

        image = np.asarray(image, dtype=np.float32)
        image /= 255.0
        H, W = image.shape[:2]

        # resize if asked
        if self.resize_image:
            resized_res = self.resize_image_resolution
            if H < W:
                Hnew, Wnew = resized_res, int(resized_res * W / H)
            else:
                Hnew, Wnew = int(resized_res * H / W), resized_res
            image = cv2.resize(image, (Wnew, Hnew))

            # center crop
            H, W = image.shape[:2]
            Hstart = (H - resized_res) // 2
            Wstart = (W - resized_res) // 2
            image = image[Hstart : Hstart + resized_res, Wstart : Wstart + resized_res]

        # resize if resolution is too large
        elif H > self.max_image_resolution or W > self.max_image_resolution:
            if H < W:
                Hnew, Wnew = int(self.max_image_resolution * H / W), self.max_image_resolution
            else:
                Hnew, Wnew = self.max_image_resolution, int(self.max_image_resolution * W / H)
            image = cv2.resize(image, (Wnew, Hnew))

        image = image * 2 - 1.0
        return image

    def _unnormalize_image(self, image: np.ndarray) -> np.ndarray:
        # (H, W, 3) from [-1, 1] to [0, 255]
        # NOTE: clip happens here
        image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def reset(self) -> np.ndarray:
        # if ground truth physics simulator is provided,
        # return the the side-by-side concatenated image

        # get the initial prompt from the physics simulator if not yet set
        if not self.allow_external_prompt and self.gt_phys_sim is not None:
            image_prompt = np.tile(self.gt_phys_sim.reset(), (self.prompt_horizon, 1, 1, 1)).astype(np.uint8)
            action_prompt = np.zeros(
                (self.prompt_horizon, self.action_stride, self.gt_phys_sim.action_dimension)
            ).astype(np.float32)
        else:
            assert self.init_prompt is not None, "Initial state is not set."
            image_prompt, action_prompt = self.init_prompt

        # standardize the image
        image_prompt = [self._unnormalize_image(self._normalize_image(frame)) for frame in image_prompt]

        current_image = image_prompt[-1]

        action_prompt = torch.from_numpy(action_prompt).to(device=self.device)
        self.cached_actions = action_prompt

        # convert to latent
        self.cached_latent_frames = torch.stack([self._encode_image(frame) for frame in image_prompt], axis=0)

        if self.resize_image:
            current_image = cv2.resize(current_image, (self.resize_image_resolution, self.resize_image_resolution))

        if self.gt_phys_sim is not None:
            current_image = np.concatenate([current_image, current_image], axis=1)

        self.step_count = 0

        return current_image

    def close(self):
        if self.gt_phys_sim is not None:
            try:
                self.gt_phys_sim.close()
            except NotImplementedError:
                pass

    @property
    def dt(self):
        return 1.0 / self.genie_frequency
