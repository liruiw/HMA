"""
Example Usage:
python hma/evaluate.py  --checkpoint_dir  data/kaist_model/step_50/  --val_data_dir data/kaist_nonprehensile_converted_externally_to_rlds_magvit_max1000000_val --use_tokeniz
ed_images
"""

import argparse
import pickle
import time
import os
import sys
from collections import defaultdict
from pathlib import Path
import math

import accelerate
import wandb

import lpips
import torch
import transformers
from accelerate import DataLoaderConfiguration
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator
import numpy as np

sys.path.append(os.getcwd())
import re

from hma.data import RawTokenDataset
from hma.visualize import decode_latents_wrapper
from hma.model.st_mask_git import STMaskGIT
from skimage import metrics as image_metrics
from hma.data import RawFeatureDataset, RawImageDataset
from hma.model.st_mar import STMAR
from datasets import utils

from external.common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from hma.eval_utils import decode_tokens, compute_lpips, AvgMetric, compute_loss

# Hardcoded values for the v1.1 dataset
WINDOW_SIZE = 12
STRIDE = 15  # Data is 30 Hz so with stride 15, video is 2 Hz


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GENIE-style models.")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/1x_humanoid_magvit_traj10_val",
        help="A directory with video data, should have a `metadata.json` and `video.bin`."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size, current script only supports a single GPU."
    )
    parser.add_argument(
        "--maskgit_steps", type=int, default=2, help="Number of MaskGIT sampling steps."
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature. If `temperature` <= 1e-8, will do greedy sampling."
    )
    parser.add_argument(
        "--save_outputs_dir", type=str,
        help="Debug option. If specified, will save model predictions and ground truths to this directory. "
             "Specifically, will save `{pred_frames,pred_logits,gtruth_frames,gtruth_tokens}.pt`"
    )
    parser.add_argument(
        "--max_examples", type=int, default=200,
        help="If specified, will stop evaluation early after `max_examples` examples."
    )
    parser.add_argument(
        "--autoregressive_time", action="store_true",
        help="If True, autoregressive generation in time dimension."
    )
    parser.add_argument(
        "--add_action_input", action="store_true",
        help="If True, uses action in the video output."
    )
    parser.add_argument(
        "--perturbation_type", type=str, default="gaussian",
        help="Type of perturbation to apply to the action input. Options: gaussian "
    )
    parser.add_argument(
        "--perturbation_scale", type=float, default=0.1,
        help="Perturbation applied to each action dimension."
    )
    parser.add_argument(
        "--project_prefix", type=str, default="", help="Project suffix."
    )
    parser.add_argument(
        "--use_feature", action="store_true",
        help="visualize the features rather than tokens"
    )
    parser.add_argument(
        "--use_tokenized_images", action="store_true",
        help="use tokenized images as inputs, instead of raw images. Not recommended for comparison to non-VQ models like diffusion."
    )
    parser.add_argument(
        "--wandb_run_name", type=str,
        help="If specified, will use this as the name of the wandb run. Will infer name from checkpoint path otherwise."
    )
    return parser.parse_args()

def get_model_step(checkpoint_dir):
    if os.path.exists(f"{checkpoint_dir}/scheduler.bin"):
        sch = torch.load(f"{checkpoint_dir}/scheduler.bin")
        return sch['_step_count']
    return 0

class GenieEvaluator:
    def __init__(self, args, decode_latents, device="cuda"):
        super().__init__()
        if not os.path.exists(args.checkpoint_dir + "/config.json"):
            # search and find the latest modified checkpoint folder
            dirs = [os.path.join(args.checkpoint_dir, f.name) for f in os.scandir(args.checkpoint_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)

            # if len(dirs) > 3 and os.path.join(args.checkpoint_dir, "epoch_1") in dirs:
            #     dirs.remove(os.path.join(args.checkpoint_dir, "epoch_1"))

            if len(dirs) == 0:
                exit(f"No checkpoint found in {args.checkpoint_dir}")
            paths = dirs[:-3]

            # only keep the last 3
            for path in paths:
                print(f"evaluation: remove rm -rf {path}")
                os.system(f"rm -rf {path}")

            args.checkpoint_dir = dirs[-1]

        print("Loading model from:", args.checkpoint_dir)
        print("action input included:", args.add_action_input)
        self.model = STMaskGIT.from_pretrained(args.checkpoint_dir)
        self.model_step = get_model_step(args.checkpoint_dir)
        self.model = self.model.to(device=device)
        self.model.eval()

        self.decode_latents = decode_latents
        self.device = device
        self.args = args

    def predict_zframe_logits(self, input_ids: torch.Tensor, action_ids: torch.Tensor = None, domains = None,
                                 skip_normalization: bool = False) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ... frame_{T-1}],
        predict the tokens in the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        Image logits are denoised in parallel across spatial dimension and teacher-forced
        across the time dimension. To compute logits, we save both the samples and logits as we do MaskGIT generation.

        Total number of forward passes is (T-1) * maskgit steps.

        Args:
            input_ids: Tensor of size (B, T*H*W) corresponding to flattened, tokenized images.

        Returns: (samples_THW, factored_logits)
            samples_THW:
                size (B, T, H, W) corresponding to the token ids of the predicted frames.
                May differ from the argmax of `factored_logits` if not greedy sampling.
            factored_logits:
                size (B, 512, 2, T-1, H, W) corresponding to the predicted logits.
                Note that we are factorizing the 2**18 vocabulary into two separate vocabularies of size 512 each.
        """
        inputs_THW = rearrange(input_ids, "b (t h w) ... -> b t h w ...", t=WINDOW_SIZE,
                               h=self.args.latent_h, w=self.args.latent_w).to(self.device)
        all_samples = []
        all_logits = []
        samples_HW = inputs_THW.clone()

        for timestep in range(1, WINDOW_SIZE):
            print(f"Generating frame {timestep}")
            inputs_masked = inputs_THW.clone().long()
            if self.args.autoregressive_time:
                if timestep > self.model.config.num_prompt_frames:
                    inputs_masked[:, timestep-1] = samples_HW.clone()

            inputs_masked[:, timestep:] = self.model.mask_token_id

            # MaskGIT sampling
            samples_HW, factored_logits, _ = self.model.maskgit_generate(
                inputs_masked, out_t=timestep, maskgit_steps=self.args.maskgit_steps,
                temperature=self.args.temperature, action_ids=action_ids, domain=domains,
                skip_normalization=skip_normalization
            )

            all_samples.append(samples_HW)
            all_logits.append(factored_logits)

        samples_THW = torch.stack(all_samples, dim=1)
        return samples_THW, torch.stack(all_logits, dim=3)

    def predict_next_frames(self, samples_THW) -> torch.Tensor:
        """
        All model submissions should have this defined.

        Like predict_next_frames, this is teacher-forced along spatial dimension, autoregressive along time  dimension.

        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ..., frame_{T-1}],
        predict the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        For this model, the frames are generated by using the argmax of `predict_zframe_logits`
        and decoding the quantized latent space tokens back to the original image space.

        Args:
            samples_THW: LongTensor of size (B, T, H, W) corresponding to sampled images in the quantized latent space.

        Returns:
            LongTensor of size (B, T-1, 3, 256, 256) corresponding to the predicted frames.
        """
        return decode_tokens(samples_THW.cpu(), self.decode_latents)


@torch.no_grad()
def main():
    transformers.set_seed(42)
    args = parse_args()

    # allow different batch sizes in final batch
    accelerator = accelerate.Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    decode_latents = decode_latents_wrapper(device=accelerator.device)

    # save the results to wandb. hardcoded the input dataset to have magvit and will change later
    dataset = re.search(r"data/(.*?)_magvit", args.val_data_dir).group(1)
    dataset_original = dataset

    # rtrim the last / and get the last part of the path
    args.checkpoint_dir = args.checkpoint_dir.rstrip('/')
    if args.wandb_run_name is not None:
        name = args.wandb_run_name
    else:
        name = args.checkpoint_dir.split('/')[-1]

    if accelerator.is_main_process:
        wandb.teardown()
        wandb.init(project='video_val', resume="allow", id=f"{args.project_prefix}{name}", name=f"{args.project_prefix}{name}", settings=wandb.Settings(start_method="thread"))

    evaluator = GenieEvaluator(args, decode_latents)
    with_action_input = evaluator.model.config.use_actions

    assert hasattr(evaluator, "model"), "Expected Evaluator to have attribute `model`."
    evaluator.model = accelerator.prepare_model(evaluator.model, evaluation_mode=True)  # No DDP

    # compute stride from the pre-trained model for the dataset
    # this is hacky and I much rather just save the stride in the model config
    action_d = len(evaluator.model.action_preprocessor[dataset].mean)
    action_d_horizon = evaluator.model.config.d_actions[evaluator.model.config.action_domains.index(dataset)]
    stride = action_d_horizon // action_d

    if not args.use_tokenized_images:
        args.val_data_dir = args.val_data_dir.replace("magvit", "image")
        val_dataset = RawImageDataset(args.val_data_dir, window_size=WINDOW_SIZE,
                                      compute_stride_from_freq_table=False,
                                      stride=stride, filter_overlaps=True,
                                      use_actions=with_action_input)

    else:
        val_dataset = RawTokenDataset(args.val_data_dir, window_size=WINDOW_SIZE,
                                      compute_stride_from_freq_table=False,
                                      stride=stride, filter_overlaps=True,
                                      use_actions=with_action_input)

    lpips_alex = lpips.LPIPS(net="alex")  # Calculate LPIPS w/ AlexNet, which is the fastest model out of their options
    print("load data directory:", args.val_data_dir)

    if args.max_examples is not None:
        val_dataset.valid_start_inds = val_dataset.valid_start_inds[:args.max_examples]

    dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, drop_last=False)
    metrics = defaultdict(AvgMetric)
    latent_side_len = 16  # hardcoded
    args.latent_h = args.latent_w = latent_side_len
    dataloader = accelerator.prepare(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch is None:
            print(f"Stopping evaluation on {accelerator.process_index=} due to empty batch")
            break  # if we just continue, we get an error because dataloader errors the next iteration

        if not args.use_tokenized_images:
            # tokenize the batches on the fly
            images = batch["images"].detach().cpu().numpy().astype(np.uint8)
            outputs = []
            for context in images:
                output = []
                for image_t in context:
                    output_t = utils.get_quantized_image_embeddings(
                        image_t,
                        encoder_type="magvit",
                        encoder_name_or_path="data/magvit2.ckpt",
                    )

                    output.append(output_t)
                outputs.append(output)

            batch["input_ids"] = torch.from_numpy(np.stack(outputs).astype(np.int64)).to(evaluator.device)
            batch["input_ids"] = rearrange(batch["input_ids"], "b t h w -> b (t h w)")
            batch["labels"] = batch["input_ids"].clone()

        batch_size = batch["input_ids"].size(0)
        reshaped_input_ids = rearrange(batch["input_ids"], "b (t h w) ... -> b t h w ...", t=WINDOW_SIZE,
                                       h=latent_side_len, w=latent_side_len)

        start_time = time.time()
        if not with_action_input:
            samples, factored_logits = evaluator.predict_zframe_logits(batch["input_ids"].to(evaluator.device), domains=[val_dataset.name])
        else:
            samples, factored_logits = evaluator.predict_zframe_logits(batch["input_ids"].to(evaluator.device),
                                                                       batch["action_ids"].to(evaluator.device), [val_dataset.name])

        frames_per_batch = (WINDOW_SIZE - 1) * batch["input_ids"].size(0)
        metrics["gen_time"].update((time.time() - start_time) / frames_per_batch, batch_size)

        loss = compute_loss(batch["labels"].long(), factored_logits)
        acc = (reshaped_input_ids[:, 1:].to("cuda") == samples).float().mean().item()

        perplexity = math.exp(loss)
        metrics["perplexity"].update(perplexity, batch_size)
        metrics["loss"].update(loss, batch_size)
        metrics["acc"].update(acc, batch_size)

        start_time = time.time()
        pred_frames = evaluator.predict_next_frames(samples)
        metrics["dec_time"].update((time.time() - start_time) / frames_per_batch, batch_size)

        if not args.use_tokenized_images: # key: use raw image as the groundtruth
            decoded_gtruth = batch['images'].permute(0, 1, 4, 2, 3).long().cpu().detach()
        else:
            decoded_gtruth = decode_tokens(reshaped_input_ids, decode_latents)

        metrics["pred_lpips"].update_list(compute_lpips(decoded_gtruth[:, 1:], pred_frames, lpips_alex))

        gt_frames_numpy = decoded_gtruth[:, 1:].detach().cpu().numpy()
        pred_frames_numpy = pred_frames.detach().cpu().numpy()

        psnr = [image_metrics.peak_signal_noise_ratio(
            gt_frames_numpy[i][-1] / 255., pred_frames_numpy[i][-1] / 255., data_range=1.0) for i in range(gt_frames_numpy.shape[0])]

        ssim = [np.mean([image_metrics.structural_similarity(
            gt_frames_numpy[i][j]  / 255., pred_frames_numpy[i][j]  / 255., data_range=1.0, channel_axis=0) \
            for i in range(gt_frames_numpy.shape[0])]) for j in range(gt_frames_numpy.shape[1])]
        metrics["ssim"].update_list(ssim)
        metrics["psnr"].update_list(psnr)

        # As in Genie. we also compute psnr_delta = PSNR(x_t, x_t_hat) - PSNR(x_t, x_t_hatprime) where x_t_hatprime samples random actions
        # this difference in PSNR measures the controllability
        # actions need to be just uniform random actions

        # for computing delta psnr. average over 5 to be more stable
        if with_action_input:
            psnr_delta_mean = np.zeros(gt_frames_numpy.shape[0])
            N_TRIALS = 5
            for _ in range(N_TRIALS):
                action_mean = evaluator.model.action_preprocessor[dataset].mean.repeat(stride)
                action_std = evaluator.model.action_preprocessor[dataset].std.repeat(stride)

                random_action_ids = torch.randn_like(batch["action_ids"]) * action_std + action_mean
                random_samples, _ = evaluator.predict_zframe_logits(batch["input_ids"].to(evaluator.device),
                                                                                            random_action_ids.to(evaluator.device),
                                                                                            [val_dataset.name],
                                                                                            skip_normalization=False)
                random_pred_frames = evaluator.predict_next_frames(random_samples)
                random_pred_frames_numpy = random_pred_frames.detach().cpu().numpy()

                # random subtracts groundtruth
                psnr_delta = [psnr[i] - image_metrics.peak_signal_noise_ratio(
                    gt_frames_numpy[i][-1] / 255., random_pred_frames_numpy[i][-1] / 255., data_range=1.0) for i in range(gt_frames_numpy.shape[0])]
                psnr_delta_mean += np.array(psnr_delta) / N_TRIALS
            metrics[f"psnr_delta"].update_list(psnr_delta_mean)

        if batch_idx > args.max_examples:
            break

    accelerator.print(f"=== dataset {dataset} model: {name}")

    # In case other processes don't have metrics due to batches, we need them to have all the keys so that they call reduce on zeros
    # metrics_keys = ['gen_time', 'perplexity', 'loss', 'acc', 'dec_time', 'pred_lpips', 'ssim', 'psnr', 'psnr_delta']
    # Could also have just hardcoded these but... better to spend the time to not hardcode :) jk I regret this so much

    if accelerator.is_main_process:
        metrics_keys = list(metrics.keys())
        keys_bytes = pickle.dumps(metrics_keys)
        keys_tensor = torch.tensor(list(keys_bytes), dtype=torch.uint8, device=accelerator.device)

        for rank in range(1, accelerator.num_processes):
            torch.distributed.send(torch.tensor([len(keys_tensor)], dtype=torch.int64, device=accelerator.device), dst=rank, tag=0)
            torch.distributed.send(keys_tensor, dst=rank, tag=1)
    else:
        num_bytes = torch.zeros(1, dtype=torch.int64, device=accelerator.device)
        torch.distributed.recv(num_bytes, src=0, tag=0)

        keys_tensor = torch.empty(num_bytes.item(), dtype=torch.uint8, device=accelerator.device)
        torch.distributed.recv(keys_tensor, src=0, tag=1)
        metrics_keys = pickle.loads(keys_tensor.cpu().numpy().tobytes())

    prefix = "teacher_force" if not args.autoregressive_time else "autoregressive"
    for key in metrics_keys:
        try:
            val = metrics[key]
            agg_total, agg_count = accelerator.reduce(
                torch.tensor([val.total, val.count], dtype=torch.float32, device=accelerator.device)
                # float is really important, otherwise 0 was diff dtype and silently hung
            )

            mean = agg_total / agg_count
            accelerator.print(f"{key}: {mean:.4f}")
            if accelerator.is_main_process:
                wandb.log({f"{dataset_original}/{prefix}_{key}": mean})
                # wandb.log({f"{prefix}_{key}": mean})

        except Exception as e:
            print(e)

    if accelerator.is_main_process:
        wandb.log({f"{dataset_original}/num_examples": len(val_dataset)})
        wandb.log({f"{dataset_original}/perturbation_scale": args.perturbation_scale})
        wandb.log({f"model_step": evaluator.model_step})
        # model training steps

        dataset_metadata = {
            f"{dataset_original}/dataset_name": f"{dataset}",
            f"{dataset_original}/num_examples": len(val_dataset),
            f"{dataset_original}/num_features": len(val_dataset[0]) if val_dataset else 0,
            f"{dataset_original}/sample_data": val_dataset[0] if len(val_dataset) > 0 else "N/A",
            f"{dataset_original}/model_step": evaluator.model_step
        }
        for k, v in dataset_metadata.items():
            wandb.run.summary[k] = v

        wandb.finish()

if __name__ == "__main__":
    main()
