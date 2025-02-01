"""
Example usage:
python -m hma.train_multi --genie_config hma/configs/magvit_n32_h8_d256_action.json \
    --output_dir data/$script_name \
    --max_eval_steps 10 \
    --num_episodes_per_dataset 1000000 \
    --per_device_train_batch_size 1 \
    --train_split experiments/datasplit/dataset1.yaml
# Description: This script is used to train the HMA model on multiple datasets.
"""
import argparse
import contextlib
import argparse
import logging
import math
import os
import time
from datetime import datetime

import matplotlib
import mup
import torch
import torchvision.transforms.functional as transforms_f
import wandb
import yaml
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from lpips import lpips
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
import traceback

from transformers import (
    default_data_collator,
    get_scheduler,
)
from collections import defaultdict
from external import data_sampler
from hma.data import RawTokenDataset, get_maskgit_collator, SVD_SCALE
from hma.eval_utils import decode_tokens, compute_lpips
from hma.model.st_mask_git import STMaskGIT
from hma.config import GenieConfig, DiffusionGenieConfig

from hma.visualize import decode_latents_wrapper
from skimage import metrics as image_metrics
from matplotlib import pyplot as plt
from hma.data import RawFeatureDataset, get_maskgit_collator_feature
from hma.model.st_mar import STMAR

# Get current date and time
now = datetime.now()

# Format the datetime object as a string
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
torch.set_float32_matmul_precision("medium")
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a HMA model on dynamic generation.")

    # Data
    parser.add_argument(
        "--train_data_dir", type=str, default="data/kaist_nonprehensile_converted_externally_to_rlds_magvit_max1000000_train",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--val_data_dir", type=str, default="data/kaist_nonprehensile_converted_externally_to_rlds_magvit_max1000000_val",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--domain", type=str, default="kaist_nonprehensile_converted_externally_to_rlds",
        help="The domain name for the dataset"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=12,
        help="Number of frames to in a sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Difference in frame count between consecutive frames in a sequence.",
    )
    parser.add_argument(
        "--filter_overlaps",
        action="store_true",
        help=(
            "Whether to filter repeated frames in the train dataset (`filter_overlaps` always true for the val set). "
            "Filtering essentially makes the training dataset less correlated but ~16x smaller, "
            "see the `filter_overlaps` argument in `RawTokenDataset` for details."),
        default=True
    )

    # Model
    parser.add_argument(
        "--llama_config",
        type=str,
        help="`transformers.LlamaConfig` json. "
             "E.g. https://huggingface.co/1x-technologies/Llama_1B_v0/blob/main/config.json",
    )

    parser.add_argument(
        "--genie_config",
        type=str,
        help="GenieConfig json.",
        default="hma/configs/magvit_n32_h8_d256_action.json"
    ),

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # Training
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=int(1e10),
        help="Only evaluate on `max_eval_steps` batches of validation data per process, faster.",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=1000,
        help="Eval every N training steps.",
    )
    parser.add_argument(
        "--train_split", type=str, default="experiments/datasplit/dataset1.yaml",
        help="Config files for using multiple datasets."
    )

    parser.add_argument(
        "--num_episodes_per_dataset",
        type=int,
        default=1000000,
        help="Maximum number of trajectories per dataset",
    )

    parser.add_argument(
        "--vis_every_n_steps",
        type=int,
        default=20000,
        help="Visualize every N training steps.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "custom_cosine"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Threshold to clip gradients.",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.05,
        help="Attention dropout prob.",
    )
    parser.add_argument(
        "--adam_beta_1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta_2",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
    )

    # Misc
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the model checkpoints.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="10000",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--overfit_first_batch",
        action="store_true",
        help=(
            "Debug option that trains and validates on only the first batch of the training dataset."
        ),
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help=(
            "Whether to pin memory in the dataloaders. "
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--mu_transfer",
        action="store_true",
        help="If specified, will train with mu transfer reparametrizations. Only supports Llama models.",
        default=True
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="If specified, will not compile the model.",
        default=True
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="video_prediction",
        help="run name",
    )
    parser.add_argument(
        "--save_second_epoch",
        action="store_true",
        help="Whether to checkpoint at the end of the second epoch (1-indexing). This one will not be auto-deleted by cleanup.",
        default=True
    )

    # Training
    parser.add_argument(
        "--model_type",  # TODO: decide on naming
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
    )
    parser.add_argument(
        "--use_raw_image_as_latent",
        action="store_true",
        # TODO: help, mention only supported for diffusion
    )
    parser.add_argument(
        "--action_network",
        type=str,
        default=None,
        choices=["concat", "cross_attention"],  # TODO: add other methods (resampler_concat, modulate, etc)
        help="If specified, will override the action in the config. Helps reduce the number of config jsons."
    )
    args = parser.parse_args()

    # Sanity checks
    # TODO: more checks if needed
    return args


def save_checkpoint(model, accelerator, args, filename):
    """
    filename: `save_path = os.path.join(args.output_dir, filename)`
    """
    unwrapped_model = accelerator.unwrap_model(model)
    save_path = os.path.join(args.output_dir, filename)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        accelerator.save_state(save_path)


@torch.no_grad()
def visualize(accelerator, model, dataloader, window_size, metrics_prefix="train", max_steps=1):
    """
    Visualizes model's autoregressive generation outputs, logged to wandb.
    It uses teacher-forcing (causal in time axis)
    """

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if not unwrapped_model.config.jointly_predict_states:
        return
    metrics = defaultdict(list)
    if accelerator.is_main_process:
        lpips_alex = lpips.LPIPS(net="alex")  # Calculate LPIPS w/ AlexNet, the fastest option

    decode_latents = decode_latents_wrapper()  # re-initializing every time to save memory
    unwrapped_model.eval()
    rank = 0
    dataloader_iter = iter(dataloader)
    for step in range(len(dataloader)):
        try:
            batch = next(dataloader_iter)

            # Note: hardcoding 4 image cap for faster inference on small models
            TEST_NUM = 4
            reshaped_labels = rearrange(batch["labels"][:TEST_NUM], "b (t s) -> b t s", t=window_size).to(accelerator.device)  # `s` is really `(h, w)`
            domains = batch["domain"][:TEST_NUM]

            if 'action_ids' in batch:
                action_ids = batch["action_ids"][:TEST_NUM].to(accelerator.device)
            else:
                action_ids = None

            # hardcoding half of frames for context
            num_prompt_frames = unwrapped_model.config.num_prompt_frames
            num_new_tokens = batch["w"][0] * batch["h"][0] * (window_size - num_prompt_frames)
            prompt_input_ids = rearrange(reshaped_labels[:, :num_prompt_frames], "b t s -> b (t s)")
            outputs = unwrapped_model.generate(input_ids=prompt_input_ids, attention_mask=torch.ones_like(prompt_input_ids),
                                                max_new_tokens=num_new_tokens, min_new_tokens=num_new_tokens,
                                                action_ids=action_ids,
                                                domain=batch["domain"][:TEST_NUM],
                                                w=batch["w"][:TEST_NUM],
                                                h=batch["h"][:TEST_NUM])

            output_tokens = rearrange(outputs, "b (t h w) -> b t h w", t=window_size,
                                    h=batch["h"][0], w=batch["w"][0])
            gtruth_tokens = rearrange(reshaped_labels[:, num_prompt_frames:], "b t (h w) -> b t h w",
                                    h=batch["h"][0], w=batch["w"][0])

            decoded_output = decode_tokens(output_tokens.cpu(), decode_latents)
            decoded_gtruth = decode_tokens(gtruth_tokens.cpu(), decode_latents)

            decoded_output = accelerator.gather(decoded_output.to(accelerator.device)).cpu()
            decoded_gtruth = accelerator.gather(decoded_gtruth.to(accelerator.device)).cpu()

            # As in Genie. we also compute psnr_delta = PSNR(x_t, x_t_hat) - PSNR(x_t, x_t_hatprime) where x_t_hatprime samples random actions
            # this difference in PSNR measures the controllability
            # actions need to be just uniform random actions
            if action_ids is not None:
                random_action_ids = torch.randn_like(action_ids)
                random_action_outputs = unwrapped_model.generate(input_ids=prompt_input_ids, attention_mask=torch.ones_like(prompt_input_ids),
                                                    max_new_tokens=num_new_tokens, min_new_tokens=num_new_tokens,
                                                    action_ids=random_action_ids,
                                                    domain=batch["domain"][:TEST_NUM],
                                                    w=batch["w"][:TEST_NUM],
                                                    h=batch["h"][:TEST_NUM],
                                                    skip_normalization=True)

                random_output_tokens = rearrange(random_action_outputs, "b (t h w) -> b t h w", t=window_size,
                                        h=batch["h"][0], w=batch["w"][0])
                random_output_tokens = decode_tokens(random_output_tokens.cpu(), decode_latents)

                random_output_tokens = accelerator.gather(random_output_tokens.to(accelerator.device)).cpu()
                random_pred_frames_numpy = random_output_tokens[:, num_prompt_frames:].detach().cpu().numpy()


            if accelerator.is_main_process:
                exs_per_fig = 4

                for j in range(0, len(decoded_output), exs_per_fig):
                    fig, axs = plt.subplots(nrows=2 * exs_per_fig, ncols=window_size, figsize=(3 * window_size, 3 * 2 * exs_per_fig))
                    # If len(decoded_output) is not a multiple of 4, make sure to truncate properly
                    for k in range(min(exs_per_fig, len(decoded_output) - j)):
                        for i in range(num_prompt_frames):
                            for ax in (axs[k * 2, i], axs[k * 2 + 1, i]):
                                ax.imshow(transforms_f.to_pil_image(decoded_output[j + k, i]))
                                ax.set_title("Context")
                                ax.axis("off")

                        for i in range(num_prompt_frames, window_size):
                            axs[k * 2, i].imshow(transforms_f.to_pil_image(decoded_gtruth[j + k, i - num_prompt_frames]))
                            axs[k * 2, i].set_title("Ground truth")
                            axs[k * 2 + 1, i].imshow(transforms_f.to_pil_image(decoded_output[j + k, i]))
                            axs[k * 2 + 1, i].set_title("Prediction")
                            for ax in axs[:, i]:
                                ax.axis("off")

                    rank = accelerator.process_index
                    wandb_tracker = accelerator.get_tracker("wandb")
                    # wandb_tracker.log({f"vis_{metrics_prefix}_{j}": fig}, commit=False)
                    wandb_tracker.log({f"{domains[0]}/vis_{metrics_prefix}_{j}": fig}, commit=False)
                    plt.close(fig)

                metrics["ar_lpips"].extend(compute_lpips(decoded_gtruth,  # Note: not parallelizing right now
                                                        decoded_output[:, num_prompt_frames:], lpips_alex))

                gt_frames_numpy = decoded_gtruth.detach().cpu().numpy()
                pred_frames_numpy = decoded_output[:, num_prompt_frames:].detach().cpu().numpy()
                psnr = [image_metrics.peak_signal_noise_ratio(
                    gt_frames_numpy[i] / 255., pred_frames_numpy[i] / 255., data_range=1.0) for i in range(gt_frames_numpy.shape[0])]

                ssim = [np.mean([image_metrics.structural_similarity(
                    gt_frames_numpy[i][j]  / 255., pred_frames_numpy[i][j] / 255., data_range=1.0, channel_axis=0) \
                    for i in range(gt_frames_numpy.shape[0])]) for j in range(gt_frames_numpy.shape[1])]

                # compute some other metrics
                metrics[f"{metrics_prefix}/ar_psnr"].extend(psnr)
                metrics[f"{metrics_prefix}/ar_ssim"].extend(ssim)
                metrics[f"{batch['domain'][0]}/ar_lpips"].extend(compute_lpips(decoded_gtruth,  # Note: not parallelizing right now
                                                                    decoded_output[:, num_prompt_frames:], lpips_alex))

                if action_ids is not None:
                    # log controllability as random subtracts groundtruth
                    psnr_delta = [psnr[i] - image_metrics.peak_signal_noise_ratio(
                        gt_frames_numpy[i] / 255., random_pred_frames_numpy[i] / 255., data_range=1.0) for i in range(gt_frames_numpy.shape[0])]

                    metrics[f"{metrics_prefix}/ar_psnr_delta"].extend(psnr_delta)

        except Exception as e:
            print("batch failed", traceback.format_exc())

        if step + 1 >= max_steps:
            break

    unwrapped_model.train()
    if accelerator.is_main_process:
        metrics = {f"{metrics_prefix}_{key}": np.mean(val) for key, val in metrics.items() if len(val) > 0}

        print(f"{metrics=}")
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.log(metrics, commit=False)

def train(accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, experiment_config, config, args):
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None
    checkpoint_path = ""

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        try:
            if  os.path.exists(args.resume_from_checkpoint + "/pytorch_model.bin"):
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
            # else:
            #     checkpoint_path = args.resume_from_checkpoint
            #     path = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
            else:
                # Get the most recent checkpoint
                base_path = os.path.dirname(args.resume_from_checkpoint)
                dirs = [os.path.join(base_path, f.name) for f in os.scandir(base_path) if f.is_dir()]
                dirs.sort(key=os.path.getctime)

                # Sorts folders by date modified, most recent checkpoint is the last
                if len(dirs) > 0:
                    path = dirs[-1]
                    checkpoint_path = path
                    path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

            if os.path.exists(checkpoint_path):
                # for finetuning with a different structures
                print(f"loading checkpoint from {checkpoint_path}")
                accelerator.load_state(checkpoint_path, strict=False)
                # tied weights not saved so can't load strict, but also no need to tie again
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]
            else:
                print("No checkpoint found, training from scratch.")
                training_difference = "step_0"

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

        except Exception as e:
            training_difference = "step_0"
            starting_epoch = 0
            completed_steps = 0
            print("load checkpoint incomplete", traceback.format_exc())

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    loss_info = torch.zeros(2, device=accelerator.device)  # sum, count

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        _time = time.time()
        dataloader_iter = iter(active_dataloader)

        # Switch back to train mode
        model.train()
        num_iters_per_epoch = max(len(active_dataloader) - 8, 1) # avoid the last few iters

        for step in range(num_iters_per_epoch):
            try:
                train_action_loss = 0
                batch = next(dataloader_iter)
                # to reduce the numerical instability in the very beginning of training
                gradient_accumulation_steps = args.gradient_accumulation_steps
                batch_size = batch["input_ids"].size(0)
                # Manual gradient accumulation because accelerator somehow taking a lot of memory
                is_update_step = (step + 1) % gradient_accumulation_steps == 0
                ctx_manager = contextlib.nullcontext() if is_update_step else accelerator.no_sync(model)

                with ctx_manager:
                    accelerator.wait_for_everyone()
                    outputs = model(**batch)
                    loss = outputs.loss

                    if not torch.isnan(loss).any():
                        loss_info[0] += loss.detach().mean() * batch_size # only video loss
                        if "action_loss" in outputs:
                            train_action_loss = outputs.action_loss.item()
                            loss += config.action_loss_weight * outputs.action_loss

                        loss_info[1] += batch_size
                        accelerator.backward(loss / gradient_accumulation_steps)
                    else:
                        print("Warning: NaN or Inf detected in loss. Skipping backward pass.")
                        dummy_loss = torch.zeros_like(loss, requires_grad=True)
                        accelerator.backward(dummy_loss)

                if not is_update_step:
                    continue

            except Exception as e:
                # avoid final iteration batch concatenation problems
                print("batch failed",  traceback.format_exc())
                continue
            # Everything below only happens on update step
            if args.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_info = accelerator.reduce(loss_info)

            avg_train_loss = (loss_info[0] / loss_info[1]).item()  # sum / count
            loss_info *= 0  # reset sum and count
            try:
                perplexity = math.exp(avg_train_loss)
            except OverflowError:
                print("overflow error for perplexity")
                perplexity = float("inf")

            # print(f"{perplexity=} {avg_train_loss=}")
            batch_time = time.time() - _time  # accumulated batch
            rank = accelerator.process_index

            domain_iter = str(batch['domain'][0])
            _time = time.time()
            accelerator.log(
                {
                    "train_perplexity": perplexity,
                    "train_loss": avg_train_loss,
                    "train_action_loss": train_action_loss,
                    f"stat/{domain_iter}_action_loss": train_action_loss / loss_info[1],
                    f"stat/{domain_iter}_train_perplexity": perplexity,
                    f"stat/{domain_iter}_train_loss": avg_train_loss,
                    "epoch": epoch,
                    "update_step": completed_steps,
                    "examples_processed": completed_steps * args.per_device_train_batch_size
                                          * args.gradient_accumulation_steps * accelerator.num_processes,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "flops": (completed_steps + 1) * experiment_config["FLOPs_per_update_step"],
                    "throughput_examples": experiment_config["effective_batch_size"] / batch_time,
                }, step=completed_steps)

            progress_bar.update(1)
            completed_steps += 1


            # print(f"{completed_steps %  args.checkpointing_steps=} {completed_steps=} {args.checkpointing_steps=}")
            if  completed_steps % int(args.checkpointing_steps) == 0:
                print(f"Saving checkpoint at step {completed_steps}!")
                save_checkpoint(model, accelerator, args, f"step_{completed_steps}")

            if completed_steps % args.eval_every_n_steps == 0:
                time.sleep(1) # manual adding time sleep
                model.eval()
                eval_losses = []

                # Compute token-level accuracy (w/ teacher forcing)
                num_correct = 0
                num_total = 0

                # barrier

                # to resolve the data collating issues
                eval_dataloader_iter = iter(eval_dataloader)
                for step in range(args.max_eval_steps):
                    eval_action_loss = 0
                    try:
                        batch = next(eval_dataloader_iter)
                        batch_size = len(batch["input_ids"])  # Last batch might not be full
                        with torch.no_grad():
                            outputs = model(**batch)

                        loss = outputs.loss
                        if "action_loss" in outputs:
                            eval_action_loss = outputs.action_loss.item()
                        eval_losses.append(accelerator.gather_for_metrics(loss.repeat(batch_size)))
                    except Exception as e:
                        print("error:", e)
                        continue

                    if "acc" in outputs:
                        # `num_correct` and `num_total` actually track mean accuracy in this case.
                        num_correct_batch = accelerator.reduce(outputs.acc, reduction="mean").item() * batch_size
                        num_total_batch = batch_size
                        num_correct += num_correct_batch
                        num_total += num_total_batch
                    else:
                        shifted_preds = torch.argmax(outputs.logits[:, :-1, :], dim=-1)
                        shifted_labels = batch["labels"][:, 1:]
                        num_correct_batch = accelerator.gather_for_metrics((shifted_preds == shifted_labels).sum()).sum().item()
                        num_total_batch = accelerator.gather_for_metrics(torch.tensor(torch.numel(shifted_labels),
                                                                device=accelerator.device)).sum().item()
                        num_correct += num_correct_batch
                        num_total += num_total_batch

                    if step >= args.max_eval_steps * args.num_datasets:
                        break

                    try:
                        accelerator.log(
                        {
                            f'stat/{domain_iter}_eval_teacher_acc': num_correct_batch / num_total_batch,
                            f'stat/{domain_iter}_eval_loss': (torch.mean(eval_losses[-1])).item(),
                            f'stat/{domain_iter}_eval_action_loss': eval_action_loss,

                        },
                        step=completed_steps,
                        )
                    except Exception as e:
                        print("log failed", e)
                        continue

                if len(eval_losses) > 0:
                    eval_losses = torch.cat(eval_losses)
                    eval_loss = torch.mean(eval_losses).item()
                    eval_teacher_acc = num_correct / num_total
                    try:
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        print("overflow error for perplexity")
                        perplexity = float("inf")
                else:
                    continue

                logger.info(f"{completed_steps=} {perplexity=} {eval_loss=} {eval_teacher_acc=}")
                accelerator.log(
                    {
                        "eval_perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "eval_action_loss": eval_action_loss,
                        "eval_teacher_acc": eval_teacher_acc,
                        "epoch": epoch,
                        "update_step": completed_steps,
                        "examples_processed": completed_steps * args.per_device_train_batch_size
                                              * args.gradient_accumulation_steps * accelerator.num_processes,
                        "flops": completed_steps * experiment_config["FLOPs_per_update_step"],
                    },
                    step=completed_steps,
                )

            if completed_steps % args.vis_every_n_steps == 0 or completed_steps >= args.max_train_steps:
                if "encoder_type" not in experiment_config:
                    experiment_config["encoder_name_or_path"] = "data/magvit2.ckpt"
                    experiment_config["encoder_type"] = "magvit"

                if not args.overfit_first_batch:  # val is same as train otherwise
                    visualize(accelerator, model, eval_dataloader, args.window_size, "val")

                visualize(accelerator, model, train_dataloader, args.window_size, "train")

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch" or (args.save_second_epoch and epoch == 1):
            save_checkpoint(model, accelerator, args, f"epoch_{epoch}")

    save_checkpoint(model, accelerator, args, f"final_checkpt")
    accelerator.end_training()





def main():
    args = parse_args()

    # Set all the differences between discrete/continuous training
    if args.model_type == "discrete":
        config_cls = GenieConfig
        model_cls = STMaskGIT

        # data
        dataset_cls = RawTokenDataset
        get_collator = get_maskgit_collator
        # TODO: explain this expected format somewhere in args or README
        data_path_format = "data/{}_magvit_max1000000_{}"  # domain, split
        shared_keys = ("s", "h", "w", "vocab_size")
    else:
        config_cls = DiffusionGenieConfig
        model_cls = STMAR

        # data
        dataset_cls = RawFeatureDataset
        get_collator = get_maskgit_collator_feature
        data_path_format = "data/{}_vae_max1000000_{}"  # domain, split
        shared_keys = ("s", "h", "w", "latent_channels",  # omitting "vocab_size" effectively makes it None
                   "encoder_type", "encoder_name_or_path", "quantized")

    # Manual gradient accumulation
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=1, log_with=args.report_to,
                                even_batches=False, project_dir=args.output_dir, kwargs_handlers=[ddp_kwargs])
    accelerator.init_trackers("video")

    if accelerator.is_main_process:
        accelerator.trackers[0].run.name = f"{formatted_date}_{args.run_name}"

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # create multiple datasets
    with open(args.train_split, 'r') as file:
        datasplit = yaml.safe_load(file)

    config = config_cls.from_pretrained(args.genie_config)
    if args.model_type == "continuous" and config.drop_action_ratio > 0:
        raise NotImplementedError

    # Extract the 'domains' value and split it into a list
    domains_list = [domain.strip() for domain in datasplit['domains'].split(',')]
    train_datasets = []
    val_datasets = []
    dataset_num_samples = []
    val_dataset_num_samples = []

    action_dimensions = []
    action_stats = []
    total_num_videos = 0

    dataset_kwargs = {
        "window_size": args.window_size,
        "stride": args.stride,
        "max_traj_num": args.num_episodes_per_dataset,
        "use_actions": config.use_actions,
    }

    for domain in domains_list:
        train_data_dir = data_path_format.format(domain, "train")
        val_data_dir = data_path_format.format(domain, "val")

        if args.model_type == "discrete":  # dropping actions only supported w/ CE loss
            dataset_kwargs["drop_action_ratio"] = config.drop_action_ratio
            dataset_kwargs["name"] = domain  # TODO: rename name -> domain
        else:
            dataset_kwargs["domain"] = domain

        train_dataset = dataset_cls(train_data_dir, filter_overlaps=args.filter_overlaps, **dataset_kwargs)
        dataset_num_samples.append(len(train_dataset))
        action_dimensions.append(train_dataset.n_action)
        total_num_videos += train_dataset.num_videos

        if config.use_actions:
            action_stats.append(train_dataset.action_stat)

        if not args.overfit_first_batch:
            eval_dataset = dataset_cls(val_data_dir, filter_overlaps=True, **dataset_kwargs)
        else:
            train_dataset.valid_start_inds = train_dataset.valid_start_inds[:args.per_device_train_batch_size
                                                                            * args.gradient_accumulation_steps
                                                                            * accelerator.num_processes]
            eval_dataset = train_dataset

        # Shuffle eval dataset and then set shuffle=False on the dataloader.
        # Shuffling in the dataloader results in reshuffling with each iteration.
        eval_dataset.valid_start_inds = torch.tensor(eval_dataset.valid_start_inds)[
            torch.randperm(len(eval_dataset), generator=torch.Generator().manual_seed(0))
        ].tolist()
        val_dataset_num_samples.append(len(eval_dataset))

        train_datasets.append(train_dataset)
        val_datasets.append(eval_dataset)
        assert all(train_dataset.metadata.get(shared_key) == eval_dataset.metadata.get(shared_key)
                   for shared_key in shared_keys)  # TODO: check this across all datasets

    print("dataset_num_samples:", dataset_num_samples)

    # Will not store key in metadata if it's missing, so that defaults can be filled by functions later?  # TODO: handle missing keys
    shared_metadata = {shared_key: train_dataset.metadata[shared_key]
                       for shared_key in shared_keys if shared_key in train_dataset.metadata}

    config.use_mup = args.mu_transfer  # Note: changing this may affect pre-trained model due to attn scaling
    config.image_vocab_size = shared_metadata.get("vocab_size", None)
    config.T = args.window_size
    config.S = shared_metadata["h"] * shared_metadata["w"]  # TODO: make STMaskGIT use h and w instead of S

    if args.model_type == "continuous":
        config.vae_embed_dim = shared_metadata["latent_channels"]

    if args.action_network is not None:
        print("Using action network", args.action_network)
        config.action_network = args.action_network

    model = model_cls(config)

    if config.use_actions:
        # TODO: use new list instead of domains_list, in case domain fails
        model.init_action_projectors(domains_list, action_dimensions, action_stats, config.action_network)

    if args.mu_transfer:
        model.set_mup_shapes(rescale_params=True)

    # Optimizer. Split weights in two groups, one with weight decay and the other not.
    opt_class = mup.MuAdamW if args.mu_transfer else torch.optim.AdamW
    # scale base learning rate
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps \
                           * accelerator.num_processes
    args.learning_rate = args.learning_rate * min(max(1, effective_batch_size / 64), 8)

    # Optimizer. Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    optimizer = opt_class(optimizer_grouped_parameters, lr=args.learning_rate,
                          betas=(args.adam_beta_1, args.adam_beta_2), eps=args.adam_eps)

    # DataLoaders creation:
    collate_fn = get_collator(config)
    combined_dataset = torch.utils.data.ConcatDataset(train_datasets)

    batch_sampler = data_sampler.MultiTaskBatchSampler(
        dataset_num_samples,
        batch_size=args.per_device_train_batch_size,
        temperature=3.0  # the higher, the flatter the distribution
    )
    dataset_traj_image = data_sampler.make_dataset_pie_plot(domains_list, dataset_num_samples)
    accelerator.log(({"dataset_mixture": wandb.Image(dataset_traj_image)}), log_kwargs={"wandb": {"commit": False}})
    dataset_weights = batch_sampler.generate_tasks_distribution().cpu().numpy()
    dataset_weight_image = data_sampler.make_dataset_pie_plot(domains_list, dataset_weights)
    accelerator.log(({"dataset_mixture_weight": wandb.Image(dataset_weight_image)}), log_kwargs={"wandb": {"commit": False}})

    train_dataloader = DataLoader(combined_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                   num_workers=args.num_workers, pin_memory=args.pin_memory)

    batch_val_sampler = data_sampler.MultiTaskBatchSampler(
        val_dataset_num_samples,
        batch_size=args.per_device_train_batch_size,
        temperature=4. # the higher the more flat the distribution
    )

    combined_val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    eval_dataloader = DataLoader(combined_val_dataset, batch_sampler=batch_val_sampler, collate_fn=collate_fn,
                                    num_workers=args.num_workers, pin_memory=args.pin_memory)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        if args.max_train_steps < 2000 and args.resume_from_checkpoint is None: # minimal number of trainng steps
            args.max_train_steps = 2000
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.lr_scheduler_type == "custom_cosine":  # decay to `end_ratio` of the peak learning rate
        def get_lr_wrapper(warmup_steps, max_steps, end_ratio=0.1):
            def get_lr(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps

                remaining_steps = max_steps - warmup_steps
                return ((1 + math.cos(math.pi * (step - warmup_steps) / remaining_steps)) / 2) \
                    * (1 - end_ratio) + end_ratio
            return get_lr

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, get_lr_wrapper(args.num_warmup_steps * accelerator.num_processes,
                                      args.max_train_steps if overrode_max_train_steps
                                      else args.max_train_steps * accelerator.num_processes)
        )
    else:
        lr_scheduler = transformers.get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        )

    # Prepare everything with our `accelerator`.
    accelerator.wait_for_everyone()
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if not args.no_compile:
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.optimize_ddp = False  # https://github.com/pytorch/pytorch/issues/104674
        # TODO: https://github.com/pytorch/pytorch/issues/109774#issuecomment-2046633776
        model = torch.compile(model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    experiment_config = vars(args) | vars(config)

    seq_len = shared_metadata["h"] * shared_metadata["w"] * args.window_size
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps \
                           * accelerator.num_processes
    args.num_datasets = len(train_datasets)
    model_module = model.module if hasattr(model, "module") else model

    experiment_config.update(shared_metadata | {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_parameters_M": round(sum(p.numel() for p in model.parameters()) / 1e6),
        "trunk_parameters": sum(p.numel() for p in model_module.decoder.parameters()),
        "trunk_parameters_M": round(sum(p.numel() for p in model_module.decoder.parameters()) / 1e6),
        "seq_len": seq_len,
        "train_data_tokens": len(train_dataset) * seq_len,
        "effective_batch_size": effective_batch_size,
        "effective_batch_size_tokens": effective_batch_size * seq_len,
        "mixed_precision": accelerator.mixed_precision,
        "num_datasets": args.num_datasets,
        "total_num_videos": total_num_videos,
    })

    experiment_config["FLOPs_per_update_step"] = 6 * experiment_config["model_parameters"] \
                                                 * experiment_config["effective_batch_size_tokens"]

    accelerator.init_trackers(project_name="video", config=experiment_config)

    # Train!
    train(accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, experiment_config, config, args)


if __name__ == "__main__":
    main()
