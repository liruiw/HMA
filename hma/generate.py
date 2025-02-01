"""
Example Usage:
python hma/generate.py  --checkpoint_dir  data/kaist_model/step_50/  --val_data_dir data/kaist_nonprehensile_converted_externally_to_rlds_magvit_max1000000_val
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.append(os.getcwd())
from hma.data import RawTokenDataset
from hma.model.st_mask_git import STMaskGIT

from hma.data import RawFeatureDataset
from hma.model.st_mar import STMAR
from torch.utils.data import DataLoader
from einops import rearrange
import re
from transformers import default_data_collator

def parse_args():
    parser = argparse.ArgumentParser(description="Generates samples (as tokens) from GENIE model. "
                                                 "Optionally visualizes these tokens as GIFs or comics.")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/1x_humanoid_magvit_traj10_val",
        help="A directory with video data, should have a `metadata.json` and `video.bin` We generate using the first frames of this dataset."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/genie_generated",
        help="Directory to save generated outputs."
    )
    parser.add_argument(
        "--num_prompt_frames", type=int, default=4, help="The number of context frames."
    )
    parser.add_argument(
        "--window_size", type=int, default=12,
        help="Will generate `window_size - num_prompt_frames` frames."
    )
    parser.add_argument(
        "--example_ind", type=int, default=0,
        help="The index in the dataset of the example to generate on."
    )
    parser.add_argument(
        "--teacher_force_time", action="store_true",
        help="If True, teacher-forces generation in time dimension."
    )
    parser.add_argument(
        "--maskgit_steps", type=int, default=2, help="Number of MaskGIT sampling steps."
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature. If `temperature` <= 1e-8, will do greedy sampling."
    )
    parser.add_argument(
        "--add_action_input", action="store_true",
        help="If True, uses action in the video output."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size, current script only supports a single GPU."
    )
    parser.add_argument(
        "--max_example", type=int, default=16,
        help="Maximum number of examples."
    )
    parser.add_argument(
        "--use_feature", action="store_true",
        help="visualize the features rather than tokens"
    )
    return parser.parse_args()

def get_model_step(checkpoint_dir):
    if os.path.exists(f"{checkpoint_dir}/scheduler.bin"):
        sch = torch.load(f"{checkpoint_dir}/scheduler.bin")
        return sch['_step_count']
    return 0

def compute_stride_from_model(model, dataset):
    action_d = len(model.action_preprocessor[dataset].mean)
    action_d_horizon = model.config.d_actions[model.config.action_domains.index(dataset)]
    stride = action_d_horizon // action_d
    return stride

@torch.no_grad()
def main():
    args = parse_args()
    assert args.num_prompt_frames <= args.window_size

    if not os.path.exists(args.checkpoint_dir + "/config.json"):
        # search and find the latest modified checkpoint folder
        dirs = [os.path.join(args.checkpoint_dir, f.name) for f in os.scandir(args.checkpoint_dir) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        if len(dirs) == 0:
            print(f"No checkpoint directories found in {args.checkpoint_dir}")
            sys.exit(1)
        args.checkpoint_dir = dirs[-1]

    dataset = re.search(r"data/(.*?)_magvit", args.val_data_dir).group(1)
    # Load the model checkpoint
    if not args.use_feature:
        print(f"loading STMaskGIT")
        model = STMaskGIT.from_pretrained(args.checkpoint_dir).to("cuda")
        stride = compute_stride_from_model(model, dataset)
        val_dataset = RawTokenDataset(args.val_data_dir, window_size=args.window_size,
                                        compute_stride_from_freq_table=False,
                                        stride=stride,
                                        use_actions=model.config.use_actions)
    else:
        print(f"loading STMAR")
        model = STMAR.from_pretrained(args.checkpoint_dir).to("cuda")
        stride = compute_stride_from_model(model, dataset)
        args.val_data_dir = args.val_data_dir.replace("magvit", "vae")

        val_dataset = RawFeatureDataset(args.val_data_dir,
                                      compute_stride_from_freq_table=False,
                                      stride=stride, window_size=args.window_size,
                                        use_actions=model.config.use_actions)
        val_dataset.metadata["token_dtype"] = "float32"

    latent_side_len = val_dataset.data.shape[-1] # assume square
    dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    # Get single example
    if args.max_example > len(val_dataset):
        print(f"Example index {args.example_ind} is out of bounds for dataset of length {len(val_dataset)}")
        sys.exit(1)

    model.eval()
    output_list = []

    for batch_idx, batch in enumerate(dataloader):
        samples = []
        if args.use_feature:
            example_THW = rearrange(batch["input_ids"].to("cuda"), "b (t h w) c -> b t h w c", t=args.window_size,
                                h=latent_side_len, w=latent_side_len)
        else:
            example_THW = rearrange(batch["input_ids"].to("cuda"), "b (t h w) -> b t h w", t=args.window_size,
                                h=latent_side_len, w=latent_side_len)
        example_actions = None
        domain = None

        if model.config.use_actions:
            example_actions = batch["action_ids"].to("cuda")
            domain = [val_dataset.name.replace("_noquant", "")] * args.batch_size

        prompt_THW = example_THW.clone()
        prompt_THW[:, args.num_prompt_frames:] = model.mask_token if args.use_feature else model.mask_token_id

        for timestep in range(args.num_prompt_frames, args.window_size):
            # Teacher-forced, maskgit generation
            if args.teacher_force_time:
                prompt_THW = example_THW.clone()
                # Masked prediction for this timestep only, after which we provide ground-truth
                prompt_THW[:, timestep:] = model.mask_token if args.use_feature else model.mask_token_id

            samples_HW, _, _ = model.maskgit_generate(
                prompt_THW, out_t=timestep, temperature=args.temperature,
                action_ids=example_actions, domain=domain
            )

            samples.append(samples_HW)
            if not args.teacher_force_time:
                # autoregressive
                prompt_THW[:, timestep] = samples_HW


        outputs = torch.stack(samples, dim=1)
        # prepend prompt sequence
        outputs = torch.cat([example_THW[:, :args.num_prompt_frames], outputs], dim=1)

        # append ground-truth targets next to generated outputs for comic strip generation
        # [<prompt frames><predicted frames><ground truth frames>]
        outputs = torch.cat([outputs, example_THW[:, args.num_prompt_frames:]], dim=1)
        output_list.append(outputs)
        if batch_idx >= args.max_example // args.batch_size:
            break

    outputs = torch.cat(output_list, dim=0)
    if args.use_feature:
        # use chw
        outputs = rearrange(outputs, "b t h w c -> b t c h w")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs.cpu().numpy().astype(np.dtype(val_dataset.metadata["token_dtype"])).tofile(output_dir / "video.bin")
    print(f"Saved generated video to {output_dir / 'video.bin'} {outputs.shape}")
    model_steps = get_model_step(args.checkpoint_dir)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(vars(args) | val_dataset.metadata | {
            "num_images":  outputs.shape[1],
            "h": latent_side_len,
            "w": latent_side_len,
            "t": args.window_size,
            "model_checkpoint": args.checkpoint_dir,
            "dataset": val_dataset.name,
            "trained_steps": model_steps,
        }, f)


if __name__ == "__main__":
    main()
