# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import argparse
import json
import os
import time
import traceback
from typing import Optional

import numpy as np
from tqdm import tqdm

from datasets.encode_openx_dataset import (
    MIN_VAL_EXAMPLES,
    MAX_VAL_EXAMPLES,
    get_shard_inds,
    VAL_RATIO,
    process_dataset_step,
    DATA_FREQ_TABLE,
)
from datasets.extern.ego4d import ego4d_dataset_size, ego4d_dataset_generator
from datasets.extern.egoexo4d import egoexo4d_dataset_size, egoexo4d_dataset_generator
from datasets.extern.robomimic import robomimic_dataset_generator, robomimic_dataset_size
from . import utils


SCRIPT_DESCRIPTION = """
Similar to encode_openx_dataset.py except for non-OpenX datasets.
Again, each split can be partitioned into multiple shards,
which is useful for parallelized encoding across GPUs.

Example usage:
    CUDA_VISIBLE_DEVICES=0 python -m datasets.encode_extern_dataset --dataset_name egoexo4d --data_split train --num_shards 1000 --curr_shard_rank 400

Untested usage (SVD tokenizer):
CUDA_VISIBLE_DEVICES=0 python -m datasets.encode_extern_dataset --dataset_name robomimic --data_split val --no_quantization --encoder_type temporalvae --encoder_name_or_path 'stabilityai/stable-video-diffusion-img2vid'
""".strip()

DATASET_TO_GEN_AND_SIZE = {
    "ego4d": (ego4d_dataset_generator, ego4d_dataset_size),
    "egoexo4d": (egoexo4d_dataset_generator, egoexo4d_dataset_size),
    "robomimic": (robomimic_dataset_generator, robomimic_dataset_size),
}


def encode_dataset_split(
    extern_dataset_name: str,
    split: str,
    max_episodes: Optional[int],
    original_res: bool,
    no_quantization: bool,
    curr_shard_rank: int,
    num_shards: int,
    root_dir: str,
    encoder_type: str,
    encoder_name_or_path: str,
    dataset_postfix: str = "",
    no_encoding: bool = False,
):
    """
    Encodes (e.g. tokenizes) dataset.
    The data written to disk can be used to load a `RawTokenDataset` (or the continuous version.)

    Args:
        extern_dataset_name:  TODO
        split: expected to be either "train" or "val". TODO: decide how to split
        max_episodes: the maximum number of trajectories to include in the dataset.
        dataset_postfix: will be a suffix of the output dirname.
        image_encoder: string specifying the type of image encoder/tokenizer to use.
        original_res: if True, will maintain original resolution of the video rather than resizing it to 256x256.
        no_quantization: if True, will not perform quantization step in image encoder.
    """
    extern_dataset_name = extern_dataset_name.strip()  # never modified
    suffixed_dataset_name = extern_dataset_name  # will modify later

    if original_res:
        suffixed_dataset_name = f"{suffixed_dataset_name}_originalres"
    if no_quantization:
        suffixed_dataset_name = f"{suffixed_dataset_name}_noquant"
    if no_encoding:
        suffixed_dataset_name = f"{suffixed_dataset_name}_noencoding"
    save_dirname = "_".join([suffixed_dataset_name, encoder_type, dataset_postfix, split])
    dataset_path = os.path.join(root_dir, save_dirname)
    print("=" * 25)
    print(f"{dataset_path=}")
    utils.mkdir_if_missing(dataset_path)

    # Load data
    generator, size_func = DATASET_TO_GEN_AND_SIZE[extern_dataset_name]
    num_examples = size_func()
    if max_episodes is not None:
        num_examples = min(num_examples, max_episodes)  # clip num_examples

    # We will only operate on a subset of the training examples, depending on:
    #      1) The split (train/val). Some examples are reserved for the other split.
    #      2) Sharding
    assert num_examples > MIN_VAL_EXAMPLES  # non-positive number of train examples otherwise
    num_val_examples = np.clip(int(VAL_RATIO * num_examples), MIN_VAL_EXAMPLES, MAX_VAL_EXAMPLES)

    if split == "train":  # first_ind inclusive, last_ind exclusive
        first_split_ind, last_split_ind = num_val_examples, num_examples
    elif split == "val":
        first_split_ind, last_split_ind = 0, num_val_examples
    else:
        raise NotImplementedError(f"{split=}")

    first_shard_ind, last_shard_ind = get_shard_inds(first_split_ind, last_split_ind, curr_shard_rank, num_shards)
    print(f"Total number of examples in {suffixed_dataset_name}: {num_examples}")
    print(
        f"Number of examples for {split=}, shard {curr_shard_rank} of {num_shards}: "
        f"{last_shard_ind - first_shard_ind}. {first_shard_ind=} {last_shard_ind=}"
    )

    ##### Encode data #####
    traj_lens = []  # only used to print statistics
    videos = []  # NOTE: videos/actions for the entire shard are stored in RAM until the end
    actions = []
    segment_ids = []

    # split based on some fixed batch sizes to reset RAM.
    max_batch_per_loading = 10
    pbar = tqdm(range(first_shard_ind, last_shard_ind, max_batch_per_loading), position=0, leave=True)
    start_time = time.time()

    for start_idx in pbar:
        end_idx = min(start_idx + max_batch_per_loading, last_shard_ind)
        pbar.set_description(f"{suffixed_dataset_name} caching episodes: {start_idx}:{end_idx}")
        ds = generator(range(start_idx, end_idx))

        for chunk_idx, episode in enumerate(tqdm(ds, position=1, leave=False)):
            segment_id = start_idx + chunk_idx
            try:
                # batchify the data and then process
                for step_ind, step_data in enumerate(episode["steps"]):
                    dataset_step = process_dataset_step(
                        step_data,
                        encoder_type=encoder_type,
                        encoder_name_or_path=encoder_name_or_path,
                        keep_res=original_res,
                        quantize=not no_quantization,
                        no_encoding=no_encoding,
                    )

                    segment_ids.append(segment_id)
                    videos.append(dataset_step["image"])
                    actions.append(dataset_step["action"])

                traj_lens.append(step_ind + 1)  # number of steps in this trajectory
            except:
                print("-" * 25)
                print(f"Add episode failed: {segment_id=}", traceback.format_exc(), suffixed_dataset_name)

            # 2 day timeout
            if time.time() - start_time > 86400 * 2:
                print(f"Writing dataset {suffixed_dataset_name} timed out")
                break

    if len(videos) == 0:
        print("Empty shard!")
        with open(f"{dataset_path}/error.json", "w") as f:
            json.dump({"status": "empty_shard"}, f)

        return

    if no_quantization:
        num_channels, height, width = videos[-1].shape[:3]  # num_channels is not actually stored in metadata
    else:
        height, width = videos[-1].shape[:2]
        num_channels = None

    ##### Write videos, actions, segment_ids, and metadata #####
    # align format to save segment_ids.bin, video.bin, actions/action.bin, metadata.json
    # save videos
    videos = np.stack(videos, axis=0)
    # fp = np.memmap(f'{dataset_path}/video.bin', dtype=video_dtype, mode='w+', shape=videos.shape)
    # fp[:] = videos[:]
    videos.tofile(f"{dataset_path}/video.bin")

    # save action
    utils.mkdir_if_missing(f"{dataset_path}/actions")
    actions = np.stack(actions, axis=0)
    # fp = np.memmap(f'{dataset_path}/actions/actions.bin', dtype=np.float32, mode='w+', shape=actions.shape)
    # fp[:] = actions[:]
    actions = actions.astype(np.float32)
    actions.tofile(f"{dataset_path}/actions/actions.bin")

    # save segment_ids
    segment_ids = np.array(segment_ids)
    # fp = np.memmap(f'{dataset_path}/segment_ids.bin', dtype=np.int32, mode='w+', shape=segment_ids.shape)
    # fp[:] = segment_ids[:]  # map to trajectory index
    segment_ids = segment_ids.astype(np.int32)
    segment_ids.tofile(f"{dataset_path}/segment_ids.bin")

    # feature_mean = np.mean(videos)
    # feature_std = np.std((videos - feature_mean) / 1e9) * 1e9

    # save metadata
    if encoder_type == "magvit":
        vocab_size = int(2**18)
    elif encoder_type == "temporalvae":
        vocab_size = None
    else:
        raise NotImplementedError(f"{encoder_type=}")

    with open(f"{dataset_path}/metadata.json", "w") as f:  # Technically only need to save most of this data for shard 0
        json.dump(
            {
                "token_dtype": str(np.dtype(videos.dtype)),
                "action_dim": actions[0].shape[-1],
                "s": 16,
                "h": height,
                "w": width,
                "vocab_size": vocab_size,
                "hz": DATA_FREQ_TABLE.get(extern_dataset_name, 1),  # to be loaded from the data code
                "encoder_name_or_path": encoder_name_or_path,
                "encoder_type": encoder_type,
                "num_images": len(videos),
                "latent_channels": num_channels,
                "name": extern_dataset_name,
                # "feature_mean": feature_mean,
                # "feature_std": feature_std,
            },
            f,
        )

    print(f"{len(traj_lens)=} {np.mean(traj_lens)=} {np.sum(traj_lens)=}")
    print(f"Dataset creation time: {time.time() - start_time:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)

    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_TO_GEN_AND_SIZE.keys(), help="TODO")
    parser.add_argument(
        "--data_split", type=str, choices=["train", "val"], required=True, help="The split of the dataset to create."
    )
    parser.add_argument(
        "--episode_cnt", type=int, help="If specified, will limit the maximum number of trajectories to encode."
    )
    parser.add_argument(
        "--original_res",
        action="store_true",
        help="Maintain original resolution of the video rather than resizing it to 256x256.",
    )
    parser.add_argument("--no_quantization", action="store_true", help="Skip quantization step in visual encoder.")
    parser.add_argument(
        "--num_shards", type=int, default=1, help="The number of shards to partition the train/val dataset into."
    )
    parser.add_argument("--curr_shard_rank", type=int, default=0, help="The (0-indexed) shard number to encode.")
    parser.add_argument("--root_dir", type=str, default="data", help="The root directory to write all datasets to.")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="magvit",
        choices=["magvit", "temporalvae"],
        help="Type of the image tokenizer.",
    )
    parser.add_argument(
        "--encoder_name_or_path", type=str, default="data/magvit2.ckpt", help="The path or name of the image encoder."
    )
    parser.add_argument(
        "--no_encoding",
        action="store_true",
        help="Preserve the groundtruth raw images to compute metrics in validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.set_seed(233)

    dataset_postfix = f"shard{args.curr_shard_rank}_of_{args.num_shards}"
    if args.episode_cnt is not None:
        dataset_postfix = f"max{args.episode_cnt}_{dataset_postfix}"

    encode_dataset_split(
        extern_dataset_name=args.dataset_name,
        split=args.data_split,
        max_episodes=args.episode_cnt,
        dataset_postfix=dataset_postfix,
        original_res=args.original_res,
        no_quantization=args.no_quantization,
        num_shards=args.num_shards,
        curr_shard_rank=args.curr_shard_rank,
        root_dir=args.root_dir,
        encoder_type=args.encoder_type,
        encoder_name_or_path=args.encoder_name_or_path,
        no_encoding=args.no_encoding,
    )
