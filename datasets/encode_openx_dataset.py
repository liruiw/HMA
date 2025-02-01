# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import argparse
import json
import os
import time
import traceback
from typing import Optional

import math
import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetBuilder
from tqdm import tqdm

from . import utils


SCRIPT_DESCRIPTION = """
Converts an Open X-Embodiment dataset from GS to encoded/tokenized data on disk.
This script only encodes one split (specified by `--data_split`)
of a one OpenX dataset (specified by `--dataset_name`) at a time.

Optionally, each split can be partitioned into multiple shards,
which is useful for parallelized encoding across GPUs.

Example usage:
    CUDA_VISIBLE_DEVICES=0 python -m datasets.encode_openx_dataset --dataset_name bc_z --data_split train --episode_cnt 500 --num_shards 16 --curr_shard_rank 0
    CUDA_VISIBLE_DEVICES=1 python -m datasets.encode_openx_dataset --dataset_name bc_z --data_split train --episode_cnt 500 --num_shards 16 --curr_shard_rank 1

    set -e
    for ((i = 0; i < 64; i += 2)); do
        CUDA_VISIBLE_DEVICES=0 python -m datasets.encode_openx_dataset --dataset_name bridge --data_split train --num_shards 64 --curr_shard_rank $i --root_dir sharded_data
    done

    set -e
    for ((i = 1; i < 64; i += 2)); do
        CUDA_VISIBLE_DEVICES=1 python -m datasets.encode_openx_dataset --dataset_name bridge --data_split train --num_shards 64 --curr_shard_rank $i --root_dir sharded_data
    done

Example usage (SVD tokenizer):
CUDA_VISIBLE_DEVICES=0 python -m datasets.encode_openx_dataset --dataset_name language_table --data_split val --no_quantization --encoder_type temporalvae --encoder_name_or_path 'stabilityai/stable-video-diffusion-img2vid'
""".strip()

# The validation set is the first VAL_RATIO examples in the dataset, and clipped to [MIN_VAL_EXAMPLES, MAX_VAL_EXAMPLES]
VAL_RATIO = 0.05
MIN_VAL_EXAMPLES, MAX_VAL_EXAMPLES = 20, 200


DATA_FREQ_TABLE = {
    "austin_sailor_dataset_converted_externally_to_rlds": 20,
    "stanford_hydra_dataset_converted_externally_to_rlds": 10,
    "austin_buds_dataset_converted_externally_to_rlds": 20,
    "austin_sirius_dataset_converted_externally_to_rlds": 20,
    "berkeley_mvp_converted_externally_to_rlds": 5,
    "berkeley_rpt_converted_externally_to_rlds": 30,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 2,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 20,
    "utaustin_mutex": 20,
    "imperialcollege_sawyer_wrist_cam": 10,
    "language_table": 2,  # changed to match frequency
    "kuka": 2,  # changed to match frequency
    "bc_z": 10,
    "robo_net": 1,
    "dlr_sara_pour_converted_externally_to_rlds": 10,
    "stanford_robocook_converted_externally_to_rlds": 5,
    "cmu_play_fusion": 5,
    "bridge": 5,
    "furniture_bench_dataset_converted_externally_to_rlds": 10,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": 3,
    "usc_cloth_sim_converted_externally_to_rlds": 10,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": 20,
    "roboturk": 10,
    "kaist_nonprehensile_converted_externally_to_rlds": 10,
    "asu_table_top_converted_externally_to_rlds": 12,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": 10,
    "berkeley_cable_routing": 10,
    "droid": 15,
    "uiuc_d3field": 1,
    "robo_set": 5,
    "toto": 30,
    "nyu_door_opening_surprising_effectiveness": 3,
    "nyu_franka_play_dataset_converted_externally_to_rlds": 3,
    "mimic_play": 15,
    "maniskill_dataset_converted_externally_to_rlds": 20,
    "columbia_cairlab_pusht_real": 10,
    "conq_hose_manipulation": 30,
    "dlr_edan_shared_control_converted_externally_to_rlds": 5,
    "berkeley_gnm_sac_son": 10,
    "berkeley_autolab_ur5": 5,
    "aloha_mobile": 30,
    "1x_humanoid": 30,
    "epic_kitchen_originalres": 30,
    "epic_kitchen": 30,
    "exoego4d": 30,
    "ego4d": 1,  # less than this.
    "robomimic": 6,  # average length around 50
    "metaworld": 6,
    "frodobot": 30,
    "fractal20220817_data": 3,
    # robomimic
    "robomimic": 6,  # average length around 50
    "robomimic_new": 6,  # average length around 50
    "robomimic_multitask_new": 6,  # average length around 50
    "robomimic_new_perturb": 6,  # average length around 50
    "robomimic_multitask_new_perturb": 6,  # average length around 50
}


def select_image(observation, verbose=False):
    """
    Select a canonical frame as image observation.
    """
    imgs = []
    # does not need to prefer wrist camera
    for key in ["rgb", "image"]:
        for obs_key in observation:
            if key in obs_key and "depth" not in obs_key:
                image = observation[obs_key]
                if type(observation[obs_key]) is not np.ndarray:
                    image = image.numpy()
                if verbose:
                    print("selected image key:", obs_key)
                imgs.append(image)

    return imgs


def process_dataset_step(
    step, encoder_type: str, encoder_name_or_path: str, keep_res=False, quantize=True, no_encoding=False
):
    """
    Map dataset-specific keys and values to a unified format.

    Args:
        step (dict): The step dictionary containing the dataset-specific information.
        encoder_type (str, optional): The image encoder to use.
    Returns:
        dict: The processed step dictionary with the mapped keys and values.
    """
    step_dict = {}
    try:
        if "action" in step:
            step_dict["action"] = np.array(step["action"])

            # handle action
            if type(step["action"]) is dict:
                step_dict["action"] = step_dict["action"].item()

                # outlier cases
                action = []
                for k, v in sorted(step_dict["action"].items()):
                    action.append(v.numpy().reshape(-1))
                step_dict["action"] = np.concatenate(action)

        # handle image
        images = select_image(step["observation"])

        # compute the embeddings.
        if no_encoding:
            step_dict["image"] = utils.resize_image(images[0])
        elif quantize:
            step_dict["image"] = utils.get_quantized_image_embeddings(
                images[0],
                encoder_type=encoder_type,
                encoder_name_or_path=encoder_name_or_path,
                keep_res=keep_res,
            )
        else:
            step_dict["image"] = utils.get_vae_image_embeddings(
                images[0],
                encoder_type=encoder_type,
                encoder_name_or_path=encoder_name_or_path,
                keep_res=keep_res,
            )
    except Exception as e:
        print("--------------------------")
        print("process_dataset_step exception:", traceback.format_exc())

    return step_dict


def get_dataset_builder(gs_dataset_name) -> tuple[DatasetBuilder, int]:
    """
    Returns the dataset builder and the total number of examples (for the train split).
    """
    try:
        builder = tfds.builder_from_directory(builder_dir=f"gs://gresearch/robotics/{gs_dataset_name}/0.1.0/")
    except:
        try:
            builder = tfds.builder_from_directory(builder_dir=f"gs://gresearch/robotics/{gs_dataset_name}/1.0.0/")
        except:
            builder = tfds.builder_from_directory(builder_dir=f"gs://gresearch/robotics/{gs_dataset_name}/0.0.1/")

    info = builder.info
    num_examples = info.splits["train"].num_examples

    return builder, num_examples


def get_shard_inds(first_split_ind: int, last_split_ind: int, curr_shard_rank: int, num_shards: int) -> tuple[int, int]:
    """
    Given the indices of the first (inclusive) and last (exclusive) examples in the data split (i.e. entire train dataset or val dataset),
    returns the indices of the first (inclusive) and last (exclusive) examples for the current shard in this data split.
    """
    split_num_examples = last_split_ind - first_split_ind
    shard_size_float = split_num_examples / num_shards  # average number of examples per shard
    return (
        first_split_ind + math.ceil(curr_shard_rank * shard_size_float),
        min(first_split_ind + math.ceil((curr_shard_rank + 1) * shard_size_float), last_split_ind),
    )


def encode_dataset_split(
    gs_dataset_name: str,
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
    Converts an Open X-Embodiment dataset from GS to encoded/tokenized data on disk.
    The data written to disk can be used to load a `RawTokenDataset` (or the continuous version.)

    Args:
        gs_dataset_name: the name of the dataset in Google Storage.
            Can be checked with gsutil ls -d gs://gresearch/robotics/*/
        split: expected to be either "train" or "val". TODO: decide how to split
        max_episodes: the maximum number of trajectories to include in the dataset.
        dataset_postfix: will be a suffix of the output dirname.
        image_encoder: string specifying the type of image encoder/tokenizer to use.
        original_res: if True, will maintain original resolution of the video rather than resizing it to 256x256.
        no_quantization: if True, will not perform quantization step in image encoder.
    """
    gs_dataset_name = gs_dataset_name.strip()  # never modified
    suffixed_dataset_name = gs_dataset_name  # will modify later
    if no_quantization:
        video_dtype = np.float16
    elif no_encoding:
        video_dtype = np.uint8
    else:
        video_dtype = np.uint32
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
    builder, num_examples = get_dataset_builder(gs_dataset_name)
    if max_episodes is not None:
        num_examples = min(num_examples, max_episodes)  # clip num_examples

    # We will only operate on a subset of the training examples, depending on:
    #      1) The split (train/val). Some examples are reserved for the other split.
    #      2) Sharding
    assert (
        num_examples > MIN_VAL_EXAMPLES
    ), f"{num_examples=} {MIN_VAL_EXAMPLES=}"  # non-positive number of train examples otherwise
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
        ds = builder.as_dataset(split=f"train[{start_idx}:{end_idx}]")

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

    if no_quantization:
        num_channels, height, width = videos[-1].shape[:3]
    else:
        height, width = videos[-1].shape[:2]
        num_channels = None

    ##### Write videos, actions, segment_ids, and metadata #####
    # align format to save segment_ids.bin, video.bin, actions/action.bin, metadata.json
    # save videos
    videos = np.stack(videos, axis=0)
    fp = np.memmap(f"{dataset_path}/video.bin", dtype=video_dtype, mode="w+", shape=videos.shape)
    fp[:] = videos[:]

    # save action
    utils.mkdir_if_missing(f"{dataset_path}/actions")
    actions = np.stack(actions, axis=0)
    fp = np.memmap(f"{dataset_path}/actions/actions.bin", dtype=np.float32, mode="w+", shape=actions.shape)
    fp[:] = actions[:]

    # save segment_ids
    segment_ids = np.array(segment_ids)
    fp = np.memmap(f"{dataset_path}/segment_ids.bin", dtype=np.int32, mode="w+", shape=segment_ids.shape)
    fp[:] = segment_ids[:]  # map to trajectory index

    # feature_mean = float(np.mean(videos))
    # feature_std = float(np.std((videos - feature_mean) / 1e9)) * 1e9
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
                "hz": DATA_FREQ_TABLE.get(gs_dataset_name, 1),  # to be loaded from the data code  TODO: remove default?
                "encoder_name_or_path": encoder_name_or_path,
                "encoder_type": encoder_type,
                "num_images": len(videos),
                "name": gs_dataset_name,
                "latent_channels": num_channels,
                "quantized": not args.no_quantization,
                # "feature_mean": feature_mean,
                # "feature_std": feature_std,
            },
            f,
        )

    print(f"{len(traj_lens)=} {np.mean(traj_lens)=} {np.sum(traj_lens)=}")
    print(f"Dataset creation time: {time.time() - start_time:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the Open X-Embodiment dataset on Google Storage. "
        "Can be checked with gsutil ls -d gs://gresearch/robotics/*/. ",
    )
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

    dataset_postfix = f"shard{args.curr_shard_rank}_of_{args.num_shards}" if args.num_shards > 1 else ""
    if args.episode_cnt is not None:
        dataset_postfix = f"max{args.episode_cnt}_{dataset_postfix}" if dataset_postfix else f"max{args.episode_cnt}"

    encode_dataset_split(
        gs_dataset_name=args.dataset_name,
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
