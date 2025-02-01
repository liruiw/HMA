"""
Merge data shards generated from `encode_{extern,openx}_dataset.py`
In addition to CLI args, `SHARD_DATA_FORMAT` must be changed depending on the dataset.
"""

import argparse
import json
import os

import numpy as np
from tqdm.auto import tqdm

SHARD_DATA_FORMAT = "/private/home/xinleic/LR/HPT-Video-KZ/sharded_data/droid_magvit_shard{}_of_{}_train"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_data_dir", type=str, required=True, help="Directory to save merged data, must not exist."
    )
    parser.add_argument("--num_shards", type=int, required=True, help="Number of shards the dataset was split into.")

    args = parser.parse_args()
    assert not os.path.exists(args.out_data_dir), "Will not overwrite existing directory."
    os.makedirs(os.path.join(args.out_data_dir, "actions"), exist_ok=True)

    num_frames = 0
    valid_inds = []

    for shard_ind in range(args.num_shards):
        shard_path = SHARD_DATA_FORMAT.format(shard_ind, args.num_shards)
        if os.path.isfile(os.path.join(shard_path, "metadata.json")):
            valid_inds.append(shard_ind)
            with open(os.path.join(shard_path, "metadata.json"), "r") as f:
                shard_metadata = json.load(f)

            num_frames += shard_metadata["num_images"]
        else:
            print(f"{shard_ind=} is invalid.")

    if num_frames == 0:
        print("No valid shards")
        exit(0)

    token_dtype = np.dtype(shard_metadata["token_dtype"])
    if shard_metadata["quantized"]:
        frame_dims = (shard_metadata["h"], shard_metadata["w"])
    else:
        frame_dims = (shard_metadata["latent_channels"], shard_metadata["h"], shard_metadata["w"])

    action_dim = shard_metadata["action_dim"]
    videos = np.memmap(
        os.path.join(args.out_data_dir, "video.bin"), dtype=token_dtype, mode="write", shape=(num_frames, *frame_dims)
    )

    actions = np.memmap(
        os.path.join(args.out_data_dir, "actions", "actions.bin"),
        dtype=np.float32,
        mode="write",
        shape=(num_frames, action_dim),
    )

    segment_ids = np.memmap(
        os.path.join(args.out_data_dir, "segment_ids.bin"), dtype=np.int32, mode="write", shape=(num_frames,)
    )

    prev_frame_ind = 0
    prev_segment_id = 0

    for shard_ind in tqdm(valid_inds):
        shard_path = SHARD_DATA_FORMAT.format(shard_ind, args.num_shards)
        with open(os.path.join(shard_path, "metadata.json"), "r") as f:
            shard_metadata = json.load(f)

        shard_num_frames = shard_metadata["num_images"]
        videos[prev_frame_ind : prev_frame_ind + shard_num_frames] = np.memmap(
            os.path.join(shard_path, "video.bin"),
            dtype=np.dtype(shard_metadata["token_dtype"]),
            mode="r",
            shape=(shard_num_frames, *frame_dims),
        )

        actions[prev_frame_ind : prev_frame_ind + shard_num_frames] = np.memmap(
            os.path.join(shard_path, "actions", "actions.bin"),
            dtype=np.float32,
            mode="r",
            shape=(shard_num_frames, action_dim),
        )

        segment_ids[prev_frame_ind : prev_frame_ind + shard_num_frames] = (
            np.memmap(
                os.path.join(shard_path, "segment_ids.bin"),
                dtype=np.int32,
                mode="r",
                shape=(shard_num_frames,),
            )
            + prev_segment_id
        )

        prev_segment_id = segment_ids[prev_frame_ind + shard_num_frames - 1] + 1
        prev_frame_ind += shard_num_frames

    assert prev_frame_ind == num_frames
    print("Finished")

    with open(os.path.join(args.out_data_dir, "metadata.json"), "w") as f:
        merged_metadata = (
            shard_metadata
            | vars(args)
            | {"num_images": num_frames, "input_path": SHARD_DATA_FORMAT.format(0, args.num_shards)}
        )

        json.dump(merged_metadata, f)
