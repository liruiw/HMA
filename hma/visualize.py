#!/usr/bin/env python3

"""
Script to decode tokenized video into images/video.
Example usage:
python hma/visualize.py   --token_dir data/genie_generated

"""

import argparse
import math
import os
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.distributed.optim
import torch.utils.checkpoint
import torch.utils.data
import torchvision.transforms.v2.functional as transforms_f
from diffusers import AutoencoderKLTemporalDecoder
from einops import rearrange
from matplotlib import pyplot as plt
from hma.data import RawFeatureDataset

from hma.data import RawTokenDataset
from datasets.utils import get_image_encoder
from external.magvit2.config import VQConfig
from external.magvit2.models.lfqgan import VQModel
from hma.eval_utils import decode_tokens, decode_features
import wandb

SVD_SCALE = 0.18215

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tokenized video as GIF or comic.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame skip",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="data/genie_generated",
        help="Directory of tokens, in the format of `video.bin` and `metadata.json`. "
             "Visualized gif and comic will be written here.",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset to start generating images from"
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second"
    )
    parser.add_argument(
        "--max_images", type=int, default=None, help="Maximum number of images to generate. None for all."
    )
    parser.add_argument(
        "--example_ind", type=int, default=0,
        help="The index in the dataset of the example to generate on."
    )
    parser.add_argument(
        "--project_prefix", type=str, default="", help="Project suffix."
    )
    parser.add_argument(
        "--disable_comic", action="store_true",
        help="Comic generation assumes `token_dir` follows the same format as generate: e.g., "
             "`prompt | predictions | gtruth` in `video.bin`, `window_size` in `metadata.json`."
             "Therefore, comic should be disabled when visualizing videos without this format, such as the dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size, current script only supports a single GPU."
    )
    parser.add_argument(
        "--max_example", type=int, default=4,
        help="Maximum number of examples."
    )
    parser.add_argument(
        "--use_feature", action="store_true",
        help="visualize the features rather than tokens"
    )
    args = parser.parse_args()

    return args


def export_to_gif(frames: list, output_gif_path: str, fps: int):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Desired frames per second.
    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    duration_ms = 1000 / fps
    pil_frames[0].save(output_gif_path.replace(".mp4", ".gif"),
                       format="GIF",
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=duration_ms,
                       loop=0)
    # return the gif
    return output_gif_path.replace(".mp4", ".gif")

def unnormalize_imgs(normalized_imgs):
    """
    [-1, 1] -> [0, 255]

    Important: clip to [0, 255]
    """
    normalized_imgs = torch.clamp(normalized_imgs, -1, 1)
    rescaled_output = ((normalized_imgs.detach().cpu() + 1) * 127.5)
    clipped_output = torch.clamp(rescaled_output, 0, 255).to(dtype=torch.uint8)
    return clipped_output


def decode_latents_wrapper(
    batch_size: int = 16,
    encoder_type: str = "magvit",
    encoder_name_or_path: str = "data/magvit2.ckpt",
    max_images: int = None,
    device: str = "cuda",
):
    dtype = torch.bfloat16 # torch.bfloat16
    model = get_image_encoder(encoder_type, encoder_name_or_path)
    model = model.to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(video_data: np.array):
        """
        video_data: (b, h, w) for quantized data, or (b, c, h, w) for continuous data,
        where b is `batch_size` and different from training/eval batch size.
        """
        decoded_imgs = []

        for shard_ind in range(math.ceil(len(video_data) / batch_size)):
            shard_data = video_data[shard_ind * batch_size: (shard_ind + 1) * batch_size]
            if isinstance(model, VQModel):  # TODO: class agnostic wrapper
                # expecting quantized
                assert shard_data.ndim == 3, f"{shard_data.shape=} {shard_data.dtype=}"
                torch_shard = torch.from_numpy(shard_data.astype(np.int64))
                quant = model.quantize.get_codebook_entry(rearrange(torch_shard, "b h w -> b (h w)"),
                                                          bhwc=torch_shard.shape + (model.quantize.codebook_dim,)).flip(1)
                normalized_imgs = model.decode(quant.to(device=device, dtype=dtype))
            elif isinstance(model, AutoencoderKLTemporalDecoder):
                # expecting continuous
                assert shard_data.ndim == 4, f"{shard_data.shape=} {shard_data.dtype=}"
                torch_shard = torch.from_numpy(shard_data)

                torch_shard = torch.clamp(torch_shard, -25, 25)
                normalized_imgs = model.decode(torch_shard.to(device=device, dtype=dtype), num_frames=1).sample # sample to mean

            else:
                raise NotImplementedError(f"{model=}")

            decoded_imgs.append(unnormalize_imgs(normalized_imgs))
            if max_images and len(decoded_imgs) * batch_size >= max_images:
                break

        return [transforms_f.to_pil_image(img) for img in torch.cat(decoded_imgs)]

    return decode_latents


def caption_image(pil_image: Image, caption: str):
    """
    Add a bit of empty space at the top, and add the caption there
    """
    border_size = 36
    font_size = 24
    # convert pil_image to PIL.Image.Image if it's not already
    if not isinstance(pil_image, Image.Image):
        pil_image = transforms_f.to_pil_image(pil_image)

    width, height = pil_image.size
    new_width = width
    new_height = height + border_size

    new_image = Image.new("RGB", (new_width, new_height), "white")
    new_image.paste(pil_image, (0, border_size))

    # Draw the caption
    draw = ImageDraw.Draw(new_image)

    # Center text (`align` keyword doesn't work)
    _, _, text_w, text_h = draw.textbbox((0, 0), caption, font_size=font_size)
    draw.text(((width - text_w) / 2, (border_size - text_h) / 2), caption, fill="black", font_size=font_size)

    return new_image


@torch.no_grad()
def main():
    args = parse_args()
    name = args.token_dir.split('/')[-2]
    name_split = name.find('nodes')
    model = name[:name_split-7]
    dataset = name[name_split+8:]

    # Load tokens
    if args.use_feature:
        token_dataset = RawFeatureDataset(args.token_dir, 1, compute_stride_from_freq_table=False,
                                          filter_interrupts=False, filter_overlaps=False)
        video_tokens = token_dataset.data

        print(f"Loaded {video_tokens.shape=}")
    else:
        token_dataset = RawTokenDataset(args.token_dir, 1, compute_stride_from_freq_table=False,
                                        filter_interrupts=False, filter_overlaps=False)
        video_tokens = token_dataset.data
        print(f"Loaded {video_tokens.shape=}")

    metadata = token_dataset.metadata
    video_tokens = video_tokens.reshape(-1, metadata["window_size"] * 2 - metadata["num_prompt_frames"], *video_tokens.shape[1:])
    decode_func = decode_latents_wrapper
    print(metadata)
    print(f"Reshape {video_tokens.shape=}")

    wandb.init(project='video_eval_vis', settings=wandb.Settings(start_method="thread"), name=f"{args.project_prefix}vis_{model}", id=f"{args.project_prefix}vis_{model}", resume="allow")
    for example_id in range(min(args.max_example, len(video_tokens))):
        if args.use_feature:
            if "encoder_type" not in metadata:
                metadata["encoder_type"] = "temporalvae"
                metadata["encoder_name_or_path"] = "stabilityai/stable-video-diffusion-img2vid"
            decode_latents = decode_func(max_images=args.max_images, encoder_name_or_path=metadata["encoder_name_or_path"],
                                       encoder_type=metadata["encoder_type"])  # args.offset::args.stride
            this_video_token = torch.FloatTensor(video_tokens[example_id].copy())[None] / SVD_SCALE
            this_video_token = rearrange(this_video_token, "b t c h w -> b t h w c")
            video_frames = decode_features(this_video_token, decode_latents)
            video_frames = rearrange(video_frames, "b t c h w -> b t h w c")
            video_frames = video_frames.detach().cpu().numpy()[0].astype(np.uint8)
        else:
            decode_latents = decode_func(max_images=args.max_images)
            this_video_token = torch.LongTensor(video_tokens[example_id])[None]
            video_frames = decode_tokens(this_video_token, decode_latents)
            video_frames = rearrange(video_frames, "b t c h w -> b t h w c")
            video_frames = video_frames.detach().cpu().numpy()[0].astype(np.uint8)

        output_gif_path = os.path.join(args.token_dir, f"example{args.offset}.gif")

        # `generate` should populate `metadata.json` with these keys, while ground truth metadata does not have them
        is_generated_data = all(key in metadata for key in ("num_prompt_frames", "window_size"))
        if is_generated_data:
            if video_tokens[example_id].shape[0] != metadata["window_size"] * 2 - metadata["num_prompt_frames"]:
                raise ValueError(f"Unexpected {video_tokens.shape=} given {metadata['window_size']=}, {metadata['num_prompt_frames']=}")

            captioned_frames = []
            for i, frame in enumerate(video_frames):
                if i < metadata["num_prompt_frames"]:
                    caption = "Prompt"
                elif i < metadata["window_size"]:
                    caption = "Generated"
                else:
                    caption = "Ground truth"

                captioned_frames.append(caption_image(frame, caption))
        else:
            # Leave ground truth frames uncaptioned
            captioned_frames = video_frames

        gif_path = export_to_gif(captioned_frames, output_gif_path, args.fps)
        print(f"Saved to {output_gif_path}")

        if not args.disable_comic:
            fig, axs = plt.subplots(nrows=2, ncols=metadata["window_size"], figsize=(3 * metadata["window_size"], 3 * 2))
            for i, image in enumerate(video_frames):
                if i < metadata["num_prompt_frames"]:
                    curr_axs = [axs[0, i], axs[1, i]]
                    title = "Prompt"

                elif i < metadata["window_size"]:
                    curr_axs = [axs[0, i]]
                    title = "Prediction"
                else:
                    curr_axs = [axs[1, i - metadata["window_size"] + metadata["num_prompt_frames"]]]
                    title = "Ground truth"

                for ax in curr_axs:
                    ax.set_title(title)
                    ax.imshow(image)
                    ax.axis("off")

            output_comic_path = os.path.join(args.token_dir, f"example{args.offset}.png")
            plt.savefig(output_comic_path, bbox_inches="tight")
            plt.close()
            print(f"Saved to {output_comic_path}")
        wandb.log({f"{dataset}/gif_{example_id}": wandb.Video(gif_path)})

    # add wandb logging
    wandb.run.summary["model_checkpoint"] = metadata["model_checkpoint"]
    wandb.run.summary["dataset"] = metadata["dataset"]
    wandb.run.summary["trained_steps"] = metadata["trained_steps"]

    wandb.finish()


if __name__ == "__main__":
    main()
