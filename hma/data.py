import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset
from hma.config import GenieConfig
from datasets.encode_openx_dataset import DATA_FREQ_TABLE
from hma.model.factorization_utils import factorize_token_ids, unfactorize_token_ids
from hma.model.st_mask_git import cosine_schedule

SVD_SCALE = 0.18215

def normalize_actions(actions: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    """
    compute mean and std of actions. Normalize actions is done inside the network.
    """
    mean = np.mean(actions, axis=0).tolist()
    std = np.std(actions, axis=0).tolist()
    return actions, [mean, std]



def get_maskgit_collator(config: GenieConfig):
    mask_token_id = config.image_vocab_size
    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1
        h = features[0]["h"]
        w = features[0]["w"]
        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        x_THW = rearrange(input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T,
                        h=h, w=w)
        x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
        labels = x_THW.clone()

        if config.dataloader_apply_corruption:
            # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
            r = torch.rand(x_THWC.size(), device=device)
            u01 = torch.rand((), device=device)
            random_patches_mask = r < config.max_corrupt_rate * u01
            random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THWC.size(),
                                        dtype=torch.long, device=device)
            x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            # first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(config.dataloader_mask_ratio_min, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand((len(features), h, w, config.num_factored_vocabs), device=device)
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1)
        if config.dataloader_apply_mask:
            c = 0

            while mask.max() == 0:  # We could get unlucky and mask no tokens?
                # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
                mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))
                r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
                mask = r < mask_prob_T
                c += 1

            if c > 1:
                print(f"Generated mask {c} > 1 times.")

            x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
            x_THW[:, first_masked_frame:][mask] = mask_token_id

        data_dict = {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
        }

        if "action_ids" in features[0]:
            data_dict['action_ids'] = torch.stack([ex["action_ids"] for ex in features])
        data_dict['domain'] = [ex["domain"] for ex in features]
        data_dict['h'] = [ex["h"] for ex in features]
        data_dict['w'] = [ex["w"] for ex in features]
        return data_dict


    return collate_fn




def get_maskgit_collator_feature(config: GenieConfig):
    # mask_token_id = config.image_vocab_size

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

        h = features[0]["h"]
        w = features[0]["w"]
        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        x_THWC = rearrange(input_ids, "b (t h w) c -> b t h w c", b=len(features), t=config.T, h=h, w=w)
        labels = x_THWC.clone()
        first_masked_frame = config.T

        mask = torch.zeros(1).long()
        mask_token_indicator = torch.zeros((len(features), config.T, h, w)).long()

        if config.dataloader_apply_mask:
            if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
                # Leave frames [0, first_masked_frame) unmasked.
                first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            else:  # Typical MLM masking
                first_masked_frame = 1

            c = 0
            while mask.max() == 0:  # We could get unlucky and mask no tokens?
                # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
                rand = torch.rand(len(features), config.T - first_masked_frame, 1, 1)
                # add a minimum mask ratio
                rand_mask = rand * (1 - config.dataloader_mask_ratio_min) + config.dataloader_mask_ratio_min
                mask_prob_T = cosine_schedule(rand_mask)
                r = torch.rand_like(x_THWC[:, first_masked_frame:, ..., 0], dtype=torch.float)
                mask = r < mask_prob_T
                c += 1

            if c > 1:
                print(f"Generated mask {c} > 1 times.")

            mask_token_indicator = torch.cat([
                torch.zeros((len(features), first_masked_frame, h, w), dtype=mask.dtype), mask], dim=1)

        data_dict = {
            "input_ids": rearrange(x_THWC, "b t h w c -> b (t h w) c"),
            "labels": rearrange(labels, "b t h w c-> b (t h w) c"),
            "masked_tokens_indicator": mask_token_indicator,
        }

        if "action_ids" in features[0]:
            data_dict['action_ids'] = torch.stack([ex["action_ids"] for ex in features])
        data_dict['domain'] = [ex["domain"] for ex in features]
        data_dict['h'] = [ex["h"] for ex in features]
        data_dict['w'] = [ex["w"] for ex in features]
        return data_dict
    return collate_fn

class RawTokenDataset(TorchDataset):
    """ Loads raw uint32 tokens as memmap-backed array """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False,
        use_actions=False,
        name='',
        max_traj_num=1000000,
        compute_stride_from_freq_table=True,
        natural_hz=2,
        drop_action_ratio=0.0
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
            use_actions: If True, will load the actions from the `actions` folder for the models
            name: the name of the dataset

        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["h"], self.metadata["w"])
        video_tokens_path, segment_ids_path, action_tokens_path = [data_dir / f"{name}.bin"
                                                                   for name in ["video", "segment_ids", "actions"]]
        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
        self.data = np.memmap(video_tokens_path, dtype=token_dtype, mode="r", shape=shape)
        self.window_size, self.stride = window_size, stride

        if len(name) == 0:
            self.name = self.metadata["name"]
        else:
            self.name = name

        if compute_stride_from_freq_table:
            self.stride = max(DATA_FREQ_TABLE.get(self.name, 1) // natural_hz, 1)
        print(f"RawTokenDataset: {self.name=} {self.stride=}")

        self.n_action = self.metadata.get("action_dim", 1) * (self.stride)
        self.drop_action_ratio = drop_action_ratio

        if use_actions:
            actions = []
            for action_file in sorted((data_dir / "actions").iterdir()):
                actions.append(np.memmap(action_file, dtype=np.float32, mode="r").reshape(len(self.data), -1))

            self.actions = np.concatenate(actions, axis=-1)
            self.actions, self.action_stat = normalize_actions(self.actions)

        if os.path.isfile(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(self.metadata["num_images"],)
            )
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride

        self.valid_start_inds = []
        for start_ind in range(len(self.data) - self.video_len - self.stride):
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted (or too short)
            # if the first and last frames have different segment ids.
            if not (filter_interrupts and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

            if  self.segment_ids is not None and self.segment_ids[start_ind] >= max_traj_num: # because we will filter based on window size later
                break

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

        self.num_videos = len(np.unique(self.valid_start_inds))
        print(f"Loaded {len(self)} sequences from {data_dir} {self.stride=} {self.window_size=} {self.n_action=} {self.num_videos=}")

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy((self.data[start_ind : start_ind + self.video_len + 1 : self.stride]).astype(np.int64))
        x = x.flatten() # 16 x 16 x 16

        # reconstructions since the input ids and the labels are the same
        attention_mask = torch.ones_like(x)
        data_dict = {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
            "h": self.metadata["h"],
            "w": self.metadata["w"],
        }
        if hasattr(self, "actions") and np.random.uniform() > self.drop_action_ratio:
            # we want to have all actions within the stride to predict the next frame at the end of the stride
            # we will concatenate the actions from [window_size, d_action] to [window_size, d_action * stride]
            # S x T x d_action
            data_dict['action_ids'] = self.actions[start_ind:start_ind + self.video_len + self.stride].reshape(self.window_size, -1)
            data_dict['action_ids'] = torch.from_numpy(data_dict['action_ids'].astype(np.float32))

        data_dict["domain"] = self.name
        return data_dict



class RawFeatureDataset(TorchDataset):
    """ Loads raw float32 tokens as memmap-backed array """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False,
        use_actions=False,
        max_traj_num=1000000,
        compute_stride_from_freq_table=True,
        natural_hz=2,
        datio_noise_ratio=0.0,
        use_raw_image_as_latent=False,
        domain=None,
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
            use_actions: If True, will load the actions from the `actions` folder for the models
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata.get("latent_channels", 4), self.metadata["h"], self.metadata["w"]) #

        video_tokens_path, segment_ids_path, action_tokens_path = [data_dir / f"{name}.bin"
                                                                   for name in ["video", "segment_ids", "actions"]]

        token_dtype = np.dtype(self.metadata.get("token_dtype", "float16"))
        self.data = np.memmap(video_tokens_path, mode="r", shape=shape, dtype=token_dtype)
        self.window_size, self.stride = window_size, stride
        self.datio_noise_ratio = datio_noise_ratio

        if domain is not None:
            self.name = domain
        else:
            self.name = self.metadata["name"]

        self.name = self.name.replace("_noquant", "")
        self.stride = stride
        if compute_stride_from_freq_table:
            self.stride = max(DATA_FREQ_TABLE.get(self.name, 1) // natural_hz, 1)
        self.n_action = self.metadata.get("action_dim", 1) * (self.stride)

        if use_actions:
            actions = []
            for action_file in sorted((data_dir / "actions").iterdir()):
                actions.append(np.memmap(action_file, dtype=np.float32, mode="r").reshape(len(self.data), -1))

            self.actions = np.concatenate(actions, axis=-1)
            self.actions, self.action_stat = normalize_actions(self.actions)

        if os.path.isfile(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(self.metadata["num_images"],)
            )
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride
        self.valid_start_inds = []

        for start_ind in range(len(self.data) - self.video_len - self.stride):
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted (or too short)
            # if the first and last frames have different segment ids.
            if not (filter_interrupts and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

            if len(self.valid_start_inds) >= max_traj_num:
                break

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

        num_videos = len(np.unique(self.segment_ids))
        print(f"Loaded {len(self)} sequences from {data_dir} {self.stride=} {self.window_size=} {self.n_action=} {num_videos=}")

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = self.data[start_ind : start_ind + self.video_len + 1 : self.stride].copy()
        x = torch.FloatTensor(x).float()
        x = x * SVD_SCALE

        x = rearrange(x, "t c h w -> (t h w) c")
        attention_mask = torch.ones_like(x)
        data_dict = {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
            "h": self.metadata["h"],
            "w": self.metadata["w"],
            "c": self.metadata["latent_channels"],
        }
        if hasattr(self, "actions"):
            # we want to have all actions within the stride to predict the next frame at the end of the stride
            # we will concatenate the actions from [window_size, d_action] to [window_size, d_action * stride]
            data_dict['action_ids'] = self.actions[start_ind:start_ind + self.video_len + self.stride].reshape(self.window_size, -1)
            data_dict['action_ids'] = torch.from_numpy(data_dict['action_ids'].astype(np.float32))

        data_dict["domain"] = self.name.replace("_noquant", "")
        return data_dict

class RawImageDataset(TorchDataset):
    """ Loads raw image dataset as memmap-backed array """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False,
        use_actions=False,
        max_traj_num=1000000,
        compute_stride_from_freq_table=True,
        natural_hz=2,
        datio_noise_ratio=0.0,
        domain=None,
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
            use_actions: If True, will load the actions from the `actions` folder for the models
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["h"], self.metadata["w"], 3) #
        video_tokens_path, segment_ids_path, action_tokens_path = [data_dir / f"{name}.bin"
                                                                   for name in ["video", "segment_ids", "actions"]]

        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint8"))
        self.data = np.memmap(video_tokens_path, mode="r", shape=shape, dtype=token_dtype)

        self.window_size, self.stride = window_size, stride
        self.datio_noise_ratio = datio_noise_ratio

        if domain is not None:  # TODO: remove
            self.name = domain
        else:
            self.name = self.metadata["name"]

        if compute_stride_from_freq_table:
            self.stride = max(DATA_FREQ_TABLE.get(self.name, 1) // natural_hz, 1)
        self.n_action = self.metadata.get("action_dim", 1) * (self.stride)

        if use_actions:
            actions = []
            for action_file in sorted((data_dir / "actions").iterdir()):
                actions.append(np.memmap(action_file, dtype=np.float32, mode="r").reshape(len(self.data), -1))

            self.actions = np.concatenate(actions, axis=-1)
            self.actions, self.action_stat = normalize_actions(self.actions)

        if os.path.isfile(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(self.metadata["num_images"],)
            )
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride
        self.valid_start_inds = []

        for start_ind in range(len(self.data) - self.video_len - self.stride):
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted (or too short)
            # if the first and last frames have different segment ids.
            if not (filter_interrupts and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

            if len(self.valid_start_inds) >= max_traj_num:
                break

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds
        print(f"Loaded {len(self)} sequences from {data_dir} {self.stride=} {self.window_size=} {self.n_action=}")

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = self.data[start_ind : start_ind + self.video_len + 1 : self.stride].copy()
        x = torch.FloatTensor(x).float()

        # reconstructions since the input ids and the labels are the same
        attention_mask = torch.ones_like(x)
        data_dict = {
            "images": x,
            "labels": x,  # Do we need labels/attention mask?
            "attention_mask": attention_mask,
            "h": self.metadata["h"],
            "w": self.metadata["w"],
        }
        if hasattr(self, "actions"):
            # we want to have all actions within the stride to predict the next frame at the end of the stride
            # we will concatenate the actions from [window_size, d_action] to [window_size, d_action * stride]
            data_dict['action_ids'] = self.actions[start_ind:start_ind + self.video_len + self.stride].reshape(self.window_size, -1)
            data_dict['action_ids'] = torch.from_numpy(data_dict['action_ids'].astype(np.float32))

        data_dict["domain"] = self.name
        return data_dict