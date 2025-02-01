# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
TODO: explain
"""
import h5py
import numpy as np
import cv2
import time
from collections import OrderedDict
import robomimic.utils.file_utils as FileUtils

from sim.robomimic.robomimic_runner import create_env, OBS_KEYS, RESOLUTION
from sim.robomimic.robomimic_wrapper import RobomimicLowdimWrapper

from typing import Optional, Iterable

DATASET_DIR = "data/robomimic/datasets"
SUPPORTED_ENVS = ["lift", "square", "can"]
NUM_EPISODES_PER_TASK = 200


def render_step(env, state):
    env.env.env.sim.set_state_from_flattened(state)
    env.env.env.sim.forward()
    img = env.render()
    img = cv2.resize(img, RESOLUTION)
    return img


def robomimic_dataset_size() -> int:
    return len(SUPPORTED_ENVS) * NUM_EPISODES_PER_TASK


def robomimic_dataset_generator(example_inds: Optional[Iterable[int]] = None):
    if example_inds is None:
        example_inds = range(robomimic_dataset_size())

    curr_env_name = None
    for idx in example_inds:
        # get env_name corresponding to idx
        env_name = SUPPORTED_ENVS[idx // NUM_EPISODES_PER_TASK]
        if curr_env_name is None or curr_env_name != env_name:
            # need to load new env
            dataset = f"{DATASET_DIR}/{env_name}/ph/image.hdf5"
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset)
            env_meta["use_image_obs"] = True
            env = create_env(env_meta=env_meta, obs_keys=OBS_KEYS)
            env = RobomimicLowdimWrapper(env=env)
            env.reset()  # NOTE: this is necessary to remove green laser bug
            curr_env_name = env_name

        with h5py.File(dataset) as file:
            demos = file["data"]
            local_episode_idx = idx % NUM_EPISODES_PER_TASK
            if f"demo_{local_episode_idx}" not in demos:
                continue

            demo = demos[f"demo_{local_episode_idx}"]
            obs = demo["obs"]
            states = demo["states"]
            action = demo["actions"][:].astype(np.float32)
            step_obs = np.concatenate([obs[key] for key in OBS_KEYS], axis=-1).astype(np.float32)
            steps = []
            for a, o, s in zip(action, step_obs, states):
                # break into step dict
                image = render_step(env, s)
                step = {
                    "observation": {"state": o, "image": image},
                    "action": a,
                    "language_instruction": f"{env_name}",
                }
                steps.append(OrderedDict(step))
            data_dict = {"steps": steps}
            yield data_dict

    env.close()
