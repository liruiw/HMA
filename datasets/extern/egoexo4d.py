# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes/blob/master/notebooks/demo.ipynb
import os
from typing import Iterable

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os
import numpy as np
from pathlib import Path


CURRENT_DIR = os.path.dirname(__file__)
import cv2
from os.path import expanduser
import json


# Adjust these to the where-ever your detections and frames are stored.
CAM = "cam01"  # cam01
ROOT = "/datasets01/egoexo4d/v2/"
LABEL_ROOT = ROOT + "annotations/ego_pose/train/hand/automatic/{}.json"
VIDEO_PATH = ROOT + "takes/{}/frame_aligned_videos/{}.mp4"
TAKE_ROOT = ROOT + "takes.json"


def compute_state_and_actions(image, curr_frame, next_frame, idx, save=False):
    img_width, img_height = image.shape[1], image.shape[0]

    # already normalized
    curr_hand1_center = curr_frame[0]["annotation2D"][CAM]["left_wrist"]
    curr_hand2_center = curr_frame[0]["annotation2D"][CAM]["right_wrist"]

    # normalized them
    curr_hand1_center = np.array([curr_hand1_center["x"] / img_width, curr_hand1_center["y"] / img_height])
    curr_hand2_center = np.array([curr_hand2_center["x"] / img_width, curr_hand2_center["y"] / img_height])

    next_hand1_center = next_frame[0]["annotation2D"][CAM]["left_wrist"]
    next_hand2_center = next_frame[0]["annotation2D"][CAM]["right_wrist"]

    # normalize them
    next_hand1_center = np.array([next_hand1_center["x"] / img_width, next_hand1_center["y"] / img_height])
    next_hand2_center = np.array([next_hand2_center["x"] / img_width, next_hand2_center["y"] / img_height])

    state = np.concatenate(
        (curr_hand1_center, curr_hand2_center)
    )  #  - np.array(curr_hand1_center)  - np.array(curr_hand2_center)
    action = np.concatenate(
        (
            np.array(next_hand1_center),
            np.array(next_hand2_center),
        )
    )
    if save:
        # draw the bounding boxes
        cv2.circle(
            image, (int(curr_hand1_center[0] * img_width), int(curr_hand1_center[1] * img_height)), 10, (0, 255, 0), -1
        )
        cv2.circle(
            image, (int(curr_hand2_center[0] * img_width), int(curr_hand2_center[1] * img_height)), 10, (0, 255, 0), -1
        )
        cv2.circle(
            image, (int(next_hand1_center[0] * img_width), int(next_hand1_center[1] * img_height)), 10, (0, 0, 255), -1
        )
        cv2.circle(
            image, (int(next_hand2_center[0] * img_width), int(next_hand2_center[1] * img_height)), 10, (0, 0, 255), -1
        )
        # save the image
        cv2.imwrite(f"output/inspect/test_{idx}.png", image)
    return state, action


def parse_raw_video(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def egoexo4d_dataset_size() -> int:
    """Returns the number of takes in the dataset. ~5k for v2."""
    takes = json.load(open(TAKE_ROOT))
    return len(takes)


# define your own dataset conversion
def egoexo4d_dataset_generator(example_inds: Iterable[int] = None):
    """
    Generator yielding data from Ego-Exo4D.
    Args:
        example_inds: if specified, will only yield data from these indices.
            Otherwise, will default to yielding the entire dataset.
    """
    # convert to a list of episodes that can be added to replay buffer
    MAX_EPISODE_LENGTH = 5000
    TAKE_FILE = json.load(open(TAKE_ROOT))
    print("total takes", len(TAKE_FILE))
    # find the first camera with aria
    global CAM

    def find_aria_name(take):
        for cam in take["cameras"]:
            if "aria" in cam["name"]:
                return cam["name"]
        return None

    if example_inds is None:
        example_inds = range(len(TAKE_FILE))

    for example_ind in example_inds:
        take = TAKE_FILE[example_ind]
        take_name = take["take_name"]
        take_uid = take["take_uid"]
        # CAM = find_aria_name(take)
        # if CAM is None:
        #     continue

        video_path = VIDEO_PATH.format(take_name, CAM)
        label_path = LABEL_ROOT.format(take_uid)

        if not os.path.exists(video_path) or not os.path.exists(label_path):
            continue

        video_frames = parse_raw_video(video_path)
        label_detections = json.load(open(label_path))
        print("video_path:", video_path)
        print("len video frames", len(video_frames))
        print("len label detections", len(label_detections))

        # action extractions over bounding boxes subtractions of both hands.
        max_frame_idx = len(video_frames) - 1
        DS_FACTOR = 1
        frame_idx = 0
        start_frame_idx = 0
        MIN_CLIP_LENGTH = 300

        def get_continuous_chunk(start_idx, label_detections):
            end_idx = start_idx + 1
            while (
                str(start_idx) in label_detections
                and len(label_detections[str(start_idx)]) > 0
                and str(end_idx) in label_detections
                and len(label_detections[str(end_idx)]) > 0
            ):
                end_idx += 1
            return end_idx

        print("TAKE", take_name)

        # some frames might not have label. if there is a gap, skip
        while start_frame_idx < max_frame_idx - DS_FACTOR:
            lang = "use human hands to do some tasks"  # dummies
            if str(start_frame_idx) not in label_detections or str(start_frame_idx + DS_FACTOR) not in label_detections:
                start_frame_idx += DS_FACTOR
                continue

            end_frame_idx = get_continuous_chunk(start_frame_idx, label_detections)
            if end_frame_idx - start_frame_idx < MIN_CLIP_LENGTH:
                start_frame_idx = end_frame_idx
                continue

            print("start clipping from", start_frame_idx, "to", end_frame_idx)
            steps = []
            for frame_idx in range(start_frame_idx, end_frame_idx - DS_FACTOR, DS_FACTOR):
                image = video_frames[frame_idx][..., [2, 1, 0]]  # RGB
                try:
                    s, a = compute_state_and_actions(
                        image,
                        label_detections[str(frame_idx)],
                        label_detections[str(frame_idx + DS_FACTOR)],
                        frame_idx,
                        save=False,
                    )
                except:
                    break
                # break into step dict
                step = {
                    "observation": {"image": image, "state": s},
                    "action": a,
                    "language_instruction": lang,
                }
                steps.append(OrderedDict(step))
                if len(steps) > MAX_EPISODE_LENGTH:
                    break

            start_frame_idx = end_frame_idx
            if len(steps) < MIN_CLIP_LENGTH:
                data_dict = {"steps": steps}
                print(f"max_frame_idx: {max_frame_idx} ds factor: {DS_FACTOR} {len(steps)}")
                yield data_dict
