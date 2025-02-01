# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os
import numpy as np
from pathlib import Path


CURRENT_DIR = os.path.dirname(__file__)
import cv2
from os.path import expanduser
from epic_kitchens.hoa.types import BBox, FloatVector, HandSide
from epic_kitchens.hoa import load_detections

RESOLUTION = (480, 480)
home = expanduser("~")

# Adjust these to the where-ever your detections and frames are stored.
DETECTION_ROOT = f"/checkpoint/xinleic/LR/epic-kitchens-100-hand-object-bboxes/labels/hand-objects"
FRAMES_ROOT = f"/datasets01/EPIC-KITCHENS-100"

# DETECTION_ROOT = f'{home}/Projects/epic_kitchen_labels/hand-objects'
# FRAMES_ROOT = f'{home}/EPIC-KITCHENS'
detections_root = Path(DETECTION_ROOT)
frames_root = Path(FRAMES_ROOT)


def compute_state_and_actions(curr_frame, next_frame):
    curr_hand1, curr_hand2 = curr_frame.hands[0], curr_frame.hands[1]
    if curr_hand1.side != HandSide.LEFT:  # flip
        curr_hand1, curr_hand2 = curr_hand2, curr_hand1

    # already normalized
    curr_hand1_center = curr_hand1.bbox.center
    curr_hand2_center = curr_hand2.bbox.center

    next_hand1, next_hand2 = next_frame.hands[0], next_frame.hands[1]
    if next_hand1.side != HandSide.LEFT:  # flip
        next_hand1, next_hand2 = next_hand2, next_hand1

    # already normalized even
    next_hand1_center = next_hand1.bbox.center
    next_hand2_center = next_hand2.bbox.center
    state = np.concatenate((curr_hand1_center, curr_hand2_center))
    action = np.concatenate(
        (
            np.array(next_hand1_center) - np.array(curr_hand1_center),
            np.array(next_hand2_center) - np.array(curr_hand2_center),
        )
    )
    return state, action


# define your own dataset conversion
def convert_dataset_image():
    # convert to a list of episodes that can be added to replay buffer
    ALL_EPISODES = os.listdir(FRAMES_ROOT)
    MAX_EPISODE_LENGTH = 5000

    for EPS in ALL_EPISODES:
        rgb_path = os.path.join(FRAMES_ROOT, EPS, "rgb_frames")
        if not os.path.exists(rgb_path):
            continue
        for video_id in os.listdir(rgb_path):
            full_path = os.path.join(rgb_path, video_id)
            if (
                not full_path.endswith(".tar") and not full_path.endswith(".jpg") and not full_path.endswith("home")
            ):  # folder

                # action extractions over bounding boxes subtractions of both hands.
                participant_id = video_id[:3]
                video_detections = load_detections(detections_root / participant_id / (video_id + ".pkl"))
                max_frame_idx = len(video_detections) - 1
                DS_FACTOR = 1
                print(full_path)
                steps = []

                for frame_idx in range(0, max_frame_idx - DS_FACTOR, DS_FACTOR):
                    if (
                        len(video_detections[frame_idx].hands) != 2
                        or len(video_detections[frame_idx + DS_FACTOR].hands) != 2
                    ):
                        continue

                    s, a = compute_state_and_actions(
                        video_detections[frame_idx], video_detections[frame_idx + DS_FACTOR]
                    )
                    lang = "use human hands to do some tasks"  # dummies
                    image_path = frames_root / participant_id / "rgb_frames" / video_id / f"frame_{frame_idx:010d}.jpg"
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    image = image[..., [2, 1, 0]]  # RGB

                    # break into step dict
                    step = {
                        "observation": {"image": image, "state": s},
                        "action": a,
                        "language_instruction": lang,
                    }
                    steps.append(OrderedDict(step))
                    if len(steps) > MAX_EPISODE_LENGTH:
                        break
                data_dict = {"steps": steps}
                print(f"max_frame_idx: {max_frame_idx} ds factor: {DS_FACTOR} {len(steps)}")
                yield data_dict
