# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
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
import matplotlib.pyplot as plt

RESOLUTION = (480, 480)
home = expanduser("~")

# Adjust these to the where-ever your detections and frames are stored.
ROOT = "/datasets01/ego4d_track2/"
LABEL_ROOT = ROOT + "v2_1/annotations/fho_main.json"
VIDEO_PATH = ROOT + "v2_1/full_scale/"
# from epic_kitchens.hoa import load_detections


# labels = json.load(open("/datasets01/ego4d_track2/v2_1/annotations/fho_main.json"))
# videos = /datasets01/ego4d_track2/v2_1/clips
def parse_video_frame(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ret, frame = cap.read()
    return frame


def parse_raw_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def compute_state_and_actions(image, curr_frame, next_frame, frame_idx, save=False):
    # curr_frame is a list of bounding box labels
    img_width, img_height = image.shape[1], image.shape[0]
    for box in curr_frame:
        if box["object_type"] == "left_hand":
            curr_hand1_center = [
                box["bbox"]["x"] + box["bbox"]["width"] / 2,
                box["bbox"]["y"] + box["bbox"]["height"] / 2,
            ]

        if box["object_type"] == "right_hand":
            curr_hand2_center = [
                box["bbox"]["x"] + box["bbox"]["width"] / 2,
                box["bbox"]["y"] + box["bbox"]["height"] / 2,
            ]

    for box in next_frame:
        if box["object_type"] == "left_hand":
            next_hand1_center = [
                box["bbox"]["x"] + box["bbox"]["width"] / 2,
                box["bbox"]["y"] + box["bbox"]["height"] / 2,
            ]

        if box["object_type"] == "right_hand":
            next_hand2_center = [
                box["bbox"]["x"] + box["bbox"]["width"] / 2,
                box["bbox"]["y"] + box["bbox"]["height"] / 2,
            ]

    # normalized them
    curr_hand1_center = np.array([curr_hand1_center[0] / img_width, curr_hand1_center[1] / img_height])
    curr_hand2_center = np.array([curr_hand2_center[0] / img_width, curr_hand2_center[1] / img_height])

    # normalize them
    next_hand1_center = np.array([next_hand1_center[0] / img_width, next_hand1_center[1] / img_height])
    next_hand2_center = np.array([next_hand2_center[0] / img_width, next_hand2_center[1] / img_height])

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
        cv2.imwrite(f"/private/home/xinleic/LR/hpt_video/data/ego4d_video_label_check/img_{frame_idx}.png", image)
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


def chunk_actions_and_concatenate(actions):
    chunk_size = 4
    chunked_actions = [actions[i : i + chunk_size] for i in range(0, len(actions), chunk_size)][:-1]
    concatenated_frames = []

    for chunk in chunked_actions:
        frames_to_concat = []
        for action in chunk:
            frames = action["frames"]  # Assuming 'frames' is a list or iterable
            if frames is not None:
                frames_to_concat.extend(frames)  # Collect frames from each action
        concatenated_frames.append(frames_to_concat)  # Store the concatenated frames for this chunk

    return concatenated_frames


def ego4d_dataset_size() -> int:
    """Returns the number of trajectories in the dataset. ~1725 for Ego4D."""
    labels = json.load(open(LABEL_ROOT))
    return len(labels["videos"])


# define your own dataset conversion
def ego4d_dataset_generator(example_inds: Iterable[int] = None):
    """
    Generator yielding data from Ego4D.
    Args:
        example_inds: if specified, will only yield data from these indices.
            Otherwise, will default to yielding the entire dataset.
    """
    # convert to a list of episodes that can be added to replay buffer
    labels = json.load(open(LABEL_ROOT))

    if example_inds is None:
        example_inds = range(len(labels["videos"]))

    for example_ind in example_inds:
        label = labels["videos"][example_ind]
        # ['annotated_intervals'][2]['narrated_actions']
        video_path = VIDEO_PATH + label["video_uid"] + ".mp4"
        if not os.path.exists(video_path):
            print("skip", video_path)
            continue

        label_detections = labels
        print("video_path:", video_path)
        print("len label detections", len(label_detections))

        # action extractions over bounding boxes subtractions of both hands.
        for interval in label["annotated_intervals"]:
            # print(video_detections[frame_idx].hands)

            lang = "use human hands to do some tasks"  # dummies
            # import IPython; IPython.embed()
            print(f"Interval [{interval['start_sec']} - {interval['end_sec']}]")
            actions = list(
                filter(
                    lambda x: not (x["is_invalid_annotation"] or x["is_rejected"]) and x["stage"] is not None,
                    interval["narrated_actions"],
                )
            )
            print(f"Actions: {len(actions)}")

            # because we need to concatenate
            if len(actions) < 3:
                continue

            # the number of frames is usually 7 and it also does not follow strict 2hz
            chunk_actions = chunk_actions_and_concatenate(actions)
            for frame_idx, frames in enumerate(chunk_actions):
                # lang = frame['narration_text']
                steps = []
                # need to use dummy actions to expand from 6 frames to 16 frames
                for idx, frame in enumerate(frames[:-1]):
                    frame_id = frame["frame_number"]
                    next_frame = frames[idx + 1]
                    image = parse_video_frame(video_path, frame_id)

                    if len(frame["boxes"]) > 2 and len(next_frame["boxes"]) > 2:
                        try:
                            s, a = compute_state_and_actions(
                                image, frame["boxes"], next_frame["boxes"], idx, save=False
                            )
                        except:
                            print(f"compute action failed idx {idx} frame idx {frame_idx}")
                            continue
                        # break into step dict
                        step = {
                            "observation": {"image": image, "state": s},
                            "action": a,
                            "language_instruction": lang,
                        }
                        steps.append(OrderedDict(step))

                if len(steps) < 16:
                    print("skip this traj because frame window length < 16")
                    continue
                data_dict = {"steps": steps}
                yield data_dict
