# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_gr00t_so100.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab markers and place into pen holder."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import sys
import os

import draccus
import matplotlib.pyplot as plt
import numpy as np


# Image Server

## Robot Controller
from g1_low_control import Custom, G1JointIndex, Mode
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# temp, later to be imported from config
G1_NUM_MOTOR = 29

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]


# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/drive2/humanoid_ws/src/Isaac-GR00T/gr00t/eval/"))
# from service import ExternalRobotInferenceClient

from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################


class Gr00tRobotInferenceClient:

    def __init__(
        self,
        host="localhost",
        port=5555,
        show_images=False,
        # If True, convert incoming images from BGR (OpenCV) to RGB before sending
        assume_bgr: bool = True,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.show_images = show_images
        self.assume_bgr = assume_bgr
        self.logger = logging.getLogger(__name__)

    def get_action(self, observation_dict, lang: str):
        """
        Convert the G1 observation dict to the format expected by the policy server.
        Based on unitree_g1__modality.json structure.
        """
        # Create the observation dict with proper key naming
        obs_dict = {}
        
        # Add state modalities - flatten the nested state dict
        for state_key, state_value in observation_dict["state"].items():
            obs_dict[f"state.{state_key}"] = state_value
        
        # Add video modalities - ensure dtype and channel order (uint8 RGB)
        for video_key, video_value in observation_dict["video"].items():
            # try to coerce to numpy array
            try:
                vid = np.asarray(video_value)
            except Exception:
                self.logger.warning(f"Video modality '{video_key}' is not array-like; sending as-is")
                obs_dict[f"video.{video_key}"] = video_value
                continue

            # Ensure last axis is color channels
            if vid.ndim == 3 and vid.shape[2] == 3:
                # convert dtype if needed
                if vid.dtype != np.uint8:
                    vid = vid.astype(np.uint8)
                # optionally convert BGR->RGB for OpenCV-sourced frames
                if self.assume_bgr:
                    try:
                        vid = vid[..., ::-1]
                    except Exception:
                        pass
            obs_dict[f"video.{video_key}"] = vid
        
        # Add annotation
        obs_dict["annotation.human.task_description"] = lang

        # Show images if enabled
        if self.show_images:
            view_img({k: v for k, v in obs_dict.items() if k.startswith("video.")})

        # Add a dummy dimension for history (batch dimension)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # print("Sending observation to policy server:")
        # # print("  Keys:", obs_dict.keys())
        # for k, v in obs_dict.items():
        #     if isinstance(v, np.ndarray):
        #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        #     else:
        #         print(f"  {k}: {type(v)}")

        # Get the action chunk via the policy server
        action_chunk = self.policy.get_action(obs_dict)

        # Robust logging of policy response
        if isinstance(action_chunk, dict):
            # print keys and shapes where possible
            info_lines = []
            for k, v in action_chunk.items():
                try:
                    shape = np.asarray(v).shape
                except Exception:
                    shape = None
                info_lines.append(f"{k}: shape={shape}")
            self.logger.info("Policy returned dict with keys: %s", ", ".join(info_lines))
        else:
            try:
                self.logger.info("Policy returned array with shape=%s", np.asarray(action_chunk).shape)
            except Exception:
                self.logger.info("Policy returned response of type %s", type(action_chunk))

        # print("Received action_chunk keys:", action_chunk.keys())
        
        # # Convert action chunk to list of actions
        # # Based on modality.json, actions have same structure as states
        # action_modalities = ["left_leg", "right_leg", "waist", "left_arm", 
        #                    "left_hand", "right_arm", "right_hand"]
        
        # and emty numpy zero array list of 43 index made using np.zeros
        
        # # Get action horizon from first action modality
        # if f"action.{action_modalities[0]}" in action_chunk:
        #     action_horizon = action_chunk[f"action.{action_modalities[0]}"].shape[0]
            
        #     for i in range(action_horizon):
        #         action_dict = {}
        #         for modality in action_modalities:
        #             action_key = f"action.{modality}"
        #             if action_key in action_chunk:
        #                 action_dict[modality] = action_chunk[action_key][i]
        #         g1_actions.append(action_dict)

        # Expected action keys for unitree_g1 modality (we require these keys)
        required = [
            'action.left_hand',
            'action.right_hand',
            'action.left_arm',
            'action.right_arm',
        ]

        # If server returned a dict, validate keys
        if isinstance(action_chunk, dict):
            for k in required:
                if k not in action_chunk:
                    self.logger.error("Policy response missing required key: %s", k)
                    # return a safe zero action chunk of shape (16, 43)
                    return np.zeros((16, 43), dtype=np.float64)

            # determine horizon from one of the modalities
            try:
                horizon = int(np.asarray(action_chunk[required[0]]).shape[0])
            except Exception:
                horizon = 16

            # prepare container
            g1_actions = np.zeros((horizon, 43), dtype=np.float64)

            # fill in the arrays (convert to numpy arrays)
            try:
                arr_left = np.asarray(action_chunk['action.left_hand']).astype(np.float64)
                arr_right = np.asarray(action_chunk['action.right_hand']).astype(np.float64)
                arr_left_arm = np.asarray(action_chunk['action.left_arm']).astype(np.float64)
                arr_right_arm = np.asarray(action_chunk['action.right_arm']).astype(np.float64)

                # validate shapes: expected widths
                # left_hand/right_hand: 7 elements each -> indices 29:36 and 36:43
                # left_arm/right_arm: 7 elements each -> indices 15:22 and 22:29
                for arr, name, expected_len in [
                    (arr_left, 'action.left_hand', 7),
                    (arr_right, 'action.right_hand', 7),
                    (arr_left_arm, 'action.left_arm', 7),
                    (arr_right_arm, 'action.right_arm', 7),
                ]:
                    if arr.ndim != 2 or arr.shape[1] != expected_len:
                        # allow shape (H,) for single-step and broadcast
                        if arr.ndim == 1 and arr.shape[0] == expected_len:
                            arr = np.tile(arr[np.newaxis, :], (horizon, 1))
                        else:
                            self.logger.error("Unexpected shape for %s: %s", name, arr.shape)
                            return np.zeros((16, 43), dtype=np.float64)

                # assign into the action tensor (horizon x 43)
                g1_actions[:, 29:36] = arr_left[: g1_actions.shape[0], :]
                g1_actions[:, 36:43] = arr_right[: g1_actions.shape[0], :]
                g1_actions[:, 15:22] = arr_left_arm[: g1_actions.shape[0], :]
                g1_actions[:, 22:29] = arr_right_arm[: g1_actions.shape[0], :]

                return g1_actions
            except Exception as e:
                self.logger.exception("Failed to construct g1 action array: %s", e)
                return np.zeros((16, 43), dtype=np.float64)

        # If server returned an ndarray-like object, try to coerce and validate
        try:
            arr = np.asarray(action_chunk)
            if arr.ndim == 2 and arr.shape[1] == 43:
                return arr.astype(np.float64)
            else:
                self.logger.error("Policy returned array of unexpected shape: %s", arr.shape)
                return np.zeros((16, 43), dtype=np.float64)
        except Exception:
            self.logger.error("Policy returned unrecognized response type: %s", type(action_chunk))
            return np.zeros((16, 43), dtype=np.float64)


#################################################################################


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    policy_host: str = "0.0.0.0"  # host of the gr00t server
    policy_port: int = 5555  # port of the gr00t server
    action_horizon: int = 8  # number of actions to execute from the action chunk
    lang_instruction: str = "Pick the apple from the table and place it into the basket."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 60  # timeout in seconds
    show_images: bool = False  # whether to show images


@draccus.wrap()
def eval(cfg: EvalConfig):
    # Step 1: Initialize the robot
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    
    try:
        use_sim = False
        if use_sim:
            print("Initializing simulation mode...")
            ChannelFactoryInitialize(1, "lo")
        else:
            print("Initializing real robot mode...")
            ChannelFactoryInitialize(0)

        # Small delay to allow DDS to initialize
        time.sleep(0.5)
        
        custom = Custom(use_sim=use_sim)
        custom.Init()
        custom.Start()  # Start the control thread
        
        print("Robot initialized successfully")
        
        # Wait for robot state to be received
        print("Waiting for robot state...")
        timeout = 10
        start_time = time.time()
        while custom.low_state is None and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if custom.low_state is None:
            print("ERROR: Failed to receive robot state within timeout")
            return
        
        print("Robot state received!")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize robot: {e}")
        import traceback
        traceback.print_exc()
        return

    language_instruction = cfg.lang_instruction

    # Step 2: Initialize the policy
    try:
        policy = Gr00tRobotInferenceClient(
            host=cfg.policy_host,
            port=cfg.policy_port,
            show_images=cfg.show_images
        )
        print(f"Policy client connected to {cfg.policy_host}:{cfg.policy_port}")
    except Exception as e:
        print(f"ERROR: Failed to initialize policy client: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Run the Eval Loop
    # print(f"\nStarting evaluation with instruction: '{language_instruction}'")
    # print(f"Action horizon: {cfg.action_horizon}")
    
    try:
        iteration = 0
        # while iteration<3:
        while True:
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")
            
            # Get the realtime observation
            observation_dict = custom.get_observation_gr00t()
            # print("Observation keys:", observation_dict.keys())
            
            # Get action chunk from policy
            action_chunk = policy.get_action(observation_dict, language_instruction)
            
            # if not action_chunk:
            #     print("WARNING: No actions received from policy")
            #     continue
            
            print(f"Received {action_chunk.shape} actions")
            
            for i in range(16):
                joint_positions = action_chunk[i].tolist()
                left_leg     = joint_positions[0:6]
                right_leg    = joint_positions[6:12]
                waist        = joint_positions[12:15]
                right_arm     = joint_positions[15:22]
                left_hand    = joint_positions[29:36]
                left_arm    = joint_positions[22:29]
                right_hand   = joint_positions[36:43]
                
                pub_arr = left_leg + right_leg + waist + right_arm + left_arm + left_hand + right_hand
                print(len(pub_arr))
                custom.target_states = pub_arr
                time.sleep(0.02)  # 50Hz control loop
            
    except KeyboardInterrupt:
        print("\n\nStopping evaluation...")
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        # Add any cleanup code here if needed


if __name__ == "__main__":
    eval()
# EOF
