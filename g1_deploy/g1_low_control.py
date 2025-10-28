import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import AudioData_


from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

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

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Custom:
    def __init__(self, use_sim=False):
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
        self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.left_hand_state = None
        self.right_hand_state = None
        self.low_state = None 
        self.update_mode_machine_ = False
        self.target_states = [0.0] * 43
        self.rgb_image = None
        self.depth_image = None
        self.width, self.height = 640, 480

        self.crc = CRC()
        self.use_sim = use_sim


    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        if not self.use_sim:
            status, result = self.msc.CheckMode()
            # time.sleep(1)
            while result['name']:
                self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.right_hand_cmd_publisher_ = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.right_hand_cmd_publisher_.Init()
        self.left_hand_cmd_publisher_ = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self.left_hand_cmd_publisher_.Init()


        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.lefhandstate_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.lefhandstate_subscriber.Init(self.LeftHandStateHandler, 10)
        self.righthandstate_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self.righthandstate_subscriber.Init(self.RightHandStateHandler, 10)

        self.rgb_sub = ChannelSubscriber("rt/camera/rgb", AudioData_)
        self.depth_sub = ChannelSubscriber("rt/camera/depth", AudioData_)
        
        self.rgb_sub.Init(self._sdk2_rgb_handler, 10)
        self.depth_sub.Init(self._sdk2_depth_handler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        self.counter_ +=1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            # print(self.low_state.imu_state.rpy)

    def LeftHandStateHandler(self, msg: HandState_):
        self.left_hand_state = msg
        # print("Left hand state received")
    
    def RightHandStateHandler(self, msg: HandState_):
        self.right_hand_state = msg
        # print("Right hand state received")

    def _sdk2_rgb_handler(self, msg: AudioData_):
        """Handle RGB camera messages from SDK2"""
        # print(len(msg.data))
        try:
            if len(msg.data) > 0:
                # Convert list of integers to numpy array
                img_array = np.array(msg.data, dtype=np.uint8)
                
                # Reshape to image dimensions (height, width, channels)
                if len(img_array) >= self.height * self.width * 3:
                    self.rgb_image = img_array[:self.height*self.width*3].reshape(
                        (self.height, self.width, 3))
                    # self.rgb_frame_count += 1

                # self.rgb_image = img_array
                    
        except Exception as e:
            print(f"RGB handler error: {e}")
    
    def _sdk2_depth_handler(self, msg: AudioData_):
        """Handle depth camera messages from SDK2"""
        try:
            if len(msg.data) > 0:
                # Convert list of integers to numpy array
                img_array = np.array(msg.data, dtype=np.uint8)
                
                # Depth is single channel
                if len(img_array) >= self.height * self.width:
                    self.depth_image = img_array[:self.height*self.width].reshape(
                        (self.height, self.width))
                    # self.depth_frame_count += 1
                    
        except Exception as e:
            print(f"Depth handler error: {e}")

    def get_rgb_image(self):
        """
        Get the latest RGB image
        
        Returns:
            numpy.ndarray: RGB image (H, W, 3) or None if no data
        """
        return self.rgb_image
    
    def get_depth_image(self):
        """
        Get the latest depth image
        
        Returns:
            numpy.ndarray: Depth image (H, W) or None if no data
        """
        return self.depth_image

    def LowCmdWrite(self):
        """
        This function runs in its own RecurrentThread (~every control_dt_).
        After it publishes a command that corresponds to a pending_seq we mark it as sent
        and set the event so the ROS timer callback can continue.
        """
        self.time_ += self.control_dt_

        # build low_cmd for stage 1 or stage 2
        if self.time_ < self.duration_:
            # stage 1: move to zero posture (interpolate from current state)
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1
                self.low_cmd.motor_cmd[i].tau = 0.
                if self.low_state is not None:
                    self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]
            
            for i in range(7):
                self.left_hand_cmd.motor_cmd[i].mode = 1
                self.left_hand_cmd.motor_cmd[i].tau = 0.0
                self.left_hand_cmd.motor_cmd[i].q = 0.0
                self.left_hand_cmd.motor_cmd[i].dq = 0.0
                self.left_hand_cmd.motor_cmd[i].kp = 20.0
                self.left_hand_cmd.motor_cmd[i].kd = 0.5
            
            for i in range(7):
                self.right_hand_cmd.motor_cmd[i].mode = 1
                self.right_hand_cmd.motor_cmd[i].tau = 0.0
                self.right_hand_cmd.motor_cmd[i].q = 0.0
                self.right_hand_cmd.motor_cmd[i].dq = 0.0
                self.right_hand_cmd.motor_cmd[i].kp = 20.0
                self.right_hand_cmd.motor_cmd[i].kd = 0.5

        else:
            # stage 2: move to desired posture
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1
                self.low_cmd.motor_cmd[i].tau = 0.
                # ensure target_states length
                if len(self.target_states) >= G1_NUM_MOTOR:
                    self.low_cmd.motor_cmd[i].q = float(self.target_states[i])
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]
            
            for i in range(7):
                self.left_hand_cmd.motor_cmd[i].mode = 1
                self.left_hand_cmd.motor_cmd[i].tau = 0.0
                if len(self.target_states) == 43:
                    self.left_hand_cmd.motor_cmd[i].q = float(self.target_states[29 + i])
                else:
                    self.left_hand_cmd.motor_cmd[i].q = 0.0
                self.left_hand_cmd.motor_cmd[i].dq = 0.0
                self.left_hand_cmd.motor_cmd[i].kp = 20.0
                self.left_hand_cmd.motor_cmd[i].kd = 0.5

            for i in range(7):
                self.right_hand_cmd.motor_cmd[i].mode = 1
                self.right_hand_cmd.motor_cmd[i].tau = 0.0
                if len(self.target_states) == 43:
                    self.right_hand_cmd.motor_cmd[i].q = float(self.target_states[36 + i])
                else:
                    self.right_hand_cmd.motor_cmd[i].q = 0.0
                self.right_hand_cmd.motor_cmd[i].dq = 0.0
                self.right_hand_cmd.motor_cmd[i].kp = 20.0
                self.right_hand_cmd.motor_cmd[i].kd = 0.5

        # compute crc and publish
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
        # self.left_hand_cmd.crc = self.crc.Crc(self.left_hand_cmd)
        self.left_hand_cmd_publisher_.Write(self.left_hand_cmd)
        # self.right_hand_cmd.crc = self.crc.Crc(self.right_hand_cmd)
        self.right_hand_cmd_publisher_.Write(self.right_hand_cmd)
    
    def get_observation_gr00t(self):
        observation_dict = {
            "state": {
                "left_leg": np.zeros((6,), dtype=np.float32),
                "right_leg": np.zeros((6,), dtype=np.float32),
                "waist": np.zeros((3,), dtype=np.float32),
                "left_arm": np.zeros((7,), dtype=np.float32),
                "left_hand": np.zeros((7,), dtype=np.float32),
                "right_arm": np.zeros((7,), dtype=np.float32),
                "right_hand": np.zeros((7,), dtype=np.float32)
            },
            "video": {
                "rs_view": np.zeros((480, 640, 3), dtype=np.uint8)
            },
            "annotation": {
                "human.task_description": ""
            }
        }

        img = self.get_rgb_image()
        if img is not None:
            print(f"\n\n {img.shape} \n\n")
        observation_dict["video"]["rs_view"] = img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # get joint positions from low_state
        if self.low_state is not None:
            body_joint_positions = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)])
           
            # collect left and right hand joint positions
            left_hand_joint_positions = np.array([self.left_hand_state.motor_state[i].q for i in range(7)]) \
            if self.left_hand_state is not None else np.zeros((7,), dtype=np.float32)

            right_hand_joint_positions = np.array([self.right_hand_state.motor_state[i].q for i in range(7)]) \
            if self.right_hand_state is not None else np.zeros((7,), dtype=np.float32)

            joint_positions = np.concatenate([body_joint_positions, left_hand_joint_positions, right_hand_joint_positions], axis=0)

            # Extract parts from original ordering
            left_leg     = joint_positions[0:6]
            right_leg    = joint_positions[6:12]
            waist        = joint_positions[12:15]
            left_arm     = joint_positions[15:22]
            left_hand    = joint_positions[22:29]
            right_arm    = joint_positions[29:36]
            right_hand   = joint_positions[36:43]

            # Store in the correct structured state dict
            observation_dict["state"]["left_leg"] = left_leg
            observation_dict["state"]["right_leg"] = right_leg
            observation_dict["state"]["waist"] = waist
            observation_dict["state"]["left_arm"] = left_arm
            observation_dict["state"]["left_hand"] = left_hand
            observation_dict["state"]["right_arm"] = right_arm
            observation_dict["state"]["right_hand"] = right_hand

        return observation_dict


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    use_sim = False
    
    if len(sys.argv)>1:
        ChannelFactoryInitialize(1, sys.argv[1])
        use_sim = True
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(use_sim=use_sim)
    custom.Init()
    custom.Start()

    while True:        
        time.sleep(1)