import rospy
from galbot_ros_interfaces.srv import JSONService
from rm_msgs.msg import Gripper_Set, Gripper_Pick
import json
import numpy as np


def send_json(json_data, client, print_info=True):
    try:
        if print_info:
            rospy.loginfo(f"Sending request JSON string:\n{json_data}")
        response = client(json_data)

        if print_info:
            rospy.loginfo(f"Received response JSON string:\n{response.json_response}")
        # Get JSON data from response
        received_json_data = prase_json(response.json_response)
        return received_json_data
    except Exception as e:
        rospy.logerr(f"Failed to call service: {str(e)}")


def clean_json(test_json_data, test_json_data_string, print_info=True):
    test_json_data.clear()
    test_json_data_string = ""
    if print_info:
        rospy.loginfo("test_json_data and test_json_data_string cleared. Waiting for next test...")


def prase_json(json_data):
    json_data = json.loads(json_data)
    return json_data


def start_client():
    rospy.init_node("galbot_json_client_code")
    client = rospy.ServiceProxy("json_command_service", JSONService)
    rospy.wait_for_service("json_command_service", timeout=10)
    return client



class Gripper():
    """Gripper Class"""
    def __init__(self) -> None:
        self.open_gripper()
        self.close_gripper()
        self.open_gripper()
    
    def _set_gripper_position(self, position: int):
        pub = rospy.Publisher('/rm_driver_left/Gripper_Set', Gripper_Set, queue_size=10)
        msg = Gripper_Set()
        msg.position = position
        pub.publish(msg)
        rospy.loginfo(f"Set gripper position: {position}")
        rospy.sleep(2.0)
    
    def _set_gripper_force_speed(self, force: int, speed: int):
        pub = rospy.Publisher('/rm_driver_left/Gripper_Pick', Gripper_Pick, queue_size=10)
        msg = Gripper_Pick()
        msg.force = force
        msg.speed = speed
        pub.publish(msg)
        rospy.loginfo(f"Set gripper force: {force}, speed: {speed}")
        rospy.sleep(2.0)

    def open_gripper(self):
        self._set_gripper_position(1000)

    def close_gripper(self):
        self._set_gripper_position(0)
    
    def set_gripper(self, position: int):
        self._set_gripper_position(position)


class CommandSender():
    def __init__(self, use_gripper=True) -> None:
        self.client = start_client()
        if use_gripper:
            self.gripper = Gripper()
    
    def _get_qpos(self, print_info=False):
        if print_info:
            rospy.logwarn("Testing command: get_arm_joint_angle LEFT ARM")

        test_json_data_string = json.dumps({
            "hardware_type": "left_arm",
            "command": "get_arm_joint_angle"
        })
        get_arm_joint_angle_result = send_json(test_json_data_string, self.client, print_info)
        joint_angle_1 = get_arm_joint_angle_result["data"]["joint_1_angle"]
        joint_angle_2 = get_arm_joint_angle_result["data"]["joint_2_angle"]
        joint_angle_3 = get_arm_joint_angle_result["data"]["joint_3_angle"]
        joint_angle_4 = get_arm_joint_angle_result["data"]["joint_4_angle"]
        joint_angle_5 = get_arm_joint_angle_result["data"]["joint_5_angle"]
        joint_angle_6 = get_arm_joint_angle_result["data"]["joint_6_angle"]

        if print_info:
            rospy.loginfo(f"joint_angle_1: {joint_angle_1}")
            rospy.loginfo(f"joint_angle_2: {joint_angle_2}")
            rospy.loginfo(f"joint_angle_3: {joint_angle_3}")
            rospy.loginfo(f"joint_angle_4: {joint_angle_4}")
            rospy.loginfo(f"joint_angle_5: {joint_angle_5}")
            rospy.loginfo(f"joint_angle_6: {joint_angle_6}")

        clean_json({}, test_json_data_string, print_info)
        return np.array([joint_angle_1, joint_angle_2, joint_angle_3, joint_angle_4, joint_angle_5, joint_angle_6])

    def get_qpos(self):
        return self._get_qpos()
    
    def move_arm(self, qpos, print_info=True, speed=0.01, wait=True):
        if print_info:
            rospy.logwarn("Testing left arm...")
            rospy.logwarn("Testing command: set_arm_joint_angle LEFT ARM")

        test_json_data_string = json.dumps({
            "hardware_type": "left_arm",
            "command": "set_arm_joint_angle",
            "parameters": {
                "speed": speed,
                "joint_1_angle": qpos[0],
                "joint_2_angle": qpos[1],
                "joint_3_angle": qpos[2],
                "joint_4_angle": qpos[3],
                "joint_5_angle": qpos[4],
                "joint_6_angle": qpos[5]
            }
        })
        send_json(test_json_data_string, self.client, print_info)
        clean_json({}, test_json_data_string, print_info)
        
        sleep = 1 / 60
        threshold = 1e-2
        
        if wait:
            while np.max(np.abs(np.array(qpos) - np.array(self._get_qpos(print_info=False)))) > threshold:
                rospy.sleep(sleep)  

    def open_gripper(self):
        self.gripper.open_gripper()

    def close_gripper(self):
        self.gripper.close_gripper()


def main():
    sender = CommandSender()
    target = [0., 0., 0., 0., 0., 0.]
    sender.move_arm(target, speed=0.1)

    target = [0., 0., 0., 0., 0.2, 0.]
    sender.move_arm(target, speed=0.1)

    target = [0., 0., 0., 0., 0., 0.]
    sender.move_arm(target, speed=0.1)

    sender.open_gripper()
    sender.close_gripper()
    sender.open_gripper()


if __name__ == "__main__":
    main()
