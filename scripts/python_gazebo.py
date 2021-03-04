#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import math

# gazebo default: meters, radians
FOOT = 0.3048 # meters

class PythonGazebo:
    """
    A collection of services for controlling a Gazebo robot via Python.
    """

    def __init__(self, model_name="basicjetbot", ros_rate=10):
        self.model_name = model_name

        rospy.init_node('python_gazebo')

        self.model_state_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self._state)
        self.contact_sensor_subscriber = rospy.Subscriber('/{}/contact_sensor'.format(model_name), ContactsState, self._collision)
        self.camera_subscriber = rospy.Subscriber('/{}/camera/image_raw'.format(model_name), Image, self._image)

        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher('/{}/cmd_vel'.format(model_name), Twist, queue_size=1)

        self.r = rospy.Rate(ros_rate)
        self.r.sleep()

    def stop(self):
        """
        Command the robot to stop moving.
        Returns the robot's final positional state.
        Returns:
            [float] final x-coordinate (meters)
            [float] final y-coordinate (meters)
            [float] final heading (radians)
        """
        self.cmd_vel_publisher.publish(Twist())
        self.r.sleep()
        return self.x, self.y, self.heading

    def rotate_to(self, heading, tol=0.04, max_speed=4, kP=10):
        """
        Command the robot to rotate itself to some absolute heading, then stop.
        Returns the robot's final positional state and a collision flag.
        Parameters:
            heading: target heading (radians, absolute)
            tol: final heading tolerance (radians)
            max_speed: maximum rotational speed (radians/second)
            kP: proportional gain constant
        Returns:
            [tuple]
             | [float] final x-coordinate (meters)
             | [float] final y-coordinate (meters)
             | [float] final heading (radians)
            [bool] collision detected? (T/F)
        """
        cmd = Twist()

        while not self.collision:
            err = normalize(heading - self.heading)

            if abs(err) < tol:
                return self.stop(), self.collision

            cmd.angular.z = min(max(kP * err, -max_speed), max_speed)

            self.cmd_vel_publisher.publish(cmd)
            self.r.sleep()

        return self.stop(), True

    def move_to(self, x, y, tol = 0.04, max_speed = 0.6, kP_lin = 1.5, kP_rot = 7.0):
        """
        Command the robot to move itself to a set of absolute x-y coordinates, then stop.
        Returns the robot's final positional state and a collision flag.
        Parameters:
            x: target x-coordinate (meters, absolute)
            y: target y-coordinate (meters, absolute)
            tol: final position tolerance (meters)
            max_speed: maximum linear speed (meters/second)
            kP_lin: proportional gain constant for linear movement
            kP_rot: proportional gain constant for rotational corrections
        Returns:
            [tuple]
             | [float] final x-coordinate (meters)
             | [float] final y-coordinate (meters)
             | [float] final heading (radians)
            [bool] collision detected? (T/F)
        """
        cmd = Twist()

        self.rotate_to(math.atan2(y - self.y, x - self.x))

        while not self.collision:
            err = math.sqrt((x - self.x)**2 + (y - self.y)**2)

            if err < tol:
                return self.stop(), self.collision

            x_err = x - self.x
            y_err = y - self.y
            cmd.angular.z = kP_rot * normalize(math.atan2(y_err, x_err) - self.heading)
            cmd.linear.x = min(kP_lin * err, max_speed)

            self.cmd_vel_publisher.publish(cmd)
            self.r.sleep()

        return self.stop(), True

    def teleport_to(self, x, y, heading):
        """
        Instantly move the robot to a specified poisition.
        Returns the robot's final positional state.
        Note: The robot will not accurately teleport into a collision state. Error greater
            than 0.01 between target and final coordinates typically indicates that a collision
            occurred at the target.
        Parameters:
            x: target x-coord (meters, absolute)
            y: target y-coord (meters, absolute)
            heading: target heading (radians, absolute)
        Returns:
            [float] final x-coordinate (meters)
            [float] final y-coordinate (meters)
            [float] final heading (radians)
        """
        state = ModelState()
        state.model_name = self.model_name
        pos = state.pose.position
        q = state.pose.orientation
        pos.x = x
        pos.y = y
        q.x, q.y, q.z, q.w = euler_to_quaternion(0, 0, heading)

        self.model_state_publisher.publish(state)
        self.r.sleep()
        self.r.sleep() #huh

        return self.stop()

    def get_raw_image(self):
        """
        Returns the robot's most recently transmitted image data as a list.
        Returns:
            [list] raw rgb image data
        """
        self.r.sleep()
        return self.img

    # --- internal functions (called automatically by ROS) ---
    def _state(self, msg):
        """
        Called automatically via subscription to '/gazebo/model_states' ROS channel.
        Updates internal tracking of the robot's positional state.
        """
        pose = msg.pose[msg.name.index(self.model_name)]
        pos = pose.position
        q = pose.orientation
        *_, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

        self.x = pos.x
        self.y = pos.y
        self.heading = yaw

    def _collision(self, msg):
        """
        Called automatically via subscription to '/{model_name}/contact_sensor' ROS channel.
        Updates internal tracking of the robot's current collision state.
        """
        self.collision = msg.states != []

    def _image(self, msg):
        """
        Called automatically via subscription to '/{model_name}/camera/image_raw' ROS channel.
        Updates internal tracking of the robot's most recently transmitted camera frame.
        """
        self.img = list(msg.data)

def normalize(radians):
    """
    Map an absolute radian angle to its equivalent value in the interval [-pi, pi].
    """
    while radians > math.pi:
        radians -= 2*math.pi
    while radians < -math.pi:
        radians += 2*math.pi
    return radians

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert a set of Euler angles (roll, pitch, yaw) to a unit quaternion (x, y, z, w).
    Algorithm source:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_to_quaternion_conversion
    """
    cr = math.cos(roll/2)
    sr = math.sin(roll/2)
    cp = math.cos(pitch/2)
    sp = math.sin(pitch/2)
    cy = math.cos(yaw/2)
    sy = math.sin(yaw/2)

    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    w = cr*cp*cy + sr*sp*sy

    return x, y, z, w

def quaternion_to_euler(x, y, z, w):
    """
    Convert a unit quaternion (x, y, z, w) to a set of Euler angles (roll, pitch, yaw).
    Algorithm source:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
    """
    sr_cp = 2 * (w*x + y*z)
    cr_cp = 1 - 2 * (x**2 + y**2)
    roll = math.atan2(sr_cp, cr_cp)

    sp = 2 * (w*y - z*x)
    sp = 1 if sp > 1 else sp
    sp = -1 if sp < -1 else sp
    pitch = math.asin(sp)

    sy_cp = 2 * (w*z + x*y)
    cy_cp = 1 - 2 * (y**2 + z**2)
    yaw = math.atan2(sy_cp, cy_cp)

    return roll, pitch, yaw
