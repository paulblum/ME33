#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
import math

# gazebo default: meters, radians
FOOT = 0.3048 # meters

class Teleport:
    """
    Teleport your Gazebo models.
    """

    def __init__(self, model_name="basicjetbot", ros_rate=100):
        self.model_name = model_name
        rospy.init_node('Teleport')
        self.model_state_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self._update_state)
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.r = rospy.Rate(ros_rate)
        self.r.sleep()
        
    def to(self, x, y, heading, tol = 0.01):
        """
        Parameters:
            x: target x-coord (meters, absolute)
            y: target y-coord (meters, absolute)
            heading: target heading (radians, absolute)
            tol: desired tolerance
        Returns:
            [bool] destination within tolerance? (T/F)
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
        
        x_err = abs(x - self.x)
        y_err = abs(y - self.y)
        head_err = abs(normalize(heading - self.heading))
        return (x_err < tol and y_err < tol and head_err < 10*tol)

    def _update_state(self, msg):
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
