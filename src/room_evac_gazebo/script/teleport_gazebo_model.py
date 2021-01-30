#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelState, ContactsState
import math

# gazebo default: meters, radians
FOOT = 0.3048 # meters

class Teleport:
    """
    Teleport your Gazebo models.
    """

    def __init__(self, model_name="basicjetbot"):
        self.model_name = model_name
        self.collision_detected = False

        rospy.init_node('Teleport')
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.contact_sensor_subscriber = rospy.Subscriber('/contact_sensor', ContactsState, self._collision_check)
        self.r = rospy.Rate(40)
        self.r.sleep()
    
    def to(self, x, y, heading):
        """
        Parameters:
            x: target x-coord (meters, absolute)
            y: target y-coord (meters, absolute)
            heading: target heading (radians, absolute)
        Returns:
            [bool] collision detected on teleport? (T/F)
        """
        state = ModelState()
        state.model_name = self.model_name
        pos = state.pose.position
        q = state.pose.orientation
        pos.x = x
        pos.y = y
        q.x, q.y, q.z, q.w = euler_to_quaternion(0, 0, heading)
        self.collision_detected = False
        self.model_state_publisher.publish(state)
        self.r.sleep()
        return self.collision_detected

    def _collision_check(self, msg):
        """
        Called automatically via subscription to '/contact_sensor' ROS channel.
        Updates internal tracking of detected collisions.
        """
        if msg.states != []:
            self.collision_detected = True

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
