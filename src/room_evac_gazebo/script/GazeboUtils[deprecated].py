#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
import math
import time

class DiffDriveControl:
    def __init__(self):
        rospy.init_node('diff_drive_control')
        self.sub = rospy.Subscriber('/odom', Odometry, self.__update_pose)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.pub_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.r = rospy.Rate(10)
        self.r.sleep()

    def __update_pose(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        (roll, pitch, yaw) = quaternion_to_euler(q.x,q.y,q.z,q.w)
        self.pose = [pos.x, pos.y, yaw]

    def __get_rotate_cmd(self, target, kP, tol):
        rotate_cmd = Twist()
        yaw_err = normalize(target - self.pose[2])
        if abs(yaw_err) > tol:
            rotate_cmd.angular.z = kP * yaw_err
        return rotate_cmd

    def set_state(self, x, y, heading = 0):
        state = ModelState()
        state.model_name = "basicjetbot"
        pos = state.pose.position
        q = state.pose.orientation
        pos.x = x
        pos.y = y
        q.x, q.y, q.z, q.w = euler_to_quaternion(0, 0, math.radians(heading))
        self.pub_state.publish(state)

    def stop(self):
        self.cmd_vel.publish(Twist())
        return self.pose

    def rotate(self, degrees, kP = 0.8, tol = 0.05):
        target = self.pose[2] + math.radians(degrees)
        while True:
            command = self.__get_rotate_cmd(target, kP, tol)
            if command.angular.z == 0:
                return self.stop()
            self.cmd_vel.publish(command)
            self.r.sleep()

    def move_forward(self, distance, maxSpeed = 0.2, kP_lin = 0.3, tol_lin = 0.08, kP_rot = 10, tol_rot = 0):
        start_pose = self.pose
        while True:
            dist_remaining = distance - distance_btwn(start_pose, self.pose)
            if dist_remaining < tol_lin:
                return self.stop()
            speed = min(kP_lin * dist_remaining, maxSpeed)
            command = self.__get_rotate_cmd(start_pose[2], speed * kP_rot, tol_rot)
            command.linear.x = min(kP_lin * dist_remaining, maxSpeed)
            self.cmd_vel.publish(command)
            self.r.sleep()

def distance_btwn(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2)

def normalize(radians):
    while radians > math.pi:
        radians -= 2*math.pi
    while radians < -math.pi:
        radians += 2*math.pi
    return radians

def euler_to_quaternion(roll, pitch, yaw):
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
