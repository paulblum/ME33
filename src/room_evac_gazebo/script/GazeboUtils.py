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
        self.model_state_subscriber = rospy.Subscriber('/odom', Odometry, self.__update_state)
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.r = rospy.Rate(10)
        self.r.sleep()

    def __update_state(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        (roll, pitch, yaw) = quaternion_to_euler(q.x, q.y, q.z, q.w)
        
        self.x = pos.x
        self.y = pos.y
        self.heading = yaw

    def stop(self):
        self.cmd_vel_publisher.publish(Twist())
        return self.x, self.y, self.heading

    def set_state(self, x, y, heading = 0):
        state = ModelState()
        state.model_name = "basicjetbot"
        pos = state.pose.position
        q = state.pose.orientation
        pos.x = x
        pos.y = y
        q.x, q.y, q.z, q.w = euler_to_quaternion(0, 0, math.radians(heading))
        self.model_state_publisher.publish(state)

    def rotate_to(self, target, tol = 0.04, max_speed = 4, kP = 10.0):
        cmd = Twist()

        while True:
            rad_remaining = normalize(target - self.heading)
            
            if abs(rad_remaining) < tol:
                return self.stop()
            
            cmd.angular.z = min(max(kP * rad_remaining, -max_speed), max_speed)

            self.cmd_vel_publisher.publish(cmd)
            self.r.sleep()

    def move_to(self, x, y, tol = 0.04, max_speed = 0.6, kP_lin = 1.5, kP_rot = 7.0):
        cmd = Twist()

        self.rotate_to(math.atan2(y - self.y, x - self.x), tol = 0.07)

        while True:
            dist_remaining = math.sqrt((x - self.x)**2 + (y - self.y)**2)

            if dist_remaining < tol:
                return self.stop()
            
            x_err = x - self.x
            y_err = y - self.y
            cmd.angular.z = kP_rot * normalize(math.atan2(y_err, x_err) - self.heading)
            cmd.linear.x = min(kP_lin * dist_remaining, max_speed)

            self.cmd_vel_publisher.publish(cmd)
            self.r.sleep()

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
