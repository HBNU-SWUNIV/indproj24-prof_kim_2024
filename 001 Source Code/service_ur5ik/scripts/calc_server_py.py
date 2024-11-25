#!/usr/bin/env python3
import roboticstoolbox as rtb
from spatialmath import *

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from interfaces_ur5ik.srv import SixTheta

class calc_server(Node):

    def __init__(self):
        super().__init__('calc_server')
        self.m_server = self.create_service(
            SixTheta,
            'service_ik',
            self.service_callback)

    def service_callback(self, request, response):
        x = request.srv_target[0]
        y = request.srv_target[1]
        z = request.srv_target[2]
        roll = request.srv_target[3]
        pitch = request.srv_target[4]
        yaw = request.srv_target[5]

        robot = rtb.models.UR5()
        Tep = SE3.Trans(x, y, z) * SE3.Rz(yaw, 'rad') * SE3.Ry(pitch, 'rad') * SE3.Rx(roll, 'rad')
        sol = robot.ik_LM(Tep)         # solve IK

        theta1 = sol[0][0]
        theta2 = sol[0][1]
        theta3 = sol[0][2]
        theta4 = sol[0][3]
        theta5 = sol[0][4]
        theta6 = sol[0][5]

        response.srv_theta[0] = theta1
        response.srv_theta[1] = theta2
        response.srv_theta[2] = theta3
        response.srv_theta[3] = theta4
        response.srv_theta[4] = theta5
        response.srv_theta[5] = theta6

        self.get_logger().info('theta1 : "%f"' % theta1)
        self.get_logger().info('theta2 : "%f"' % theta2)
        self.get_logger().info('theta3 : "%f"' % theta3)
        self.get_logger().info('theta4 : "%f"' % theta4)
        self.get_logger().info('theta5 : "%f"' % theta5)
        self.get_logger().info('theta6 : "%f"' % theta6)
        self.get_logger().info('-----------------')

        return response


def main():
    rclpy.init()
    rclpy.spin(calc_server())
    rclpy.shutdown()

if __name__ == '__main__':
    main()