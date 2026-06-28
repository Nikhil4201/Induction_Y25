#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
import math
import time


class LeaderEvasionNode(Node):

    def __init__(self):
        super().__init__('leader_evasion_node')

        self._vel_pub = self.create_publisher(
            Twist, '/iris_1/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        self.create_subscription(State, '/iris_1/mavros/state', self._state_callback, 10)

        self._set_mode_client = self.create_client(SetMode,    '/iris_1/mavros/set_mode')
        self._arm_client      = self.create_client(CommandBool,'/iris_1/mavros/cmd/arming')

        self._armed   = False
        self._t_start = None

        self.create_timer(0.05, self._control_loop)
        self.get_logger().info('LeaderEvasionNode ready...')

    def _state_callback(self, msg):
        if msg.connected and not self._armed:
            self._setup_offboard_and_arm()

    def _setup_offboard_and_arm(self):
        self.get_logger().info('Arming iris_1 and switching to OFFBOARD...')
        for _ in range(30):
            self._vel_pub.publish(Twist())
        if self._set_mode_client.wait_for_service(timeout_sec=3.0):
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            self._set_mode_client.call_async(req)
        if self._arm_client.wait_for_service(timeout_sec=3.0):
            req = CommandBool.Request()
            req.value = True
            self._arm_client.call_async(req)
        self._armed   = True
        self._t_start = time.time()
        self.get_logger().info('iris_1 armed — evasion starting!')

    def _control_loop(self):
        if not self._armed or self._t_start is None:
            self._vel_pub.publish(Twist())
            return

        t      = time.time() - self._t_start
        t_loop = ((t - 10.0) % 80.0) if t >= 10.0 else -1

        if t < 10.0:
            vx, vy, vz = 0.0, 0.0, 0.5

        elif t_loop < 30.0:
            omega = 2 * math.pi / 15.0
            vx =  1.5 * math.cos(omega * t_loop)
            vy =  1.5 * math.sin(2 * omega * t_loop)
            vz =  0.3 * math.sin(omega * t_loop * 0.5)

        elif t_loop < 55.0:
            ts = t_loop - 30.0
            r  = 0.5 + ts * 0.06
            omega = 2 * math.pi / 8.0
            vx =  r * math.cos(omega * ts)
            vy =  r * math.sin(omega * ts)
            vz =  0.4 * math.sin(omega * ts * 0.7)

        else:
            ts = t_loop - 55.0
            vx = 2.0 * (1 if int(ts / 3) % 2 == 0 else -1)
            vy = 2.0 * math.sin(2 * math.pi * ts / 5.0)
            vz = 0.0

        max_v = 2.5
        vx = float(max(min(vx, max_v), -max_v))
        vy = float(max(min(vy, max_v), -max_v))
        vz = float(max(min(vz, 0.8),  -0.8))

        twist = Twist()
        twist.linear = Vector3(x=vx, y=vy, z=vz)
        self._vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LeaderEvasionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
