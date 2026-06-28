#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Vector3
from mavros_msgs.msg import MountControl, State
from mavros_msgs.srv import SetMode, CommandBool
from std_msgs.msg import Float64
import time


class PIDController:
    def __init__(self, kp, ki, kd, output_limit=2.0, integral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0
        p = self.kp * error
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -self.integral_limit, self.integral_limit))
        i = self.ki * self._integral
        d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error
        output = p + i + d
        return float(np.clip(output, -self.output_limit, self.output_limit))


class FollowerNode(Node):
    ARUCO_DICT        = cv2.aruco.DICT_4X4_50
    MARKER_LENGTH     = 0.15
    DESIRED_DISTANCE  = 2.0
    LOSS_TIMEOUT      = 5.0
    LANDING_MARKER_ID = 99
    LANDING_ALTITUDE  = -0.3
    STATE_FOLLOW      = "FOLLOW"
    STATE_LANDING     = "LANDING"

    def __init__(self):
        super().__init__('follower_node')

        self.declare_parameter('desired_distance', self.DESIRED_DISTANCE)
        self.declare_parameter('marker_length',    self.MARKER_LENGTH)
        self.declare_parameter('kp_xy',  0.6)
        self.declare_parameter('ki_xy',  0.02)
        self.declare_parameter('kd_xy',  0.15)
        self.declare_parameter('kp_z',   0.8)
        self.declare_parameter('ki_z',   0.02)
        self.declare_parameter('kd_z',   0.2)
        self.declare_parameter('kp_yaw', 0.4)
        self.declare_parameter('ki_yaw', 0.005)
        self.declare_parameter('kd_yaw', 0.1)
        self.declare_parameter('use_gimbal', True)

        self._desired_dist = self.get_parameter('desired_distance').value
        self._marker_len   = self.get_parameter('marker_length').value

        # ArUco — works on both Humble (OpenCV 4.x) and Jazzy
        aruco_dict   = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        try:
            self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            self._use_new_api = True
        except AttributeError:
            # Fallback for older OpenCV builds
            self._aruco_dict   = aruco_dict
            self._aruco_params = aruco_params
            self._use_new_api  = False

        self._camera_matrix = None
        self._dist_coeffs   = None

        kp_xy  = self.get_parameter('kp_xy').value
        ki_xy  = self.get_parameter('ki_xy').value
        kd_xy  = self.get_parameter('kd_xy').value
        self._pid_x   = PIDController(kp_xy,  ki_xy,  kd_xy)
        self._pid_y   = PIDController(kp_xy,  ki_xy,  kd_xy)
        self._pid_z   = PIDController(self.get_parameter('kp_z').value,
                                      self.get_parameter('ki_z').value,
                                      self.get_parameter('kd_z').value)
        self._pid_yaw = PIDController(self.get_parameter('kp_yaw').value,
                                      self.get_parameter('ki_yaw').value,
                                      self.get_parameter('kd_yaw').value)
        self._pid_gimbal_pitch = PIDController(0.003, 0.0, 0.001, output_limit=30.0)
        self._pid_gimbal_yaw   = PIDController(0.003, 0.0, 0.001, output_limit=30.0)

        self._bridge          = CvBridge()
        self._last_seen       = time.time()
        self._prev_time       = time.time()
        self._mission_state   = self.STATE_FOLLOW
        self._marker_visible  = False
        self._img_cx          = 320.0
        self._img_cy          = 240.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image,      '/iris_2/camera/image_raw',    self._image_callback,       sensor_qos)
        self.create_subscription(CameraInfo, '/iris_2/camera/camera_info',  self._camera_info_callback, sensor_qos)
        self.create_subscription(State,      '/iris_2/mavros/state',        self._state_callback,       10)

        self._vel_pub          = self.create_publisher(Twist,        '/iris_2/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self._gimbal_pub       = self.create_publisher(MountControl, '/iris_2/mavros/mount_control/command',               10)
        self._gimbal_pitch_pub = self.create_publisher(Float64,      '/iris_2/gimbal_pitch_controller/command',            10)
        self._gimbal_yaw_pub   = self.create_publisher(Float64,      '/iris_2/gimbal_yaw_controller/command',              10)

        self.create_timer(0.05, self._heartbeat_callback)
        self.create_timer(1.0,  self._watchdog_callback)

        self._set_mode_client = self.create_client(SetMode,    '/iris_2/mavros/set_mode')
        self._arm_client      = self.create_client(CommandBool,'/iris_2/mavros/cmd/arming')
        self._mavros_state    = State()
        self._offboard_set    = False

        self.get_logger().info('FollowerNode started — waiting for camera info...')

    def _state_callback(self, msg):
        self._mavros_state = msg
        if msg.connected and not self._offboard_set:
            self._enable_offboard_and_arm()

    def _camera_info_callback(self, msg):
        if self._camera_matrix is None:
            self._camera_matrix = np.array(msg.k).reshape(3, 3)
            self._dist_coeffs   = np.array(msg.d)
            self._img_cx = msg.width  / 2.0
            self._img_cy = msg.height / 2.0
            self.get_logger().info(f'Camera ready — centre ({self._img_cx}, {self._img_cy})')

    def _detect(self, gray):
        if self._use_new_api:
            return self._detector.detectMarkers(gray)
        else:
            return cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)

    def _image_callback(self, msg):
        if self._camera_matrix is None:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        now   = time.time()
        dt    = now - self._prev_time
        self._prev_time = now
        if dt <= 0 or dt > 1.0:
            dt = 0.033

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)

        if ids is not None:
            target_id = self.LANDING_MARKER_ID if self._mission_state == self.STATE_LANDING else None
            chosen_corners = None

            if target_id is None:
                chosen_corners = corners[0]
            else:
                for i, mid in enumerate(ids.flatten()):
                    if mid == target_id:
                        chosen_corners = corners[i]
                        break

            if chosen_corners is not None:
                self._last_seen      = now
                self._marker_visible = True

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [chosen_corners], self._marker_len,
                    self._camera_matrix, self._dist_coeffs
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]

                x_err = tvec[0]
                y_err = tvec[1]
                z_err = tvec[2] - self._desired_dist

                cx = chosen_corners[0][:, 0].mean()
                cy = chosen_corners[0][:, 1].mean()
                self._control_gimbal(cx - self._img_cx, cy - self._img_cy, dt)

                if self._mission_state == self.STATE_LANDING:
                    self._do_landing(tvec, dt)
                else:
                    self._do_follow(x_err, y_err, z_err, tvec[0], dt)

                cv2.aruco.drawDetectedMarkers(frame, [chosen_corners])
                cv2.drawFrameAxes(frame, self._camera_matrix, self._dist_coeffs, rvec, tvec, 0.1)
                cv2.putText(frame,
                    f"X:{tvec[0]:.2f} Y:{tvec[1]:.2f} Z:{tvec[2]:.2f}m | {self._mission_state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                self._marker_visible = False
                self._publish_hover()
        else:
            self._marker_visible = False
            self._publish_hover()

        h, w = frame.shape[:2]
        cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0,0,255), 1)
        cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0,0,255), 1)
        cv2.imshow('iris_2 ArUco Tracker', frame)
        cv2.waitKey(1)

    def _heartbeat_callback(self):
        if not self._marker_visible:
            self._publish_hover()

    def _watchdog_callback(self):
        elapsed = time.time() - self._last_seen
        if elapsed > self.LOSS_TIMEOUT:
            self.get_logger().error(f'Marker lost for {elapsed:.1f}s — MISSION FAILED!')
            self._publish_hover()

    def _do_follow(self, x_err, y_err, z_err, yaw_err, dt):
        vx  =  self._pid_z.update(z_err,    dt)
        vy  = -self._pid_x.update(x_err,    dt)
        vz  = -self._pid_y.update(y_err,    dt)
        yaw =  self._pid_yaw.update(yaw_err, dt)
        twist = Twist()
        twist.linear  = Vector3(x=float(vx), y=float(vy), z=float(vz))
        twist.angular = Vector3(x=0.0, y=0.0, z=float(yaw))
        self._vel_pub.publish(twist)

    def _do_landing(self, tvec, dt):
        vx = self._pid_z.update(tvec[2], dt)
        vy = -self._pid_x.update(tvec[0], dt)
        vz = self.LANDING_ALTITUDE
        twist = Twist()
        twist.linear = Vector3(x=float(vx), y=float(vy), z=float(vz))
        self._vel_pub.publish(twist)

    def _control_gimbal(self, pixel_err_x, pixel_err_y, dt):
        if not self.get_parameter('use_gimbal').value:
            return
        delta_pitch = -self._pid_gimbal_pitch.update(pixel_err_y, dt)
        delta_yaw   =  self._pid_gimbal_yaw.update(pixel_err_x,   dt)
        mount_msg = MountControl()
        mount_msg.header.stamp = self.get_clock().now().to_msg()
        mount_msg.mode  = 2
        mount_msg.pitch = float(delta_pitch)
        mount_msg.yaw   = float(delta_yaw)
        mount_msg.roll  = 0.0
        self._gimbal_pub.publish(mount_msg)
        pitch_msg = Float64(); pitch_msg.data = float(np.radians(delta_pitch))
        yaw_msg   = Float64(); yaw_msg.data   = float(np.radians(delta_yaw))
        self._gimbal_pitch_pub.publish(pitch_msg)
        self._gimbal_yaw_pub.publish(yaw_msg)

    def _publish_hover(self):
        self._vel_pub.publish(Twist())

    def trigger_landing_mode(self):
        self.get_logger().info('Switching to LANDING mode!')
        self._mission_state = self.STATE_LANDING
        for pid in (self._pid_x, self._pid_y, self._pid_z, self._pid_yaw):
            pid.reset()

    def _enable_offboard_and_arm(self):
        self.get_logger().info('Enabling OFFBOARD and arming iris_2...')
        for _ in range(20):
            self._publish_hover()
        if self._set_mode_client.wait_for_service(timeout_sec=2.0):
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            self._set_mode_client.call_async(req)
        if self._arm_client.wait_for_service(timeout_sec=2.0):
            req = CommandBool.Request()
            req.value = True
            self._arm_client.call_async(req)
        self._offboard_set = True

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
