#!/usr/bin/env python3

import csv
import math
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage


class TrajectoryRecorder(Node):

    def __init__(self):
        super().__init__('trajectory_recorder')

        self.odom_topic = '/model/vehicle/odometry'
        self.ground_truth_topic = '/world/empty/pose/info'

        self.results_dir = os.path.expanduser(
            '~/ros2_jazzy_ws/src/Intelligant_Robotics_Assign_1/results'
        )

        os.makedirs(self.results_dir, exist_ok=True)

        self.odom_path = os.path.join(
            self.results_dir, 'odometry.csv'
        )

        self.ground_truth_path = os.path.join(
            self.results_dir, 'ground_truth.csv'
        )

        self.odom_file = open(
            self.odom_path, 'w', newline=''
        )

        self.ground_truth_file = open(
            self.ground_truth_path, 'w', newline=''
        )

        self.odom_writer = csv.writer(self.odom_file)
        self.ground_truth_writer = csv.writer(
            self.ground_truth_file
        )

        header = [
            'time', 'x', 'y', 'z', 'yaw_rad', 'yaw_deg'
        ]

        self.odom_writer.writerow(header)
        self.ground_truth_writer.writerow(header)

        self.odom_count = 0
        self.ground_truth_count = 0

        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            100
        )

        self.ground_truth_subscription = (
            self.create_subscription(
                TFMessage,
                self.ground_truth_topic,
                self.ground_truth_callback,
                100
            )
        )

        self.get_logger().info(
            'Trajectory recorder started.'
        )

        self.get_logger().info(
            f'Estimated odometry: {self.odom_topic}'
        )

        self.get_logger().info(
            f'Ground truth: {self.ground_truth_topic}'
        )

        self.get_logger().info(
            f'Saving CSV files to {self.results_dir}'
        )

    @staticmethod
    def stamp_to_seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def quaternion_to_yaw(q):
        sin_yaw = 2.0 * (
            q.w * q.z + q.x * q.y
        )

        cos_yaw = 1.0 - 2.0 * (
            q.y * q.y + q.z * q.z
        )

        return math.atan2(sin_yaw, cos_yaw)

    def write_pose(self, writer, stamp, position, orientation):
        timestamp = self.stamp_to_seconds(stamp)

        yaw = self.quaternion_to_yaw(orientation)

        writer.writerow([
            f'{timestamp:.9f}',
            f'{position.x:.6f}',
            f'{position.y:.6f}',
            f'{position.z:.6f}',
            f'{yaw:.9f}',
            f'{math.degrees(yaw):.6f}'
        ])

    def odom_callback(self, msg):
        self.write_pose(
            self.odom_writer,
            msg.header.stamp,
            msg.pose.pose.position,
            msg.pose.pose.orientation
        )

        self.odom_count += 1

        if self.odom_count % 50 == 0:
            self.odom_file.flush()

    def ground_truth_callback(self, msg):
        # Gazebo publishes poses for multiple entities.
        # Select only the vehicle's transform.
        candidates = [
            transform
            for transform in msg.transforms
            if transform.child_frame_id.strip('/').split('/')[-1]
            == 'vehicle'
        ]

        if not candidates:
            return

        # If multiple transforms match, prefer the exact
        # child frame name "vehicle".
        transform = next(
            (
                item for item in candidates
                if item.child_frame_id.strip('/') == 'vehicle'
            ),
            candidates[0]
        )

        self.write_pose(
            self.ground_truth_writer,
            transform.header.stamp,
            transform.transform.translation,
            transform.transform.rotation
        )

        self.ground_truth_count += 1

        if self.ground_truth_count % 50 == 0:
            self.ground_truth_file.flush()

    def close_files(self):
        self.odom_file.flush()
        self.ground_truth_file.flush()

        self.odom_file.close()
        self.ground_truth_file.close()

        self.get_logger().info(
            f'Recorded {self.odom_count} odometry samples.'
        )

        self.get_logger().info(
            f'Recorded {self.ground_truth_count} ground-truth samples.'
        )

        self.get_logger().info(
            f'Files saved in {self.results_dir}'
        )


def main(args=None):
    rclpy.init(args=args)

    recorder = TrajectoryRecorder()

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close_files()
        recorder.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
