#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class DriveTrajectory(Node):

	def __init__(self):
		super().__init__('drive_trajectory')

		# Robot velocity command topic
		self.cmd_vel_topic = '/model/vehicle/cmd_vel'

		self.publisher = self.create_publisher(
			Twist,
			self.cmd_vel_topic,
			10
		)

		# Trajectory parameters
		self.linear_velocity = 0.2	# m/s
		self.angular_velocity = 0.5	# rad/s
		self.straight_distance = 2.0	# metres
		self.turn_angle = 1.57079632679	# 90 degrees in radians

		# Calculate approximate movement durations
		self.straight_duration = (
			self.straight_distance / self.linear_velocity
		)

		self.turn_duration = (
			self.turn_angle / self.angular_velocity
		)

		self.get_logger().info(
			'Trajectory controller initialised.'
		)

	def publish_velocity(self, linear_x, angular_z):
		"""Publish a velocity command."""

		msg = Twist()
		msg.linear.x = linear_x
		msg.angular.z = angular_z

		self.publisher.publish(msg)

	def stop_robot(self):
		"""Send zero velocity commands to stop the robot."""

		self.get_logger().info('Stopping robot.')

		for _ in range(5):
			self.publish_velocity(0.0, 0.0)
			rclpy.spin_once(self, timeout_sec=0.05)
			time.sleep(0.05)

	def run_trajectory(self):
		"""Drive forwards, turn left, and stop."""

		try:
			# Allow time for ROS discovery and connections
			self.get_logger().info(
				f'Waiting for a subscriber on {self.cmd_vel_topic}...'
			)

			timeout = time.monotonic() + 10.0

			while(
				self.publisher.get_subscription_count() == 0
				and time.monotonic() < timeout
			):
				rclpy.spin_once(self, timeout_sec=0.1)

			if self.publisher.get_subscription_count() == 0:
				self.get_logger().error(
					'No subscriber found on the command topic. '
					'Check that the Gazebo bridge is running and '
					'the robot has a working drive controller.'
				)
				return

			# Phase 1: Drive forwards
			self.get_logger().info(
				f'Driving forwards for {self.straight_duration:.2f} seconds.'
			)

			start_time = time.monotonic()

			while (
				time.monotonic() - start_time
				< self.straight_duration
			):
				self.publish_velocity(
					self.linear_velocity,
					0.0
				)
				rclpy.spin_once(self, timeout_sec=0.01)
				time.sleep(0.05)

			self.stop_robot()

			# Short pause between movements
			time.sleep(1.0)

			# Phase 2: Turn left
			self.get_logger().info(
				f'Turning left for {self.turn_duration:.2f} seconds.'
			)

			start_time = time.monotonic()

			while (
				time.monotonic() - start_time
				< self.turn_duration
			):
				self.publish_velocity(
					0.0,
					self.angular_velocity
				)
				rclpy.spin_once(self, timeout_sec=0.01)
				time.sleep(0.05)

			# Phase 3: Stop
			self.stop_robot()

			self.get_logger().info(
				'Trajectory complete: forward movement and left turn.'
			)

		except KeyboardInterrupt:
			self.get_logger().warning(
				'Trajectory interrupted by user.'
			)

		finally:
			self.stop_robot()

def main(args=None):
	rclpy.init(args=args)

	controller = DriveTrajectory()

	try:
		controller.run_trajectory()
	finally:
		controller.stop_robot()
		controller.destroy_node()

		if rclpy.ok():
			rclpy.shutdown()

if __name__ == '__main__':
	main()
