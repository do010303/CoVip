#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASSARM Robot Motion Test Scenarios (ROS 2 Humble / Gazebo Fortress)
-------------------------------------------------------------------
Kịch bản kiểm thử chuyển động cánh tay 4-DOF và ngón kẹp song song:
1. Test cơ cấu ngón kẹp & 2 thanh giằng Pivot Arm
2. Test dải hoạt động từng khớp cánh tay (Joint Limits & Directions)
3. Kịch bản gắp - đặt hoàn chỉnh (Pick and Place Cycle)
4. Quỹ đạo chuyển động sóng đa điểm mượt mà (Smooth Wave Trajectory)
5. Reset về vị trí an toàn (Home Position)
"""

import sys
import time
import argparse
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class AssarmTestScenarios(Node):
    def __init__(self):
        super().__init__('assarm_test_scenarios')

        # Khởi tạo publisher cho arm và gripper controllers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            10
        )

        self.arm_joints = [
            'Revolute_Joint_1',
            'Revolute_Joint_2',
            'Revolute_Joint_3',
            'Revolute_Joint_4'
        ]
        self.gripper_joints = ['Revolute_Joint_Active']

        self.get_logger().info('Assarm Test Scenarios Node sẵn sàng.')

    def send_arm_trajectory(self, waypoints, durations):
        """
        Gửi chuỗi điểm quỹ đạo tới arm_controller
        waypoints: danh sách các list [j1, j2, j3, j4] (rad)
        durations: danh sách thời gian đạt tới từng điểm (giây)
        """
        msg = JointTrajectory()
        msg.joint_names = self.arm_joints

        total_time = 0.0
        for pt, d in zip(waypoints, durations):
            total_time += d
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in pt]
            point.time_from_start = Duration(
                sec=int(total_time),
                nanosec=int((total_time - int(total_time)) * 1e9)
            )
            msg.points.append(point)

        self.arm_pub.publish(msg)

    def send_gripper_trajectory(self, positions, durations):
        """
        Gửi chuỗi điểm đóng/mở tới gripper_controller
        positions: danh sách các góc của Revolute_Joint_Active (rad)
        durations: danh sách thời gian đạt tới từng vị trí (giây)
        """
        msg = JointTrajectory()
        msg.joint_names = self.gripper_joints

        total_time = 0.0
        for pos, d in zip(positions, durations):
            total_time += d
            point = JointTrajectoryPoint()
            point.positions = [float(pos)]
            point.time_from_start = Duration(
                sec=int(total_time),
                nanosec=int((total_time - int(total_time)) * 1e9)
            )
            msg.points.append(point)

        self.gripper_pub.publish(msg)

    # ==========================================
    # CÁC KỊCH BẢN TEST CỤ THỂ
    # ==========================================

    def scenario_home(self):
        """Kịch bản 0: Đưa robot về vị trí Home (vị trí an toàn ban đầu)"""
        self.get_logger().info('>>> [KỊCH BẢN] Đưa cánh tay và kẹp về vị trí HOME...')
        self.send_arm_trajectory([[0.0, 0.0, 0.0, 0.0]], [2.0])
        self.send_gripper_trajectory([0.0], [1.5])
        time.sleep(2.5)
        self.get_logger().info('--> Đã hoàn thành về HOME.')

    def scenario_gripper_test(self, cycles=3):
        """
        Kịch bản 1: Kiểm tra kẹp gắp và 2 thanh giằng Pivot Arm
        Đóng và mở kẹp nhiều lần để quan sát:
        - Hai má kẹp (Gripper_Left/Right) chuyển động tịnh tiến song song
        - Hai thanh Pivot_Arm_Left/Right quay đồng bộ, giữ vững kết cấu
        """
        self.get_logger().info(f'>>> [KỊCH BẢN 1] Bắt đầu kiểm tra Gripper & Pivot Arms ({cycles} chu kỳ)...')
        self.get_logger().info('    Quan sát Gazebo: 2 thanh Pivot Arm trái/phải phải khép mở song song với má kẹp!')

        # Vị trí mở (1.2 rad) và vị trí đóng (0.0 rad)
        for i in range(1, cycles + 1):
            self.get_logger().info(f'  [Chu kỳ {i}/{cycles}] Mở kẹp (Góc: 1.2 rad)...')
            self.send_gripper_trajectory([1.2], [1.5])
            time.sleep(2.0)

            self.get_logger().info(f'  [Chu kỳ {i}/{cycles}] Đóng kẹp (Góc: 0.0 rad)...')
            self.send_gripper_trajectory([0.0], [1.5])
            time.sleep(2.0)

        self.get_logger().info('--> [KỊCH BẢN 1] Hoàn thành kiểm tra Gripper & Pivot Arms.')

    def scenario_single_joint_sweep(self):
        """
        Kịch bản 2: Kiểm tra từng khớp cánh tay (Single Joint Sweep)
        Lần lượt di chuyển Joint 1 -> 2 -> 3 -> 4 đến các góc biên an toàn (+0.8 rad và -0.8 rad)
        để kiểm tra chiều quay, encoder/feedback và giới hạn góc.
        """
        self.get_logger().info('>>> [KỊCH BẢN 2] Bắt đầu quét kiểm tra từng khớp (Joint Sweep)...')
        self.scenario_home()

        joint_names = ['Khớp 1 (Đế - Yaw)', 'Khớp 2 (Vai - Shoulder)', 'Khớp 3 (Khuỷu - Elbow)', 'Khớp 4 (Cổ tay - Wrist)']
        test_angles = [0.7, -0.7, 0.0]

        for idx, name in enumerate(joint_names):
            self.get_logger().info(f'  --- Đang test {name} ---')
            for target_angle in test_angles:
                pose = [0.0, 0.0, 0.0, 0.0]
                pose[idx] = target_angle
                self.get_logger().info(f'      Di chuyển đến {target_angle:.1f} rad...')
                self.send_arm_trajectory([pose], [1.8])
                time.sleep(2.2)

        self.get_logger().info('--> [KỊCH BẢN 2] Hoàn thành kiểm tra từng khớp.')

    def scenario_pick_and_place(self):
        """
        Kịch bản 3: Quy trình Gắp và Đặt mẫu hoàn chỉnh (Pick and Place)
        1. Về Home
        2. Mở kẹp
        3. Cúi cánh tay xuống vị trí gắp (Pick pose)
        4. Đóng kẹp giữ vật
        5. Nhấc bổng cánh tay lên (Lift pose)
        6. Quay đế 45 độ sang vị trí đặt (Transfer pose)
        7. Hạ cánh tay xuống (Place pose)
        8. Mở kẹp nhả vật
        9. Rút tay lên và quay về Home
        """
        self.get_logger().info('>>> [KỊCH BẢN 3] Bắt đầu mô phỏng quy trình GẮP VÀ ĐẶT (Pick & Place)...')

        steps = [
            ("Bước 1: Về vị trí chuẩn bị (Home)", [0.0, 0.0, 0.0, 0.0], 0.0, 2.0),
            ("Bước 2: Mở ngón kẹp sẵn sàng", [0.0, 0.0, 0.0, 0.0], 1.1, 1.2),
            ("Bước 3: Vươn tay xuống vị trí gắp vật", [0.0, 0.6, -0.7, 0.2], 1.1, 2.5),
            ("Bước 4: Kẹp chặt vật thể", [0.0, 0.6, -0.7, 0.2], 0.1, 1.5),
            ("Bước 5: Nâng vật thể lên cao an toàn", [0.0, 0.2, -0.4, 0.1], 0.1, 2.0),
            ("Bước 6: Xoay đế 45° sang trạm đích", [0.8, 0.2, -0.4, 0.1], 0.1, 2.5),
            ("Bước 7: Hạ cánh tay xuống bàn đặt", [0.8, 0.6, -0.7, 0.2], 0.1, 2.0),
            ("Bước 8: Mở ngón kẹp nhả vật thể", [0.8, 0.6, -0.7, 0.2], 1.1, 1.5),
            ("Bước 9: Rút cánh tay lên", [0.8, 0.1, -0.2, 0.0], 1.1, 1.8),
            ("Bước 10: Quay về vị trí Home và đóng kẹp", [0.0, 0.0, 0.0, 0.0], 0.0, 2.5),
        ]

        for desc, arm_pose, gripper_pos, wait_time in steps:
            self.get_logger().info(f'  {desc}')
            self.send_arm_trajectory([arm_pose], [wait_time * 0.85])
            self.send_gripper_trajectory([gripper_pos], [wait_time * 0.85])
            time.sleep(wait_time)

        self.get_logger().info('--> [KỊCH BẢN 3] Hoàn thành quy trình Gắp và Đặt.')

    def scenario_smooth_wave(self):
        """
        Kịch bản 4: Quỹ đạo đa điểm mượt mà (Continuous Wave Trajectory)
        Gửi toàn bộ chuỗi điểm (multi-waypoint trajectory) trong 1 message
        để controller nội suy spline mượt mà, kiểm tra độ ổn định động học.
        """
        self.get_logger().info('>>> [KỊCH BẢN 4] Bắt đầu kiểm tra quỹ đạo mượt mà đa điểm (Continuous Wave)...')

        waypoints = [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.3, -0.4, 0.2],
            [0.0, 0.5, -0.7, 0.4],
            [-0.5, 0.3, -0.4, -0.2],
            [-0.5, -0.2, 0.3, -0.3],
            [0.0, -0.3, 0.4, 0.0],
            [0.5, -0.2, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0]
        ]
        durations = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        self.send_arm_trajectory(waypoints, durations)
        # Đồng thời cho kẹp gắp nhấp nháy mở đóng theo nhịp
        gripper_positions = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        self.send_gripper_trajectory(gripper_positions, durations)

        total_duration = sum(durations)
        self.get_logger().info(f'  Quỹ đạo đang thực thi mượt mà trong {total_duration} giây...')
        time.sleep(total_duration + 0.5)

        self.get_logger().info('--> [KỊCH BẢN 4] Hoàn thành kiểm tra quỹ đạo mượt mà.')


def print_menu():
    print("""
============================================================
      BỘ KỊCH BẢN KIỂM THỬ CÁNH TAY ASSARM (ROS 2 / GAZEBO)
============================================================
 [1] Test Ngón Kẹp & 2 Thanh Giằng Pivot Arm (Gripper & Linkage)
 [2] Test Dải Hoạt Động Từng Khớp Cánh Tay (Single Joint Sweep)
 [3] Chu Trình Gắp - Đặt Mẫu (Pick and Place Cycle)
 [4] Quỹ Đạo Vẫy Đa Điểm Mượt Mà (Smooth Multi-point Wave)
 [5] Đưa Robot Về Vị Trí An Toàn (Reset Home)
 [6] Chạy Toàn Bộ Kịch Bản Từ Đầu Đến Cuối (Run All)
 [0] Thoát
============================================================
""")


def main(args=None):
    rclpy.init(args=args)
    tester = AssarmTestScenarios()

    parser = argparse.ArgumentParser(description='Assarm Test Scenarios')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Chạy kịch bản: 1, 2, 3, 4, 5, all')
    cli_args, _ = parser.parse_known_args()

    # Đợi 1 chút để publisher kết nối với subscribers trên Gazebo
    time.sleep(1.0)

    try:
        if cli_args.scenario is not None:
            choice = str(cli_args.scenario).strip().lower()
            if choice == '1':
                tester.scenario_gripper_test()
            elif choice == '2':
                tester.scenario_single_joint_sweep()
            elif choice == '3':
                tester.scenario_pick_and_place()
            elif choice == '4':
                tester.scenario_smooth_wave()
            elif choice in ['5', 'home']:
                tester.scenario_home()
            elif choice in ['6', 'all']:
                tester.scenario_gripper_test(cycles=2)
                tester.scenario_single_joint_sweep()
                tester.scenario_pick_and_place()
                tester.scenario_smooth_wave()
                tester.scenario_home()
            else:
                print(f'Tùy chọn không hợp lệ: {choice}')
        else:
            while rclpy.ok():
                print_menu()
                choice = input('Nhập lựa chọn của bạn (0-6): ').strip()
                if choice == '1':
                    tester.scenario_gripper_test()
                elif choice == '2':
                    tester.scenario_single_joint_sweep()
                elif choice == '3':
                    tester.scenario_pick_and_place()
                elif choice == '4':
                    tester.scenario_smooth_wave()
                elif choice == '5':
                    tester.scenario_home()
                elif choice == '6':
                    tester.scenario_gripper_test(cycles=2)
                    tester.scenario_single_joint_sweep()
                    tester.scenario_pick_and_place()
                    tester.scenario_smooth_wave()
                    tester.scenario_home()
                elif choice == '0':
                    print('Đã thoát chương trình kiểm thử.')
                    break
                else:
                    print('Lựa chọn không hợp lệ, vui lòng nhập từ 0 đến 6.')
    except KeyboardInterrupt:
        print('\nNgười dùng đã dừng chương trình.')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
