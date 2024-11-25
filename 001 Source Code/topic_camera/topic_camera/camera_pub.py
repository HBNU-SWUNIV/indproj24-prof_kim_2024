import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool
from std_srvs.srv import Trigger
import cv2 as cv
import numpy as np
import threading
import time

class CameraPublisher(Node):
    def __init__(self, cap):
        super().__init__('camera_publisher')
        self.cap = cap

        # Publisher 생성
        self.coordinate_publisher = self.create_publisher(Float64MultiArray, 'selected_coordinates', 10)
        # Subscriber 생성 (동작 완료 확인)
        self.completion_subscriber = self.create_subscription(Bool, 'task_completed', self.completion_callback, 10)
        
        self.waiting_for_completion = False

        # 카메라 초기화 확인
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open video stream")
            return

        # 마우스 콜백 함수 설정
        cv.namedWindow("Video Frame")
        cv.setMouseCallback("Video Frame", self.mouse_callback)

        # 타이머 생성 (비디오 처리)
        self.timer = self.create_timer(0.1, self.process_video)
        self.selected_point = None
        self.hsv_ranges = None
        self.previous_coordinates = []

        # 트랙바 설정
        cv.namedWindow('img_result')
        cv.createTrackbar('threshold', 'img_result', 0, 255, lambda x: None)
        cv.setTrackbarPos('threshold', 'img_result', 30)

    def process_video(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Empty frame captured")
            return
        
        # 화면 중앙 좌표 계산
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # 선택한 지점에 대한 좌표 표시
        if self.selected_point is not None:
            x, y = self.selected_point
            cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # HSV 범위 마스크 처리
        if self.hsv_ranges is not None:
            img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            img_mask1 = cv.inRange(img_hsv, self.hsv_ranges['lower_blue1'], self.hsv_ranges['upper_blue1'])
            img_mask2 = cv.inRange(img_hsv, self.hsv_ranges['lower_blue2'], self.hsv_ranges['upper_blue2'])
            img_mask3 = cv.inRange(img_hsv, self.hsv_ranges['lower_blue3'], self.hsv_ranges['upper_blue3'])
            img_mask = img_mask1 | img_mask2 | img_mask3

            kernel = np.ones((11, 11), np.uint8)
            img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
            img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img_mask)
            coordinates = []

            for idx, centroid in enumerate(centroids):
                if stats[idx][0] == 0 and stats[idx][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue

                x, y = int(centroid[0]), int(centroid[1])
                area = stats[idx, cv.CC_STAT_AREA]

                if area > 50:  # 임계값보다 큰 객체만 고려
                    coordinates.append((x, y))
                    cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 좌표가 이전과 +-2 범위 내에서 동일한 경우 publish하지 않음
            if not self.are_coordinates_similar(coordinates, self.previous_coordinates):
                self.coordinates_to_send = coordinates
                self.previous_coordinates = coordinates
                self.get_logger().info(f'Total detected points: {coordinates}')
                self.get_logger().info(f'Total detected points: {len(coordinates)}')
                self.current_index = 0

            # 작업 완료 대기 중이 아니고 보낼 좌표가 있으면 좌표 전송
            if not self.waiting_for_completion and self.coordinates_to_send and self.current_index <= len(self.coordinates_to_send):
                coordinate_pair = self.coordinates_to_send[self.current_index]
                x, y = coordinate_pair
                index = self.current_index + 1
                # 화면 중앙으로부터의 거리 계산 후 전송
                x_transformed = (x - center_x) / 1000.0
                y_transformed = (y - center_y) / 1000.0
                self.get_logger().info(f'Sending Point {index}/{len(self.previous_coordinates)}: ({x_transformed}, {y_transformed})')
                msg = Float64MultiArray()
                msg.data = [float(x_transformed), float(y_transformed)]
                self.coordinate_publisher.publish(msg)
                self.waiting_for_completion = True
                self.current_index += 1

            # 모든 좌표를 보낸 후 대기
            if self.current_index >= len(self.coordinates_to_send):
                self.get_logger().info('All coordinates sent, waiting for completion')

            img_result = cv.bitwise_and(frame, frame, mask=img_mask)
            cv.imshow('img_result', img_result)
            
        frame_resized = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        cv.imshow("Video Frame", frame_resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def are_coordinates_similar(self, current, previous):
        if len(current) != len(previous):
            return False
        for (x1, y1), (x2, y2) in zip(current, previous):
            if abs(x1 - x2) > 2 or abs(y1 - y2) > 2:
                return False
        return True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)
            frame = self.cap.read()[1]
            if frame is not None:
                color = frame[y, x]
                one_pixel = np.uint8([[color]])
                hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)[0][0]
                threshold = cv.getTrackbarPos('threshold', 'img_result')

                if hsv[0] < 10:
                    lower_blue1 = np.array([hsv[0] - 10 + 180, threshold, threshold])
                    upper_blue1 = np.array([180, 255, 255])
                    lower_blue2 = np.array([0, threshold, threshold])
                    upper_blue2 = np.array([hsv[0], 255, 255])
                    lower_blue3 = np.array([hsv[0], threshold, threshold])
                    upper_blue3 = np.array([hsv[0] + 10, 255, 255])
                elif hsv[0] > 170:
                    lower_blue1 = np.array([hsv[0], threshold, threshold])
                    upper_blue1 = np.array([180, 255, 255])
                    lower_blue2 = np.array([0, threshold, threshold])
                    upper_blue2 = np.array([hsv[0] + 10 - 180, 255, 255])
                    lower_blue3 = np.array([hsv[0] - 10, threshold, threshold])
                    upper_blue3 = np.array([hsv[0], 255, 255])
                else:
                    lower_blue1 = np.array([hsv[0], threshold, threshold])
                    upper_blue1 = np.array([hsv[0] + 10, 255, 255])
                    lower_blue2 = np.array([hsv[0] - 10, threshold, threshold])
                    upper_blue2 = np.array([hsv[0], 255, 255])
                    lower_blue3 = np.array([hsv[0] - 10, threshold, threshold])
                    upper_blue3 = np.array([hsv[0], 255, 255])

                self.hsv_ranges = {
                    'lower_blue1': lower_blue1,
                    'upper_blue1': upper_blue1,
                    'lower_blue2': lower_blue2,
                    'upper_blue2': upper_blue2,
                    'lower_blue3': lower_blue3,
                    'upper_blue3': upper_blue3
                }

    def completion_callback(self, msg):
        if msg.data:
            self.get_logger().info('Task completed, ready to send next coordinates')
            self.waiting_for_completion = False

    def destroy_node(self):
        super().destroy_node()
        if self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    cap = cv.VideoCapture('rtmp://192.168.0.163:1935/live/stream')  # 0번 카메라 사용
    node = CameraPublisher(cap)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
