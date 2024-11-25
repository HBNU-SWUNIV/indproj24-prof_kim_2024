import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import collections

class MaybePotatos:
    def __init__(self, history_length=5):
        ## best = train8(mAP50 : 0.995, mAP50-95 : 0.957)

        # 감자위치 예측 모델
        self.model = YOLO("model/detect_model_best.pt")

        # RealSense 파이프라인 구성
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 스트림 설정 (컬러와 깊이) - 1280x720, 1024x768,30 FPS
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

        # 깊이 프레임을 컬러 프레임에 맞추기 위한 align 객체 생성
        self.align = rs.align(rs.stream.color)

        # 필터 생성
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.spatial.set_option(rs.option.holes_fill, 3)

        # depth_history 및 history_length 초기화
        self.history_length = history_length
        self.depth_history = collections.deque(maxlen=history_length)
        self.depth_scale = None
    
    '''roi 영역 좌표값 표시'''
    def print_xy(self, image, result):
        if not result[0]:
            return
        # 예측된 결과에서 바운딩 박스 데이터 추출
        boxes = result[0].boxes.xyxy  # 바운딩 박스 좌표 (x1, y1, x2, y2)
        print(boxes)
        x1, y1, x2, y2 = map(int, boxes[0])
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        rx1 = x1 * 2
        rx2 = x2 * 2
        ry1 = y1 * 1.125
        ry2 = y2 * 1.125

        # rcx = cx * 2
        # rcy = cy * 1.125

        # 각 좌표에 텍스트 표시
        coord_text_1 = f'({rx1}, {ry1})'
        coord_text_2 = f'({rx2}, {ry1})'
        coord_text_3 = f'({rx1}, {ry2})'
        coord_text_4 = f'({rx2}, {ry2})'
        # coord_text_5 = f'({rcx}, {rcy})'

        # 각 좌표에 좌표값 표시 (좌표 근처에 텍스트 배치)
        cv2.putText(image, coord_text_1, (x1 - 50, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(image, coord_text_2, (x2 - 50, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(image, coord_text_3, (x1 - 50, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(image, coord_text_4, (x2 - 50, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        # cv2.putText(image, coord_text_5, (cx - 50, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    '''roi 영역 그리기 함수'''
    def make_box(self, image, result):
        if not result[0]:
            return
        # 예측된 결과에서 바운딩 박스 데이터 추출
        boxes = result[0].boxes.xyxy  # 바운딩 박스 좌표 (x1, y1, x2, y2)
        confidences = result[0].boxes.conf  # 확률 값
        classes = result[0].boxes.cls  # 클래스 라벨

        # 바운딩 박스 그리기
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 클래스와 확률 텍스트로 표시
            label = f'{int(cls)}: {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    '''roi 영역만큼 자르기'''
    def box_cropped(self, image, result):
        if not result[0]:
            return None, 0, 0, 0, 0
        # 예측된 결과에서 바운딩 박스 데이터 추출
        boxes = result[0].boxes.xyxy  # 바운딩 박스 좌표 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, boxes[0])
        x1 = int(x1*2)
        x2 = int(x2*2)
        y1 = int(y1*1.125)
        y2 = int(y2*1.125)
        cropped_image = image[y1:y2, x1:x2]

        return cropped_image, x1, x2, y1, y2

    def start(self):
        # 파이프라인 시작
        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()

        # Depth
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # RGB 자동 노출
        color_sensor = device.query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        # Low Ambient Light 프리셋 설정 for L515
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, float(rs.l500_visual_preset.short_range))


    def run(self):
        # 정렬된 RGB 및 필터링된 깊이 프레임 캡처 메서드
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 유효하지 않은 프레임이 있는 경우 기본값으로 초기화
        if not depth_frame:
            depth_frame = np.zeros((1024, 768), dtype=np.uint16)
        if not color_frame:
            color_frame = np.zeros((1280, 720, 3), dtype=np.uint8)

        # 공간 필터 적용
        depth_frame = self.spatial.process(depth_frame)
    
        # 시간 필터 적용
        depth_frame = self.temporal.process(depth_frame)

        # 이미지를 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 깊이 기록이 부족할 때도 빈 프레임 대신 기본값 제공
        self.depth_history.append(depth_image)
        if len(self.depth_history) < self.history_length:
            stacked_depth = np.dstack([depth_image] * self.history_length)
        else:
            stacked_depth = np.dstack(list(self.depth_history))
        
        median_depth_image = np.median(stacked_depth, axis=2).astype(np.uint16)

        # 예측 및 위치를 파악하기 위한 사이즈 변경
        resized_color_image = cv2.resize(color_image, (640, 640))
        resized_depth_image = cv2.resize(median_depth_image, (640,640))

        # Color 이미지 예측
        results = self.model.predict(resized_color_image, conf=0.77)
        
        # 예측된 결과를 이미지로 변환하여 저장
        annotated_color_image = results[0].plot()  # 결과 이미지를 가져옴 (numpy 배열 형식)

        # Color 이미지에 좌표값 추가
        self.print_xy(annotated_color_image, results)
        color_image_result = cv2.resize(annotated_color_image, (1280, 720))

        # Depth 이미지에 박스 추가
        depth_image_result = cv2.resize(resized_depth_image, (1280, 720))

        box_cropped_image, x1, x2, y1, y2 = self.box_cropped(color_image, results)
        box_cropped_depth, _, _, _, _ = self.box_cropped(median_depth_image, results)
        
        return color_image_result, depth_image_result, box_cropped_image, x1, x2, y1, y2, box_cropped_depth

    def stop(self):
        self.pipeline.stop()