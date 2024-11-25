from maybePotatos_detect import MaybePotatos
from unet3 import UNET
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image

'''edge 추출 함수'''
def sobel_edges(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    edges = cv2.addWeighted(sobelx, 1, sobely, 1, 0)

    return edges

'''mask 에서 좌표 추출 함수(컨투어)'''
def get_contours_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

'''외곽선 좌표의 중심값 계산 함수'''
def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # 중앙값 x 좌표
        cy = int(M["m01"] / M["m00"])  # 중앙값 y 좌표
    else:
        cx, cy = 0, 0  # 면적이 0인 경우 (예외 처리)
    return (cx, cy)

def plot_depth_in_area(cap, UN):
    device12 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """시각화 그래프 메서드"""
    # fig, ax = plt.subplots()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))


    vmin = 0.25  
    vmax = 0.35 
    # cbar = fig.colorbar(ax4, ax=ax4)

    def update(frame):
        '''
        median_color_image = 전체 color 이미지
        median_depth_image = 전체 depth 이미지
        box_image = color 이미지의 roi영역
        box_image = depth 이미지의 roi영역
        x1, x2 = 박스 꼭짓점의 x좌표
        y1, y2 = 박스 꼭짓점의 y좌표
        '''
        median_color_image, median_depth_image, box_image, x1, x2, y1, y2, box_depth = cap.run()

        ax1.clear() # 전체 color 이미지
        ax2.clear() # box color 이미지 - 각 구멍의 중심좌표
        ax3.clear() # box color 이미지 - 각 중심좌표의 depth 값
        ax4.clear() # box depth 이미지 - 각 중심좌표의 depth 값
        if box_image is not None and box_image.size > 0:
            # 구멍 위치 예측
            predicted_mask = UN.predict(UN, Image.fromarray(sobel_edges(box_image)), device12)

            predicted_mask = cv2.resize(predicted_mask.squeeze(), (x2-x1, y2-y1))
            box_image = cv2.resize(box_image, (x2-x1,y2-y1))
            box_depth = cv2.resize(box_depth, (x2-x1,y2-y1))
            # 마스크에서 외곽선 좌표 얻기
            contours = get_contours_from_mask(predicted_mask)
            
            # 예측된 마스크 및 외곽선 시각화
            # contours = 구멍 모음 리스트
            for i in range(len(contours)):
                contour = contours[i]
                # 중심좌표 텍스트로 표시
                centroid = get_centroid(contour)  # 중심좌표 계산
                box_cx, box_cy = centroid

                cropped_cx = box_cx + x1 #- 75   # 잘린 이미지 크기에 맞춘 좌표
                cx = box_cx + x1                # 실제 X 좌표
                cy = box_cy + y1                # 실제 Y 좌표
                ax1.text(cropped_cx, cy, f'({cx},{cy})', color='blue', fontsize=6)  # 중심값 표시
                ax1.scatter(cropped_cx, cy, color='blue', s=20)  # 중심점 표시, s=점 크기

                # 중심좌표값 표시
                ax2.text(box_cx, box_cy, f'({cx},{cy})', color='blue', fontsize=12)  # 중심값 표시
                ax2.scatter(box_cx, box_cy, color='blue', s=20)  # 중심점 표시, s=점 크기
                ax2.plot(contour[:, :, 0], contour[:, :, 1], 'g')  # 좌표 그리기

                depth_value = median_depth_image[cy, cx] # 중심좌표의 depth값 추출
                # depth 값 표시
                ax3.text(box_cx, box_cy, f'({i}: {depth_value})', color='blue', fontsize=12)  # 중심값 표시
                ax3.scatter(box_cx, box_cy, color='blue', s=20)  # 중심점 표시, s=점 크기

                ax4.text(box_cx, box_cy, f'({i}: {depth_value})', color='blue', fontsize=12)  # 중심값 표시
                ax4.scatter(box_cx, box_cy, color='blue', s=20)  # 중심점 표시, s=점 크기

                '''
                cx = 현재 구멍의 중심 x좌표
                cy = 현재 구멍의 중심 y좌표
                depth_value = 현재 중앙좌표의 depth값
                '''

            ax1.imshow(median_color_image)
            ax2.imshow(box_image)
            ax3.imshow(box_image)
            ax4.imshow(box_depth * cap.depth_scale, cmap='jet', vmin=vmin, vmax=vmax)
            
        else:
            ax1.imshow(median_color_image)
            empty_image = np.zeros((128, 128), dtype=np.uint8)  # 크기 128x128의 빈 이미지
            ax2.imshow(empty_image)
            ax3.imshow(median_depth_image)
            ax4.imshow(empty_image)

        

        # 그래프 업데이트
        ax1.set_title(f"image")
        ax1.set_xlabel("X axis")
        ax1.set_ylabel("Y axis")

        # 그래프 업데이트
        ax2.set_title(f"Color image in roi region")
        ax2.set_xlabel("X axis")
        ax2.set_ylabel("Y axis")

        # 그래프 업데이트
        ax3.set_title(f"Depth value in roi region")
        ax3.set_xlabel("X axis")
        ax3.set_ylabel("Y axis")

    ani = FuncAnimation(fig, update, interval=1)
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 감자위치 예측 클래스
    MP = MaybePotatos()

    # 구멍위치 예측 모델
    UN = UNET(n_classes = 1).to(device)
    UN.load_state_dict(torch.load('model/unet3_best_model(edges).pth', weights_only=True))
    try:
        MP.start()
        plot_depth_in_area(MP, UN)
    finally:
        MP.stop()

# U-Net 모델 정의 (기존과 동일한 구조여야 함)
