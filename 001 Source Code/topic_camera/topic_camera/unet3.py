import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# 기본 컨볼루션 블록 정의
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
# 인코더 블록 정의
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x), self.pool(self.conv(x))
    
# 디코더 블록 정의
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)
    
# U-Net 모델 정의
class UNET(nn.Module):
    def __init__(self, n_classes):
        super(UNET, self).__init__()
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.b = ConvBlock(512, 1024)
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1)

        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 모델 학습 시 사용한 크기로 조정
                transforms.ToTensor(),          # 텐서로 변환
            ])

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        return self.outputs(d4)
    
    # 이미지 로드 및 전처리 함수
    def load_image(self, image_path):
        # image = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 로드
        image = self.transform(image_path)  # 전처리 적용 (크기 조정 및 텐서 변환)
        image = image.unsqueeze(0)  # 배치 차원 추가 (모델에 입력할 때는 배치가 필요)
        return image

    # 예측 함수
    def predict(self, model, image_path, device):
        # 모델 평가 모드로 전환 (학습 시 사용한 dropout, batchnorm 등을 비활성화)
        model.eval()

        # 이미지를 전처리하고 텐서로 변환
        image = self.load_image(image_path).to(device)

        # 순전파 (예측 수행)
        with torch.no_grad():  # 기울기 계산 비활성화 (평가 시 필요 없음)
            output = model(image)
        
        # 출력 결과는 [batch_size, num_classes, height, width] 형태
        # 일반적으로 num_classes가 1이면 Sigmoid, 2 이상이면 Softmax로 해석
        output = torch.sigmoid(output)  # 이진 분류일 때 사용 (픽셀마다 0~1 확률)
        output = output.squeeze(0).cpu().numpy()  # 배치 차원 제거 및 CPU로 이동
        
        # 예측 결과를 바이너리 마스크로 변환 (0.5를 기준으로 클래스 결정)
        predicted_mask = (output > 0.5).astype(np.uint8)
        return predicted_mask
    