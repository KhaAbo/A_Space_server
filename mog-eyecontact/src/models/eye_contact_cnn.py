import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from torchvision import transforms
import cv2
import numpy as np

from src.utils.project_utils import select_device


class Bottleneck(nn.Module):
    """ResNet bottleneck block."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet50 architecture for eye contact classification."""
    
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_ec = nn.Linear(512 * Bottleneck.expansion, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_ec(x)
        return x


class EyeContactEstimator:
    """Wrapper for eye contact estimation using ResNet50."""
    
    def __init__(self, model_path: str, config=None, device: str = None):
        self.device = select_device(device)
        
        # Get settings from config or use defaults
        if config is not None:
            ec_cfg = config.section("eye_contact")
            self.image_size = ec_cfg.get("image_size", 224)
            self.center_crop = ec_cfg.get("center_crop", 224)
            img_mean = ec_cfg.get("img_mean", [0.485, 0.456, 0.406])
            img_std = ec_cfg.get("img_std", [0.229, 0.224, 0.225])
        else:
            self.image_size = 224
            self.center_crop = 224
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]
        
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.center_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
    
    def _load_model(self, model_path: str):
        """Load ResNet50 model with eye contact weights."""
        model = ResNet([3, 4, 6, 3])  # ResNet50
        weight_path = Path(model_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weight_path}")
        
        state_dict = torch.load(weight_path, map_location=self.device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(filtered_dict)
        
        model.load_state_dict(model_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        return self.transform(face_rgb).unsqueeze(0)
    
    def estimate(self, face_image: np.ndarray) -> float:
        """Estimate eye contact score for a face crop."""
        image_tensor = self.preprocess(face_image).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            score = torch.sigmoid(output).item()
        return score

