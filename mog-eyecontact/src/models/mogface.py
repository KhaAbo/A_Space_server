import sys
from pathlib import Path
import math
import torch
import cv2
import numpy as np
from typing import List
from torch.autograd import Variable
from src.utils.project_utils import select_device

from src.mogface.modelling import basenet, resnet, fpn, pred_net

def normalize_anchor(anchors):
    '''
    from  [c_x, cy, w, h] to [x0, x1, y0, y1] 
    '''
    return np.concatenate((anchors[:, :2] - (anchors[:, 2:] - 1) / 2,
                            anchors[:, :2] + (anchors[:, 2:] - 1) / 2), axis=1) 

def transform_anchor(anchors):
    '''
    from [x0, x1, y0, y1] to [c_x, cy, w, h]
    x1 = x0 + w - 1
    c_x = (x0 + x1) / 2 = (2x0 + w - 1) / 2 = x0 + (w - 1) / 2
    '''
    return np.concatenate(((anchors[:, :2] + anchors[:, 2:]) / 2 , anchors[:, 2:] - anchors[:, :2] + 1), axis=1)

def decode(loc, anchors):
    """
    loc: torch.Tensor
    anchors: 2-d, torch.Tensor (cx, cy, w, h)
    boxes: 2-d, torch.Tensor (x0, y0, x1, y1)
    """

    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:])), 1)

    boxes[:, 0] -= (boxes[:,2] - 1 ) / 2
    boxes[:, 1] -= (boxes[:,3] - 1 ) / 2
    boxes[:, 2] += boxes[:,0] - 1  
    boxes[:, 3] += boxes[:,1] - 1 

    return boxes

def nms(dets, thresh):
    """
    Pure Python NMS baseline.
    dets: [[x1, y1, x2, y2, score], ...]
    thresh: iou threshold
    """
    if dets.shape[0] == 0:
        return []
        
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class GeneratePriorBoxes(object):
    '''
    both for fpn and single layer, single layer need to test
    return (np.array) [num_anchros, 4] [x0, y0, x1, y1]
    '''
    def __init__(self, scale_list=[1.], \
                 aspect_ratio_list=[1.0], \
                 stride_list=[4,8,16,32,64,128], \
                 anchor_size_list=[16,32,64,128,256,512]):
        self.scale_list = scale_list
        self.aspect_ratio_list = aspect_ratio_list
        self.stride_list = stride_list
        self.anchor_size_list = anchor_size_list

    def __call__(self, img_height, img_width):
        final_anchor_list = []

        for idx, stride in enumerate(self.stride_list):
            anchor_list = []
            cur_img_height = img_height
            cur_img_width = img_width
            tmp_stride = stride 

            while tmp_stride != 1:
                tmp_stride = tmp_stride // 2
                cur_img_height = (cur_img_height + 1) // 2
                cur_img_width = (cur_img_width + 1) // 2

            for i in range(cur_img_height):
                for j in range(cur_img_width):
                    for scale in self.scale_list:
                        cx = (j + 0.5) * stride
                        cy = (i + 0.5) * stride
                        side_x = self.anchor_size_list[idx] * scale
                        side_y = self.anchor_size_list[idx] * scale
                        for ratio in self.aspect_ratio_list:
                            anchor_list.append([cx, cy, side_x / math.sqrt(ratio), side_y * math.sqrt(ratio)])

            final_anchor_list.append(anchor_list)
        final_anchor_arr = np.concatenate(final_anchor_list, axis=0)
        normalized_anchor_arr = normalize_anchor(final_anchor_arr).astype('float32')

        return normalized_anchor_arr

class MogFaceDetector:
    """Clean wrapper for MogFace face detection."""
    
    def __init__(
        self,
        model_path: str,
        config = None,
        device: str = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        max_bbox_per_img: int = 750
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_bbox_per_img = max_bbox_per_img
        
        self.device = select_device(device)
        self.model = self._load_model(model_path)
        
        self.anchor_generator = GeneratePriorBoxes(
            scale_list=[0.68],
            aspect_ratio_list=[1.0],
            stride_list=[4, 8, 16, 32, 64, 128],
            anchor_size_list=[16, 32, 64, 128, 256, 512]
        )
        
        self.normalize_pixel = True
        self.use_rgb = True
        
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        
        if self.use_rgb:
            self.img_mean = (np.array(img_mean, dtype=np.float32) * 255)[::-1]
            self.img_std = (np.array(img_std, dtype=np.float32) * 255)[::-1]
        else:
            self.img_mean = np.array(img_mean, dtype=np.float32)
            self.img_std = np.array(img_std, dtype=np.float32)
    
    def _load_model(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        # Instantiate components directly
        backbone = resnet.ResNet(depth=152)
        
        fpn_net = fpn.LFPN(
            c2_out_ch=256,
            c3_out_ch=512,
            c4_out_ch=1024,
            c5_out_ch=2048,
            c6_mid_ch=512,
            c6_out_ch=512,
            c7_mid_ch=128,
            c7_out_ch=256,
            out_dsfd_ft=True
        )
        
        pred_net_model = pred_net.MogPredNet(
            num_classes=1,
            num_anchor_per_pixel=1,
            input_ch_list=[256, 256, 256, 256, 256, 256],
            use_deep_head=True,
            deep_head_with_gn=True,
            deep_head_ch=256,
            use_ssh=False,
            use_cpm=True,
            use_dsfd=True,
            retina_cls_weight_init_fn="RetinaClsWeightInit"
        )
        
        net = basenet.WiderFaceBaseNet(
            backbone=backbone,
            fpn=fpn_net,
            pred_net=pred_net_model
        )
        
        state_dict = torch.load(str(model_path), map_location=self.device)
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        return net
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        img -= self.img_mean
        img /= self.img_std
        if self.normalize_pixel:
            img /= 255.0
        if self.use_rgb:
            img = img[:, :, ::-1].copy()
        return img
    
    def detect(self, image: np.ndarray, shrink: float = None) -> List[np.ndarray]:
        """Detect faces in image. Returns detections as [x1, y1, x2, y2, conf]."""
        if shrink is None:
            max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5
            max_im_shrink = 2.2 if max_im_shrink > 2.2 else max_im_shrink
            shrink = max_im_shrink if max_im_shrink < 1 else 1
        
        if shrink != 1:
            x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        else:
            x = image
        
        height, width = x.shape[0], x.shape[1]
        x = self._preprocess(x)
        
        x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        x_tensor = Variable(x_tensor.to(self.device))
        
        with torch.no_grad():
            out = self.model(x_tensor)
        
        anchors = transform_anchor(self.anchor_generator(height, width))
        anchors = torch.FloatTensor(anchors).to(self.device)
        
        decode_bbox = decode(out[1].squeeze(0), anchors)
        boxes = decode_bbox
        scores = out[0].squeeze(0)
        
        top_k = 5000
        v, idx = scores[:, 0].sort(0)
        idx = idx[-top_k:]
        boxes = boxes[idx]
        scores = scores[idx]
        
        boxes = boxes.cpu().numpy()
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        boxes[:, 0] /= shrink
        boxes[:, 1] /= shrink
        boxes[:, 2] = boxes[:, 0] + w / shrink - 1
        boxes[:, 3] = boxes[:, 1] + h / shrink - 1
        
        scores = scores.cpu().numpy()
        
        inds = np.where(scores[:, 0] > self.score_threshold)[0]
        if len(inds) == 0:
            return np.empty([0, 5], dtype=np.float32)
        
        c_bboxes = boxes[inds]
        c_scores = scores[inds, 0]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        
        keep = nms(c_dets, self.nms_threshold)
        c_dets = c_dets[keep, :]
        
        if self.max_bbox_per_img > 0:
            image_scores = c_dets[:, -1]
            if len(image_scores) > self.max_bbox_per_img:
                image_thresh = np.sort(image_scores)[-self.max_bbox_per_img]
                keep = np.where(c_dets[:, -1] >= image_thresh)[0]
                c_dets = c_dets[keep, :]
        
        return c_dets


