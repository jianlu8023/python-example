import numpy as np
import torch
from torchvision import transforms

from PIL import Image


class ModelProcessor:
    def __init__(self):
        # ResNet 标准预处理
        self.resnet_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def crop_rois(self, original_img, prev_results):
        """
        根据上一级的检测结果，从原图中裁剪出多个 ROI
        :return: List[Dict] -> {'image': PIL, 'offset': (x,y), 'parent_info': {}}
        """
        inputs = []
        # prev_results 格式: List[List[Dict]] (对应多张输入图的结果)
        # 这里假设我们处理的是单流输入
        for det_list in prev_results:
            for det in det_list:
                bbox = det.get('bbox')
                if not bbox:
                    continue

                # 裁剪 ROI
                roi = original_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                # --- 关键：将 det 里的 object_id 传给下一个 inputs ---
                inputs.append({
                    "image": roi,
                    "original_bbox_offset": (bbox[0], bbox[1]),
                    "label": det.get('name'),
                    "object_id": det.get("object_id")  # 透传 ID
                })
        return inputs

    @staticmethod
    def map_coords_back(local_results: list, offset: tuple):
        """将 ROI 上的相对坐标映射回原图全局坐标"""
        ox, oy = offset
        for res in local_results:
            if 'bbox' in res:
                b = res['bbox']
                res['bbox'] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
        return local_results

    def resnet_transform(self, image: Image.Image):
        return self.resnet_preprocess(image)

    # --------------------------------------------------------------------

    def prepare_step_inputs(self, prev_context: dict) -> list:
        """
        对应你代码里的 now_period_input。
        根据上一阶段的检测结果，裁剪出 ROI 作为本阶段的输入。
        """
        if not prev_context or "result" not in prev_context:
            return []

        inputs = []
        # 假设 prev_context['result'] 是一个 List[List[Dict]] (对应多张图的检测结果)
        # 我们这里简化处理第一张图的结果
        for img_idx, det_list in enumerate(prev_context['result']):
            # 获取对应的原始图片对象（存放在 input 字段中）
            raw_img = prev_context['input'][img_idx]
            if isinstance(raw_img, dict): raw_img = raw_img['image']

            for det in det_list:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                # 执行裁剪
                roi = raw_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                inputs.append({
                    "image": roi,
                    "original_bbox_offset": (bbox[0], bbox[1]),  # 记录偏移量用于坐标还原
                    "parent_class": det.get('name')
                })
        return inputs

    @staticmethod
    def extract_rois(image: np.ndarray, bboxes: list, expand_ratio: float = 0.0):
        """
        image: 原图 (H, W, C)
        bboxes: 格式为 [[x1, y1, x2, y2], ...]
        """
        rois = []
        h, w = image.shape[:2]
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            if expand_ratio > 0:
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - bw * expand_ratio))
                y1 = max(0, int(y1 - bh * expand_ratio))
                x2 = min(w, int(x2 + bw * expand_ratio))
                y2 = min(h, int(y2 + bh * expand_ratio))
            rois.append(image[y1:y2, x1:x2])
        return rois

    @staticmethod
    def get_top_k(probs: torch.Tensor, k: int = 5, names: dict = None):
        """从分类概率中提取 Top-K"""
        if probs.dim() > 1: probs = probs.squeeze()
        topk_values, topk_indices = torch.topk(probs, k)

        results = []
        for val, idx in zip(topk_values, topk_indices):
            idx_int = int(idx)
            results.append({
                "class_id": idx_int,
                "label": names[idx_int] if names else str(idx_int),
                "score": float(val)
            })
        return results

    @staticmethod
    def parse_yolo_results(yolo_output):
        """解析 Ultralytics YOLO 的 Result 对象"""
        # yolo_output 是 ultralytics.engine.results.Results
        boxes = yolo_output.boxes
        parsed = []
        for i in range(len(boxes)):
            parsed.append({
                "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                "conf": float(boxes.conf[i]),
                "cls_id": int(boxes.cls[i]),
                "name": yolo_output.names[int(boxes.cls[i])]
            })
        return parsed
