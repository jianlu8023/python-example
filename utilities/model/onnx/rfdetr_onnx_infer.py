import numpy as np
from PIL import ImageFont

from .onnx_infer import OnnxBase


class RfdetrOnnx(OnnxBase):

    def __init__(self, onnx_path, class_names, conf_threshold=0.4):
        super().__init__(onnx_path)

        self.class_names = class_names
        self.conf_threshold = conf_threshold

        self.input_h = int(self.input_shape[2]) if isinstance(self.input_shape[2], int) else 576
        self.input_w = int(self.input_shape[3]) if isinstance(self.input_shape[3], int) else 576

    def preprocess(self, input_data):
        """
        preprocess
        input_data: 可以是 PIL.Image 或 numpy 数组 (BGR)
        """
        # 1. 统一转为 numpy (RGB)
        img = self._get_base_rgb(input_data)

        # 2. Letterbox 缩放
        img = self._apply_letterbox(img, (self.input_h, self.input_w))

        # 3. 归一化与标准化
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # 4. HWC -> CHW, 增加 Batch 维度
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def postprocess(self, outputs):
        raw_boxes, raw_logits = outputs

        # 处理分类分数
        probs = self._sigmoid(raw_logits[0])
        scores = np.max(probs, axis=-1)
        class_ids = np.argmax(probs, axis=-1)

        results = []
        for i in range(len(scores)):
            score = scores[i]
            if score < self.conf_threshold:
                continue

            # 获取归一化的 cx, cy, w, h (相对于模型输入尺寸)
            cx, cy, w, h = raw_boxes[0][i]

            # 1. 先恢复到 letterbox 填充后的图坐标
            x1 = (cx - 0.5 * w) * self.input_w
            y1 = (cy - 0.5 * h) * self.input_h
            x2 = (cx + 0.5 * w) * self.input_w
            y2 = (cy + 0.5 * h) * self.input_h

            # 2. 减去实际的左边和顶部的填充 (dw, dh)
            x1 -= self.dw
            x2 -= self.dw
            y1 -= self.dh
            y2 -= self.dh

            # 3. 缩放回原图
            x1 /= self.ratio
            x2 /= self.ratio
            y1 /= self.ratio
            y2 /= self.ratio

            # 4. 边界裁剪
            x1 = int(max(0, min(self.ori_w, x1)))
            y1 = int(max(0, min(self.ori_h, y1)))
            x2 = int(max(0, min(self.ori_w, x2)))
            y2 = int(max(0, min(self.ori_h, y2)))

            label = f"{self.class_names[class_ids[i]]}: {score:.2f}"

            results.append({
                "class_id": int(class_ids[i]),
                "score": float(score),
                "box": [x1, y1, x2, y2],
                "label": label,
            })

        return results

    def _draw_custom(self, draw, results):
        """
        专门用于绘制检测框的逻辑
        """
        try:
            # 尝试加载字体
            font = ImageFont.truetype("arial.ttf", size=18)
        except OSError:
            font = ImageFont.load_default()

        for result in results:
            box = result["box"]
            label = result["label"]

            # 画框
            draw.rectangle(box, outline="lime", width=3)

            # 画标签背景
            text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(text_bbox, fill="lime")

            # 写字
            draw.text((box[0], box[1]), label, fill="black", font=font)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
