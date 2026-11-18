import cv2
import numpy as np
from PIL import ImageFont

from .onnx_infer import OnnxBase


class YoloOnnx(OnnxBase):
    def __init__(self, onnx_path, class_names, conf_threshold=0.25, iou_threshold=0.45):
        super().__init__(onnx_path)

        if isinstance(class_names, list):
            self.class_names = {i: name for i, name in enumerate(class_names)}
        else:
            self.class_names = class_names

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.input_h = int(self.input_shape[2]) if isinstance(self.input_shape[2], int) else 640
        self.input_w = int(self.input_shape[3]) if isinstance(self.input_shape[3], int) else 640

    def preprocess(self, input_data):

        # 1. 统一转为 numpy (RGB)
        img = self._get_base_rgb(input_data)

        # 2. Letterbox 缩放
        img = self._apply_letterbox(img, (self.input_h, self.input_w))

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, outputs):

        prediction = np.transpose(np.squeeze(outputs[0]))

        rows = prediction.shape[0]
        boxes, scores, class_ids = [], [], []

        # 2. 遍历每一行 (每个 anchor)
        for i in range(rows):
            # 取出类别分数 (从索引 4 开始)
            classes_scores = prediction[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= self.conf_threshold:
                class_id = np.argmax(classes_scores)

                # 获取网络输出的 cx, cy, w, h
                x, y, w, h = prediction[i][0], prediction[i][1], prediction[i][2], prediction[i][3]

                # 1. 移除填充
                x -= self.dw
                y -= self.dh
                # 2. 缩放回原图
                x /= self.ratio
                y /= self.ratio
                w /= self.ratio
                h /= self.ratio

                # 3. 转换为 [left, top, width, height] 供 cv2.dnn.NMSBoxes 使用
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)

                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(int(class_id))

        # 3. 执行 NMS
        results = []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]  # [left, top, width, height]
                score = scores[i]
                cid = class_ids[i]

                # 转换为 PIL 绘图需要的 [x1, y1, x2, y2]
                res_box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

                results.append({
                    "box": res_box,
                    "score": score,
                    "class_id": cid,
                    "label": f"{self.class_names.get(cid, str(cid))}: {score:.2f}"
                })

        return results

    def _draw_custom(self, draw, results):
        """
        使用 PIL 绘制结果
        """
        try:
            # 尽量使用字体，如果失败则使用默认
            font = ImageFont.truetype("arial.ttf", size=18)
        except OSError:
            font = ImageFont.load_default()

        for res in results:
            box = res["box"]
            label = res["label"]

            # 绘制矩形框
            draw.rectangle(box, outline="red", width=3)

            # 绘制标签背景
            text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(text_bbox, fill="red")

            # 绘制文字
            draw.text((box[0], box[1]), label, fill="white", font=font)
