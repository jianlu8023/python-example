# yolo-detect-infer.py yolo推理detect

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from utilities import apply_exif_orientation, get_dir_all_files, get_filename

CLASSES = {
    0: "0",
}

base_path = "python-example"

output_dir = f"{base_path}/testdata/"
os.makedirs(output_dir, exist_ok=True)

model = YOLO(f"{base_path}/output/yolo/best.pt")


def infer():
    input_dir = f"{base_path}/testdata/测试数据"
    all_files = get_dir_all_files(input_dir, recursion=False, extensions=(".jpg", ".jpeg", ".png"))

    for filename in all_files:
        image_name = get_filename(filename)
        print(f"当前正在处理文件: {image_name}")
        results = model(filename, conf=0.5)  # 设置置信度阈值
        result = results[0]  # 单张图，取第一个
        boxes = result.boxes  # Boxes object
        if len(boxes) == 0:
            Image.open(filename).save(os.path.join(output_dir, f"yolo-best-{image_name}.jpg"))
            print("No detections.")
            continue
        # 转为 numpy 数组以便处理
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        confidences = boxes.conf.cpu().numpy()  # (N,)
        class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

        # 读取原始图像（OpenCV BGR -> RGB）
        image_bgr = cv2.imread(filename)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 转为 PIL 图像以便绘制中文
        pil_image = Image.fromarray(image_rgb)
        pil_image = apply_exif_orientation(pil_image)
        draw = ImageDraw.Draw(pil_image)
        font_path = f"{base_path}/ttf/NotoSansCJK-Regular.ttc"
        try:
            font = ImageFont.truetype(font_path, size=20)
        except OSError:
            print(f"Warning: Font not found at {font_path}, using default font (may not support Chinese).")
            font = ImageFont.load_default()
        # 绘制每个检测框和标签
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            conf = confidences[i]
            cls_id = class_ids[i]
            label = f"{CLASSES.get(cls_id, '未知')} {conf:.4f}"

            # 绘制边界框（使用 OpenCV 颜色格式：RGB）
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            # 计算文本尺寸（Pillow >= 8.0 使用 textbbox）
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 绘制标签背景
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1], fill=(255, 0, 0))

            # 绘制标签文字
            draw.text((x1, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)

        # 保存结果
        # 转回 NumPy 并保存（需转回 BGR）
        final_image_rgb = np.array(pil_image)
        final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f"yolo-best-{image_name}.jpg")
        cv2.imwrite(output_path, final_image_bgr)

        print(f"Result saved to {output_path}")


if __name__ == '__main__':
    infer()
