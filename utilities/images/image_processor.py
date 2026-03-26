import base64
import hashlib
from io import BytesIO
from typing import Tuple

from PIL import Image, ImageOps


class ImageProcessor:

    @staticmethod
    def apply_exif_orientation_by_official(img: Image.Image) -> Image.Image:
        try:
            # ImageOps.exif_transpose 是 Pillow 内置的高级函数
            # 它会自动检查 EXIF 中的 Orientation 标签并进行相应的旋转或翻转
            # 如果没有 EXIF 或方向正确，它会返回原图的副本或原图
            return ImageOps.exif_transpose(img)
        except Exception as e:
            print(f"处理EXIF方向时出错: {e}")
            return img

    @staticmethod
    def apply_exif_orientation(img: Image.Image) -> Image.Image:
        """
        apply_exif_orientation 应用图像的EXIF方向信息，自动旋转图像到正确方向
        :param img: PIL图像对象
        :return: 应用了EXIF方向的图像
        """
        try:
            # 1. 获取 EXIF 数据 (优先使用官方推荐的 getexif)
            exif = img.getexif() if hasattr(img, 'getexif') else img._getexif()

            if exif is None:
                print("提示: 图像无EXIF数据，保持原样")
                return img

            # 2. 定位 Orientation 标签 (标准 ID 为 274)
            orientation_key = 274  # 0x0112: Orientation
            orientation = exif.get(orientation_key, None)

            if orientation is None or orientation == 1:
                # 1 代表正常方向，无需处理
                return img

            # 3. 准备兼容性常量
            if hasattr(Image, 'Transpose'):
                trans = Image.Transpose
            else:
                trans = Image  # 兼容老版本 Pillow

            # 记录原始尺寸用于对比
            old_size = img.size
            action_msg = ""

            # 4. 根据方向值进行旋转/翻转 (1-8 种情况)
            if orientation == 2:
                img = img.transpose(trans.FLIP_LEFT_RIGHT)
                action_msg = "执行 [水平翻转]"
            elif orientation == 3:
                img = img.transpose(trans.ROTATE_180)
                action_msg = "执行 [旋转 180 度]"
            elif orientation == 4:
                img = img.transpose(trans.FLIP_TOP_BOTTOM)
                action_msg = "执行 [垂直翻转]"
            elif orientation == 5:
                # 这里的顺序是先转 90 再翻转，等同于 TRANSPOSE
                img = img.transpose(trans.TRANSPOSE)
                action_msg = "执行 [顺时针 90 度 + 水平翻转]"
            elif orientation == 6:
                # 顺时针旋转 90 度 (在 PIL 中等同于 rotate 270 或使用特定常量)
                img = img.transpose(trans.ROTATE_270)
                action_msg = "执行 [顺时针 90 度旋转]"
            elif orientation == 7:
                img = img.transpose(trans.TRANSVERSE)
                action_msg = "执行 [顺时针 90 度 + 垂直翻转]"
            elif orientation == 8:
                img = img.transpose(trans.ROTATE_90)
                action_msg = "执行 [逆时针 90 度旋转]"

            # 5. 打印详细信息
            if action_msg:
                print(f"检测到 EXIF 方向: {orientation} | 原始尺寸: {old_size} -> {action_msg} -> 新尺寸: {img.size}")
            return img
        except Exception as e:
            print(f"警告: 处理 EXIF 方向时发生未知错误: {e}")
            return img

    @staticmethod
    def get_hash_and_base64(img: Image.Image) -> Tuple[str, str]:
        """
        get_hash_and_base64 生成唯一哈希和Base64供存储
        :param img: 图片对象
        :return Tuple[str,str]: 返回图片的sha256和base64
        """
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        img_bytes = buffer.getvalue()

        sha256 = hashlib.sha256(img_bytes).hexdigest()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return sha256, img_b64
