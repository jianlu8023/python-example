from abc import ABC, abstractmethod

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


class OnnxBase(ABC):
    def __init__(self, onnx_path, providers=None):
        self.ori_h = None
        self.ori_w = None
        self.dh = None
        self.dw = None
        self.ratio = None

        self.onnx_path = onnx_path
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ort.get_device() == "GPU"
                else ["CPUExecutionProvider"]
            )

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    @abstractmethod
    def preprocess(self, input_data):
        """
        preprocess 预处理
        :param input_data: 输入
        :return:
        """
        pass

    @abstractmethod
    def postprocess(self, outputs):
        """
        postprocess 后处理
        :param outputs: onnx推理结果
        :return: [{
                "class_id": 0,
                "score": score,
                "box": [x1, y1, x2, y2],
                "label": label,
            }]
        """
        pass

    @abstractmethod
    def _draw_custom(self, img_draw, results):
        """
        _draw_custom 自定义绘制
        :param img_draw:
        :param results: 推理结果
        :return:
        """
        pass

    def _get_base_rgb(self, input_data):
        """通用：转为 RGB 格式并记录原图尺寸"""
        if isinstance(input_data, Image.Image):
            self.ori_w, self.ori_h = input_data.size
            img = np.array(input_data.convert("RGB"))
        elif isinstance(input_data, np.ndarray):
            self.ori_h, self.ori_w = input_data.shape[:2]
            img = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported input type")
        return img

    def _apply_letterbox(self, img, target_size):
        """通用：执行 letterbox 并自动保存缩放信息"""
        img, self.ratio, (self.dw, self.dh) = self._letterbox(img, target_size)
        return img

    def infer(self, input_tensor):
        return self.session.run(None, {self.input_name: input_tensor})

    def predict(self, input_data):
        input_tensor = self.preprocess(input_data)
        outputs = self.infer(input_tensor)
        return self.postprocess(outputs)

    def draw_results(self, image, results):
        """
        通用绘图入口
        :param image: PIL.Image 对象 (原始图像)
        :param results: postprocess 返回的结果列表
        :return: PIL.Image 对象 (绘制后的图像)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        self._draw_custom(draw, results)
        return img_draw

    @staticmethod
    def _letterbox(img, new_shape: tuple, color=(114, 114, 114)):
        shape = img.shape[:2]  # 当前形状 [高, 宽]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 缩放后的未填充尺寸
        new_un_padding = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # 计算总填充量
        dw = new_shape[1] - new_un_padding[0]  # 宽度总缺失
        dh = new_shape[0] - new_un_padding[1]  # 高度总缺失

        # 分配到两侧：关键改动！
        # 使用 // 2 确保整数，然后用总数减去左侧，得到右侧，保证总和绝对等于 dw/dh
        dw /= 2
        dh /= 2

        # 如果你觉得上面的 round 逻辑复杂，最稳妥的办法是：
        # top = dh // 2
        # bottom = dh - top
        # left = dw // 2
        # right = dw - left
        # 但为了对齐原有的浮点偏移逻辑，通常使用以下写法：
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        if shape[::-1] != new_un_padding:
            img = cv2.resize(img, new_un_padding, interpolation=cv2.INTER_LINEAR)

        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)

        return img, float(r), (float(left), float(top))  # 注意这里返回左/上的偏移即可
