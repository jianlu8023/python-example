from typing import Dict, List, Optional

import torch
from PIL import Image

from .model_cache import ModelCache
from .model_infer_config import ModelInferConfig
from .model_processor import ModelProcessor
from .model_type import ModelType
from ..loggers import get_logger

logger = get_logger(__name__)


class ModelInferEngine:
    """
    ModelInferEngine 模型推理服务引擎
    """

    def __init__(self, device: str = None):
        """
        __init__
        :param device:
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else
                                 "cpu")
        self.model_cache = ModelCache()
        self.processor = ModelProcessor()

    def run_pipeline(self, configs: List[ModelInferConfig], original_img: Image.Image):
        """
        run_pipeline 链路形式推理
        :param configs: ModelInferConfig 列表
        :param original_img: 推理的图片
        :return:
        """
        context = None
        pipeline_results = []
        for idx, config in enumerate(configs):
            logger.info(f"当前进度 {idx + 1} / {len(configs)} -- 任务名称: {config.task_name} -- 开始推理.")
            context = self.run_step(config, idx, context, original_img)
            logger.info(f"当前进度 {idx + 1} / {len(configs)} -- 任务名称: {config.task_name} -- 推理完成.")
            pipeline_results.append(context)

            if not context.get("success") or not context.get("result"):
                break
        return pipeline_results

    def run_step(self, config: ModelInferConfig, step_idx: int,
                 prev_context: Optional[Dict], original_img: Image.Image):
        """
        run_step 推理一步
        :param config: 推理配置
        :param step_idx: 步骤index
        :param prev_context: 前一轮结果
        :param original_img: 推理图片
        :return:
        """

        # 1. 数据准备
        if step_idx == 0:
            # 第一步，输入原始大图
            inputs = [{"image": original_img, "original_bbox_offset": (0, 0)}]
        else:
            # 判断是否需要重新裁剪
            prev_results = prev_context.get('result', [])

            # 检查上一步的结果中是否有 bbox。
            # 只要有一个结果包含 bbox，我们就认为这是一个检测任务，需要执行裁剪。
            has_bbox = False
            if prev_results and len(prev_results) > 0:
                # prev_results 是 List[List[Dict]]，检查第一个非空列表里的第一个元素
                for sub_list in prev_results:
                    if sub_list and 'bbox' in sub_list[0]:
                        has_bbox = True
                        break

            if has_bbox:
                # 上一步是检测，这一步需要根据坐标裁剪 ROI
                logger.debug(f"步骤 {step_idx}: 检测到坐标，执行 ROI 裁剪")
                inputs = self.processor.crop_rois(original_img, prev_results)
            else:
                # 上一步是分类，没有新坐标，直接沿用上一步的输入图 (透传)
                logger.debug(f"步骤 {step_idx}: 未发现新坐标，透传上一步的输入图")
                inputs = prev_context.get('input', [])

        if not inputs:
            logger.warning(f"任务 {config.task_name} (Step {step_idx}) 无输入数据，跳过")
            return {"success": True, "task": config.task_name, "result": [], "input": []}

        # 2. 模型分发
        dispatch = {
            ModelType.YOLO_DETECT: self._infer_yolo_detect,
            ModelType.YOLO_CLASSIFY: self._infer_yolo_classify,
            ModelType.RESNET18: self._infer_resnet18,
        }

        if config.model_type not in dispatch:
            raise ValueError(f"未实现的模型逻辑: {config.model_type}")

        return dispatch[config.model_type](config, inputs)

    # ------------------ 内部具体实现 ------------------

    def _infer_yolo_detect(self, config: ModelInferConfig, inputs: List[Dict]):
        model = self.model_cache.get_yolo(config.model_path, self.device)
        all_res_nested = []  # 全量结果
        filtered_results = []  # 达标结果

        for img_idx, item in enumerate(inputs):
            # 1. 获取模型看到的“所有”框
            results = model.predict(
                item['image'],
                # conf=config.conf_threshold,
                # iou=config.iou_threshold,
                device=self.device,
                verbose=config.verbose
            )
            raw_boxes = results[0].boxes

            parsed_all_boxes = []
            for i in range(len(raw_boxes)):
                conf_val = float(raw_boxes.conf[i])
                cls_id = int(raw_boxes.cls[i])
                # 局部坐标 (仅保留基础坐标信息)
                bbox = raw_boxes.xyxy[i].cpu().numpy().tolist()

                # 记录每一个框
                res_obj = {
                    "object_id": item.get("object_id", f"obj_{img_idx}_{i}"),
                    "input_index": img_idx,
                    "bbox": bbox,
                    "conf": conf_val,
                    "cls_id": cls_id,
                    "name": config.labels.get(cls_id, str(cls_id)),
                    "task_name": config.task_name
                }
                parsed_all_boxes.append(res_obj)

                # 只有大于用户定义阈值的，才放入 filtered_results
                if conf_val >= config.conf_threshold:
                    filtered_results.append(res_obj)

            # 将本张图的所有框映射坐标后放入 result
            mapped = self.processor.map_coords_back(parsed_all_boxes, item['original_bbox_offset'])
            all_res_nested.append(mapped)

        return {
            "success": True,
            "task": config.task_name,
            "result": all_res_nested,  # 所有检测框
            "filtered_results": filtered_results,  # 仅包含高置信度结果
            "input": inputs
        }

    def _infer_resnet18(self, config: ModelInferConfig, inputs: List[Dict]):
        logger.debug(f"从缓存中获取resnet18模型...")
        model = self.model_cache.get_resnet18(config.model_path, config.num_classes, self.device)

        all_res_nested = []
        filtered_results = []

        for img_idx, item in enumerate(inputs):
            img_tensor = self.processor.resnet_transform(item['image']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]

            # 2. 遍历“所有”类别的概率
            current_roi_probs = []
            top_score, top_idx = 0.0, -1

            for idx, score in enumerate(probs.cpu().numpy().tolist()):
                score_val = float(score)
                res_obj = {
                    "object_id": item.get("object_id", "unknown"),
                    "input_index": img_idx,
                    "cls_id": idx,
                    "name": config.labels.get(idx, str(idx)),
                    "score": score_val,
                    "task_name": config.task_name
                }
                current_roi_probs.append(res_obj)

                # 找出 Top1 用于 filtered_results
                if score_val > top_score:
                    top_score, top_idx = score_val, idx

            all_res_nested.append(current_roi_probs)

            # 如果 Top1 达标，放入 filtered_results
            if top_score >= config.conf_threshold:
                filtered_results.append(current_roi_probs[top_idx])

        return {
            "success": True,
            "task": config.task_name,
            "result": all_res_nested,
            "filtered_results": filtered_results,
            "input": inputs
        }

    def _infer_yolo_classify(self, config, inputs):
        logger.debug(f"从缓存中获取yolo-classify模型...")
        model = self.model_cache.get_yolo(config.model_path, self.device)

        all_res_nested = []
        filtered_results = []

        for img_idx, item in enumerate(inputs):
            results = model.predict(
                item['image'],
                device=self.device,
                verbose=config.verbose,
            )
            result = results[0]

            if result.probs is not None:
                # 3. 提取“所有”类别的置信度数据
                all_probs = result.probs.data.cpu().numpy().tolist()
                class_names = result.names

                current_roi_probs = []
                top1_idx = int(result.probs.top1)

                for idx, score in enumerate(all_probs):
                    score_val = float(score)
                    res_obj = {
                        "object_id": item.get("object_id", "unknown"),
                        "input_index": img_idx,
                        "cls_id": idx,
                        "name": config.labels.get(idx, class_names[idx]),
                        "score": score_val,
                        "task_name": config.task_name
                    }
                    current_roi_probs.append(res_obj)

                    # 如果是 Top1 且达标，放入 filtered_results
                    if idx == top1_idx and score_val >= config.conf_threshold:
                        filtered_results.append(res_obj)

                all_res_nested.append(current_roi_probs)
            else:
                all_res_nested.append([])

        return {
            "success": True,
            "task": config.task_name,
            "result": all_res_nested,
            "filtered_results": filtered_results,
            "input": inputs
        }
