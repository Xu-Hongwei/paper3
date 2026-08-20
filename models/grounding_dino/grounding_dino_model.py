import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingDINODetector:
    """Hugging Face Grounding DINO 薄封装，只负责文本条件目标框定位。"""

    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-tiny",
        device=None,
        local_files_only=False,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self.model_id = model_id

    @staticmethod
    def _to_pil(image_rgb):
        if isinstance(image_rgb, Image.Image):
            return image_rgb.convert("RGB")

        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb 必须为 HWC RGB 三通道图像。")
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(image_rgb, mode="RGB")

    @torch.inference_mode()
    def predict(
        self,
        image_rgb,
        text,
        box_threshold=0.25,
        text_threshold=0.20,
    ):
        """
        返回：
        {
            "boxes": np.ndarray[N,4],  # xyxy，原输入图像坐标
            "scores": np.ndarray[N],
            "labels": list[str]
        }
        """
        image = self._to_pil(image_rgb)
        text = str(text).strip().lower()
        if not text:
            raise ValueError("text 不能为空。")

        # 当前 Transformers Grounding DINO API 使用嵌套 text_labels。
        text_labels = [[text]]
        inputs = self.processor(
            images=image,
            text=text_labels,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)

        labels = result.get("labels", result.get("text_labels", []))
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().tolist()
        labels = [str(label) for label in labels]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
