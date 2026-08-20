from contextlib import nullcontext

import numpy as np
import torch


class SAM2Segmenter:
    """SAM2.1 静态图像分割薄封装。"""

    def __init__(
        self,
        checkpoint,
        model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
        device=None,
        use_bfloat16=True,
    ):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "未安装官方 SAM2。请先安装 facebookresearch/sam2。"
            ) from exc

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_bfloat16 = bool(
            use_bfloat16 and self.device.type == "cuda"
        )

        sam_model = build_sam2(
            model_cfg,
            checkpoint,
            device=str(self.device),
        )
        self.predictor = SAM2ImagePredictor(sam_model)

    def _autocast(self):
        if not self.use_bfloat16:
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    @torch.inference_mode()
    def set_image(self, image_rgb):
        """image_rgb: HWC RGB uint8，范围 [0, 255]。"""
        image_rgb = np.asarray(image_rgb)

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb 必须为 HWC RGB 三通道图像。")

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        with self._autocast():
            self.predictor.set_image(image_rgb)

    def _format_outputs(self, masks, scores, low_res_logits, sort_by_score):
        masks = np.asarray(masks) > 0.5
        scores = np.asarray(scores)
        low_res_logits = np.asarray(low_res_logits)

        if sort_by_score:
            order = np.argsort(scores)[::-1]
            masks = masks[order]
            scores = scores[order]
            low_res_logits = low_res_logits[order]

        return masks, scores, low_res_logits

    @torch.inference_mode()
    def predict_box(self, box, multimask_output=True, sort_by_score=True):
        """box: [x1, y1, x2, y2]。"""
        box = np.asarray(box, dtype=np.float32)

        if box.shape != (4,):
            raise ValueError("box 必须是 [x1, y1, x2, y2]。")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box 必须满足 x2>x1 且 y2>y1。")

        with self._autocast():
            outputs = self.predictor.predict(
                box=box,
                multimask_output=multimask_output,
                return_logits=False,
            )

        return self._format_outputs(*outputs, sort_by_score)

    @torch.inference_mode()
    def predict_points(
        self,
        point_coords,
        point_labels=None,
        multimask_output=True,
        sort_by_score=True,
    ):
        """
        point_coords: [N,2]，每行为 [x,y]。
        point_labels: [N]，1=positive，0=negative；默认全部 positive。
        """
        point_coords = np.asarray(point_coords, dtype=np.float32)

        if point_coords.ndim != 2 or point_coords.shape[1] != 2:
            raise ValueError("point_coords 必须为 [N,2]。")
        if len(point_coords) == 0:
            raise ValueError("至少需要一个 point。")

        if point_labels is None:
            point_labels = np.ones(len(point_coords), dtype=np.int32)
        else:
            point_labels = np.asarray(point_labels, dtype=np.int32)

        if point_labels.shape != (len(point_coords),):
            raise ValueError("point_labels 必须为 [N]。")

        with self._autocast():
            outputs = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
                return_logits=False,
            )

        return self._format_outputs(*outputs, sort_by_score)

    def reset(self):
        self.predictor.reset_predictor()
