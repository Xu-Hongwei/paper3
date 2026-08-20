import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossCategoryMarginLoss(nn.Module):
    """
    跨类别 batch-hard margin loss，支持三种 T2I 模式：

        none:
            在全部合法跨类候选中直接选择 hardest negative。

        post_gate:
            先选择 fixed hardest negative，再判断该 pair 是否可靠；
            只有 A 区 pair 参与额外 T2I margin loss。

        reliable_mining:
            先对全部跨类候选构造 reliability mask，只保留 A 区候选；
            再从可靠候选中选择 hardest negative。

    A 区定义：
        G_sup(i,j) = S(T_i, c_i) - S(T_i, c_j) > support_threshold
        G_cat(i,j) = S(c_i, I_i) - S(c_i, I_j) > category_threshold

    I2T 始终保留原 fixed cross-category hard-negative 逻辑，
    便于只验证 T2I reliability。margin 作用于 raw cosine similarity。
    """

    VALID_MODES = {"none", "post_gate", "reliable_mining"}

    def __init__(
        self,
        margin=0.05,
        reliable_t2i=False,
        support_threshold=0.0,
        category_threshold=0.0,
        reliability_mode=None,
    ):
        super().__init__()

        self.margin = float(margin)
        self.support_threshold = float(support_threshold)
        self.category_threshold = float(category_threshold)

        # 兼容上一版 train.py：只传 reliable_t2i=True 时仍表示 post_gate。
        if reliability_mode is None:
            reliability_mode = (
                "post_gate"
                if reliable_t2i
                else "none"
            )
        reliability_mode = str(reliability_mode).lower()

        if reliability_mode not in self.VALID_MODES:
            raise ValueError(
                "reliability_mode must be one of "
                f"{sorted(self.VALID_MODES)}, got {reliability_mode!r}."
            )

        if reliable_t2i and reliability_mode == "none":
            raise ValueError(
                "reliable_t2i=True 与 reliability_mode='none' 冲突。"
            )

        if self.margin < 0:
            raise ValueError("margin must be >= 0.")

        self.reliability_mode = reliability_mode
        # 保留旧 trainer 的兼容检查。
        self.reliable_t2i = reliability_mode != "none"

    @staticmethod
    def _check_inputs(
        image_features,
        text_features,
        category_ids,
        caption_support=None,
        image_support=None,
        need_support=False,
    ):
        if image_features.ndim != 2:
            raise ValueError(
                "image_features must have shape [B, D]."
            )
        if text_features.ndim != 2:
            raise ValueError(
                "text_features must have shape [B, D]."
            )
        if image_features.shape != text_features.shape:
            raise ValueError(
                "image_features and text_features must have "
                "the same shape. "
                f"Got {tuple(image_features.shape)} and "
                f"{tuple(text_features.shape)}."
            )

        batch_size = image_features.shape[0]

        if category_ids.ndim != 1:
            raise ValueError(
                "category_ids must have shape [B]."
            )
        if category_ids.shape[0] != batch_size:
            raise ValueError(
                f"category_ids must have shape [{batch_size}], "
                f"got {tuple(category_ids.shape)}."
            )

        if need_support:
            if caption_support is None or image_support is None:
                raise ValueError(
                    "Reliable T2I requires caption_support and image_support."
                )
            if caption_support.ndim != 2:
                raise ValueError(
                    "caption_support must have shape [B, C]."
                )
            if image_support.ndim != 2:
                raise ValueError(
                    "image_support must have shape [B, C]."
                )
            if caption_support.shape[0] != batch_size:
                raise ValueError(
                    f"caption_support first dim must be {batch_size}, "
                    f"got {tuple(caption_support.shape)}."
                )
            if image_support.shape[0] != batch_size:
                raise ValueError(
                    f"image_support first dim must be {batch_size}, "
                    f"got {tuple(image_support.shape)}."
                )
            if caption_support.shape[1] != image_support.shape[1]:
                raise ValueError(
                    "caption_support and image_support must use "
                    "the same number of categories."
                )

    @staticmethod
    def _zero_loss(image_features, text_features):
        """返回保持计算图连接的标量 0。"""
        return (
            image_features.sum()
            + text_features.sum()
        ) * 0.0

    @staticmethod
    def _empty_bool(batch_size, device):
        return torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

    @staticmethod
    def _safe_hard_index(
        hard_neg_index,
        valid_mask,
    ):
        """无效 anchor 的 hard index 统一为 -1。"""
        return torch.where(
            valid_mask,
            hard_neg_index,
            torch.full_like(
                hard_neg_index,
                -1,
            ),
        )

    @staticmethod
    def _check_category_range(
        category_ids,
        valid_mask,
        num_categories,
    ):
        if torch.any(
            valid_mask
            & (
                (category_ids < 0)
                | (category_ids >= num_categories)
            )
        ):
            raise ValueError(
                "Valid anchors contain category_id outside "
                "support cache range."
            )

    def _fixed_pair_reliability(
        self,
        valid_mask,
        hard_neg_index,
        category_ids,
        caption_support,
        image_support,
    ):
        """计算 fixed hardest pair 的 G_sup/G_cat 与 A/B/C/D。"""
        batch_size = category_ids.shape[0]
        device = category_ids.device
        rows = torch.arange(batch_size, device=device)

        num_categories = caption_support.shape[1]
        self._check_category_range(
            category_ids,
            valid_mask,
            num_categories,
        )

        safe_category = category_ids.clamp(
            min=0,
            max=num_categories - 1,
        )
        safe_neg_index = hard_neg_index.clamp_min(0)
        neg_category = safe_category[safe_neg_index]

        g_sup = (
            caption_support[rows, safe_category]
            - caption_support[rows, neg_category]
        )
        g_cat = (
            image_support[rows, safe_category]
            - image_support[safe_neg_index, safe_category]
        )

        sup_positive = g_sup > self.support_threshold
        cat_positive = g_cat > self.category_threshold

        region_a = valid_mask & sup_positive & cat_positive
        region_b = valid_mask & ~sup_positive & cat_positive
        region_c = valid_mask & sup_positive & ~cat_positive
        region_d = valid_mask & ~sup_positive & ~cat_positive

        g_sup = torch.where(
            valid_mask,
            g_sup,
            torch.zeros_like(g_sup),
        )
        g_cat = torch.where(
            valid_mask,
            g_cat,
            torch.zeros_like(g_cat),
        )

        return {
            "g_sup": g_sup,
            "g_cat": g_cat,
            "region_a_mask": region_a,
            "region_b_mask": region_b,
            "region_c_mask": region_c,
            "region_d_mask": region_d,
        }

    def _pairwise_reliability(
        self,
        negative_mask,
        category_ids,
        caption_support,
        image_support,
    ):
        """
        对所有 T2I (anchor caption i, candidate image j) 构造可靠性。

        返回：
            reliable_candidate_mask: [B, B]
            g_sup_pair: [B, B]
            g_cat_pair: [B, B]
        """
        num_categories = caption_support.shape[1]
        known = category_ids >= 0
        self._check_category_range(
            category_ids,
            known,
            num_categories,
        )

        safe_category = category_ids.clamp(
            min=0,
            max=num_categories - 1,
        )

        # G_sup(i,j) = caption_i 对 GT 类的支持 - 对 candidate 类的支持
        gt_caption_support = caption_support.gather(
            1,
            safe_category[:, None],
        )
        candidate_class_support = caption_support[
            :,
            safe_category,
        ]
        g_sup_pair = (
            gt_caption_support
            - candidate_class_support
        )

        # G_cat(i,j) = GT 类名对 GT image_i 的支持 - 对 candidate image_j 的支持
        gt_image_support = image_support.gather(
            1,
            safe_category[:, None],
        )
        candidate_image_support = image_support[
            :,
            safe_category,
        ].t()
        g_cat_pair = (
            gt_image_support
            - candidate_image_support
        )

        reliable_candidate_mask = (
            negative_mask
            & (g_sup_pair > self.support_threshold)
            & (g_cat_pair > self.category_threshold)
        )

        return {
            "reliable_candidate_mask": reliable_candidate_mask,
            "g_sup_pair": g_sup_pair,
            "g_cat_pair": g_cat_pair,
        }

    def _direction_loss(
        self,
        similarities,
        category_ids,
        image_features,
        text_features,
        direction,
        caption_support=None,
        image_support=None,
    ):
        """
        similarities[i, j]:
            anchor i 与 candidate j 的 cosine similarity。

        T2I 可使用 none / post_gate / reliable_mining；
        I2T 始终使用 fixed cross-category mining。
        """
        batch_size = similarities.shape[0]
        device = similarities.device

        known = category_ids >= 0
        negative_mask = (
            known[:, None]
            & known[None, :]
            & (
                category_ids[:, None]
                != category_ids[None, :]
            )
        )

        has_negative = negative_mask.any(dim=1)
        valid_mask = known & has_negative
        valid_count = int(valid_mask.sum().item())

        pos_sim = similarities.diagonal()

        # --------------------------------------------------
        # 1. Fixed hardest negative：始终计算，作为对照诊断。
        # --------------------------------------------------
        fixed_masked_sim = similarities.masked_fill(
            ~negative_mask,
            float("-inf"),
        )
        fixed_hard_neg_sim, fixed_hard_neg_index = (
            fixed_masked_sim.max(dim=1)
        )
        safe_fixed_hard_neg_sim = torch.where(
            valid_mask,
            fixed_hard_neg_sim,
            torch.zeros_like(fixed_hard_neg_sim),
        )
        fixed_per_anchor_loss = F.relu(
            self.margin
            - pos_sim
            + safe_fixed_hard_neg_sim
        )
        active_before_gate = (
            valid_mask
            & (fixed_per_anchor_loss > 0)
        )

        # 默认选择 fixed hard negative。
        selected_valid_mask = valid_mask
        selected_hard_neg_index = fixed_hard_neg_index
        selected_hard_neg_sim = safe_fixed_hard_neg_sim

        fixed_reliability = None
        selected_g_sup = similarities.new_zeros(batch_size)
        selected_g_cat = similarities.new_zeros(batch_size)

        reliable_candidate_total = 0
        avg_reliable_negatives = 0.0
        no_reliable_count = 0
        replacement_mask = self._empty_bool(
            batch_size,
            device,
        )

        # --------------------------------------------------
        # 2. T2I reliability
        # --------------------------------------------------
        if (
            direction == "t2i"
            and self.reliability_mode != "none"
        ):
            fixed_reliability = self._fixed_pair_reliability(
                valid_mask=valid_mask,
                hard_neg_index=fixed_hard_neg_index,
                category_ids=category_ids,
                caption_support=caption_support,
                image_support=image_support,
            )

            if self.reliability_mode == "post_gate":
                # 旧版：fixed hardest 若不是 A 区，则直接丢掉该 anchor。
                selected_valid_mask = fixed_reliability[
                    "region_a_mask"
                ]
                selected_g_sup = fixed_reliability["g_sup"]
                selected_g_cat = fixed_reliability["g_cat"]

            elif self.reliability_mode == "reliable_mining":
                # 新版：先过滤全部候选，再在 A 区候选中选 hardest。
                pairwise = self._pairwise_reliability(
                    negative_mask=negative_mask,
                    category_ids=category_ids,
                    caption_support=caption_support,
                    image_support=image_support,
                )
                reliable_candidate_mask = pairwise[
                    "reliable_candidate_mask"
                ]
                reliable_per_anchor = (
                    reliable_candidate_mask.sum(dim=1)
                )
                selected_valid_mask = (
                    valid_mask
                    & (reliable_per_anchor > 0)
                )

                reliable_candidate_total = int(
                    reliable_per_anchor.sum().item()
                )
                avg_reliable_negatives = (
                    reliable_candidate_total
                    / max(valid_count, 1)
                )
                no_reliable_count = int(
                    (
                        valid_mask
                        & ~selected_valid_mask
                    ).sum().item()
                )

                reliable_masked_sim = similarities.masked_fill(
                    ~reliable_candidate_mask,
                    float("-inf"),
                )
                (
                    reliable_hard_neg_sim,
                    reliable_hard_neg_index,
                ) = reliable_masked_sim.max(dim=1)

                selected_hard_neg_index = reliable_hard_neg_index
                selected_hard_neg_sim = torch.where(
                    selected_valid_mask,
                    reliable_hard_neg_sim,
                    torch.zeros_like(reliable_hard_neg_sim),
                )

                safe_selected_index = (
                    reliable_hard_neg_index.clamp_min(0)
                )
                rows = torch.arange(
                    batch_size,
                    device=device,
                )
                selected_g_sup = pairwise[
                    "g_sup_pair"
                ][rows, safe_selected_index]
                selected_g_cat = pairwise[
                    "g_cat_pair"
                ][rows, safe_selected_index]
                selected_g_sup = torch.where(
                    selected_valid_mask,
                    selected_g_sup,
                    torch.zeros_like(selected_g_sup),
                )
                selected_g_cat = torch.where(
                    selected_valid_mask,
                    selected_g_cat,
                    torch.zeros_like(selected_g_cat),
                )

                replacement_mask = (
                    selected_valid_mask
                    & (
                        reliable_hard_neg_index
                        != fixed_hard_neg_index
                    )
                )

        selected_count = int(
            selected_valid_mask.sum().item()
        )

        # --------------------------------------------------
        # 3. Selected hard negative 上计算 margin loss。
        # --------------------------------------------------
        per_anchor_loss = F.relu(
            self.margin
            - pos_sim
            + selected_hard_neg_sim
        )
        active_mask = (
            selected_valid_mask
            & (per_anchor_loss > 0)
        )

        if selected_count > 0:
            loss = per_anchor_loss[
                selected_valid_mask
            ].mean()
            mean_pos_sim = pos_sim[
                selected_valid_mask
            ].mean()
            mean_hard_neg_sim = selected_hard_neg_sim[
                selected_valid_mask
            ].mean()
            mean_g_sup = selected_g_sup[
                selected_valid_mask
            ].mean()
            mean_g_cat = selected_g_cat[
                selected_valid_mask
            ].mean()
        else:
            loss = self._zero_loss(
                image_features,
                text_features,
            )
            mean_pos_sim = similarities.new_zeros(())
            mean_hard_neg_sim = similarities.new_zeros(())
            mean_g_sup = similarities.new_zeros(())
            mean_g_cat = similarities.new_zeros(())

        if valid_count > 0:
            mean_fixed_hard_neg_sim = (
                safe_fixed_hard_neg_sim[
                    valid_mask
                ].mean()
            )
        else:
            mean_fixed_hard_neg_sim = similarities.new_zeros(())

        # fixed-hard A/B/C/D 仅用于诊断。
        if fixed_reliability is None:
            region_a_mask = self._empty_bool(
                batch_size,
                device,
            )
            region_b_mask = self._empty_bool(
                batch_size,
                device,
            )
            region_c_mask = self._empty_bool(
                batch_size,
                device,
            )
            region_d_mask = self._empty_bool(
                batch_size,
                device,
            )
            fixed_g_sup = similarities.new_zeros(batch_size)
            fixed_g_cat = similarities.new_zeros(batch_size)
        else:
            region_a_mask = fixed_reliability[
                "region_a_mask"
            ]
            region_b_mask = fixed_reliability[
                "region_b_mask"
            ]
            region_c_mask = fixed_reliability[
                "region_c_mask"
            ]
            region_d_mask = fixed_reliability[
                "region_d_mask"
            ]
            fixed_g_sup = fixed_reliability["g_sup"]
            fixed_g_cat = fixed_reliability["g_cat"]

        fixed_hard_neg_index = self._safe_hard_index(
            fixed_hard_neg_index,
            valid_mask,
        )
        selected_hard_neg_index = self._safe_hard_index(
            selected_hard_neg_index,
            selected_valid_mask,
        )

        replacement_count = int(
            replacement_mask.sum().item()
        )

        return {
            "loss": loss,
            "valid_mask": valid_mask,
            "reliable_mask": selected_valid_mask,
            "active_before_gate_mask": active_before_gate,
            "active_mask": active_mask,
            "valid_count": valid_count,
            "reliable_count": selected_count,
            "active_before_gate_count": int(
                active_before_gate.sum().item()
            ),
            "active_count": int(active_mask.sum().item()),

            "pos_sim": pos_sim,
            "hard_neg_sim": selected_hard_neg_sim,
            "hard_neg_index": selected_hard_neg_index,
            "fixed_hard_neg_sim": safe_fixed_hard_neg_sim,
            "fixed_hard_neg_index": fixed_hard_neg_index,
            "mean_pos_sim": mean_pos_sim,
            "mean_hard_neg_sim": mean_hard_neg_sim,
            "mean_fixed_hard_neg_sim": mean_fixed_hard_neg_sim,

            "g_sup": selected_g_sup,
            "g_cat": selected_g_cat,
            "mean_g_sup": mean_g_sup,
            "mean_g_cat": mean_g_cat,
            "fixed_g_sup": fixed_g_sup,
            "fixed_g_cat": fixed_g_cat,

            # A/B/C/D 始终描述 fixed hardest pair，
            # 方便与上一版 post_gate 结果直接比较。
            "region_a_mask": region_a_mask,
            "region_b_mask": region_b_mask,
            "region_c_mask": region_c_mask,
            "region_d_mask": region_d_mask,
            "region_a_count": int(region_a_mask.sum().item()),
            "region_b_count": int(region_b_mask.sum().item()),
            "region_c_count": int(region_c_mask.sum().item()),
            "region_d_count": int(region_d_mask.sum().item()),

            # Reliable Mining 专用诊断。
            "reliable_candidate_total": reliable_candidate_total,
            "avg_reliable_negatives": avg_reliable_negatives,
            "no_reliable_count": no_reliable_count,
            "replacement_mask": replacement_mask,
            "replacement_count": replacement_count,
            "replacement_ratio": (
                replacement_count / max(valid_count, 1)
            ),
        }

    def forward(
        self,
        image_features,
        text_features,
        category_ids,
        caption_support=None,
        image_support=None,
    ):
        self._check_inputs(
            image_features,
            text_features,
            category_ids,
            caption_support=caption_support,
            image_support=image_support,
            need_support=self.reliable_t2i,
        )

        category_ids = category_ids.to(
            image_features.device,
            dtype=torch.long,
        )

        if caption_support is not None:
            caption_support = caption_support.to(
                image_features.device,
                dtype=image_features.dtype,
            )
        if image_support is not None:
            image_support = image_support.to(
                image_features.device,
                dtype=image_features.dtype,
            )

        image_features = F.normalize(
            image_features,
            dim=-1,
        )
        text_features = F.normalize(
            text_features,
            dim=-1,
        )

        sim_i2t = image_features @ text_features.t()
        sim_t2i = sim_i2t.t()

        i2t = self._direction_loss(
            similarities=sim_i2t,
            category_ids=category_ids,
            image_features=image_features,
            text_features=text_features,
            direction="i2t",
        )

        t2i = self._direction_loss(
            similarities=sim_t2i,
            category_ids=category_ids,
            image_features=image_features,
            text_features=text_features,
            direction="t2i",
            caption_support=caption_support,
            image_support=image_support,
        )

        return {
            "loss": 0.5 * (
                t2i["loss"]
                + i2t["loss"]
            ),
            "loss_t2i": t2i["loss"],
            "loss_i2t": i2t["loss"],

            "reliability_mode": self.reliability_mode,

            "t2i_valid_count": t2i["valid_count"],
            "i2t_valid_count": i2t["valid_count"],
            "t2i_reliable_count": t2i["reliable_count"],
            "t2i_active_before_gate_count": t2i[
                "active_before_gate_count"
            ],
            "t2i_active_count": t2i["active_count"],
            "i2t_active_count": i2t["active_count"],

            "t2i_region_a_count": t2i["region_a_count"],
            "t2i_region_b_count": t2i["region_b_count"],
            "t2i_region_c_count": t2i["region_c_count"],
            "t2i_region_d_count": t2i["region_d_count"],

            "t2i_mean_g_sup": t2i["mean_g_sup"],
            "t2i_mean_g_cat": t2i["mean_g_cat"],

            "t2i_mean_pos_sim": t2i["mean_pos_sim"],
            "i2t_mean_pos_sim": i2t["mean_pos_sim"],
            "t2i_mean_hard_neg_sim": t2i[
                "mean_hard_neg_sim"
            ],
            "t2i_mean_fixed_hard_neg_sim": t2i[
                "mean_fixed_hard_neg_sim"
            ],
            "i2t_mean_hard_neg_sim": i2t[
                "mean_hard_neg_sim"
            ],

            "t2i_reliable_candidate_total": t2i[
                "reliable_candidate_total"
            ],
            "t2i_avg_reliable_negatives": t2i[
                "avg_reliable_negatives"
            ],
            "t2i_no_reliable_count": t2i[
                "no_reliable_count"
            ],
            "t2i_replacement_count": t2i[
                "replacement_count"
            ],
            "t2i_replacement_ratio": t2i[
                "replacement_ratio"
            ],

            # debug
            "t2i_hard_neg_index": t2i[
                "hard_neg_index"
            ],
            "t2i_fixed_hard_neg_index": t2i[
                "fixed_hard_neg_index"
            ],
            "i2t_hard_neg_index": i2t[
                "hard_neg_index"
            ],
            "t2i_hard_neg_sim": t2i[
                "hard_neg_sim"
            ],
            "t2i_fixed_hard_neg_sim": t2i[
                "fixed_hard_neg_sim"
            ],
            "i2t_hard_neg_sim": i2t[
                "hard_neg_sim"
            ],
            "t2i_valid_mask": t2i["valid_mask"],
            "t2i_reliable_mask": t2i["reliable_mask"],
            "i2t_valid_mask": i2t["valid_mask"],
            "t2i_active_before_gate_mask": t2i[
                "active_before_gate_mask"
            ],
            "t2i_active_mask": t2i["active_mask"],
            "i2t_active_mask": i2t["active_mask"],
            "t2i_g_sup": t2i["g_sup"],
            "t2i_g_cat": t2i["g_cat"],
            "t2i_fixed_g_sup": t2i["fixed_g_sup"],
            "t2i_fixed_g_cat": t2i["fixed_g_cat"],
            "t2i_region_a_mask": t2i["region_a_mask"],
            "t2i_region_b_mask": t2i["region_b_mask"],
            "t2i_region_c_mask": t2i["region_c_mask"],
            "t2i_region_d_mask": t2i["region_d_mask"],
            "t2i_replacement_mask": t2i[
                "replacement_mask"
            ],
            "similarity_i2t": sim_i2t,
        }
