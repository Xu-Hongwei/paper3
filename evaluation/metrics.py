import numpy as np


def _recall_at_k(ranks, k):
    """
    ranks use 0-based indexing.

    rank = 0 -> Top-1
    rank = 4 -> Top-5
    rank = 9 -> Top-10
    """

    return 100.0 * np.mean(
        ranks < k
    )


def compute_i2t_ranks(
    scores_i2t,
    img2txt,
):
    """
    Image-to-Text ranking.

    Args:
        scores_i2t:
            numpy array [N_image, N_text]

        img2txt:
            mapping:
                image_id -> list of GT text ids

    Returns:
        ranks:
            numpy array [N_image]

    For each image, there may be multiple correct captions.
    We use the best rank among all GT captions.
    """

    num_images = scores_i2t.shape[0]

    ranks = np.zeros(
        num_images,
        dtype=np.int64,
    )

    for image_id in range(
        num_images
    ):

        scores = scores_i2t[
            image_id
        ]

        # Descending similarity ranking
        sorted_text_ids = np.argsort(
            scores
        )[::-1]

        # ------------------------------------------
        # Build inverse ranking:
        #
        # text_id -> rank
        #
        # Example:
        # sorted_text_ids = [7, 2, 5, 0]
        #
        # rank_lookup[7] = 0
        # rank_lookup[2] = 1
        # rank_lookup[5] = 2
        # rank_lookup[0] = 3
        # ------------------------------------------

        rank_lookup = np.empty_like(
            sorted_text_ids
        )

        rank_lookup[
            sorted_text_ids
        ] = np.arange(
            len(sorted_text_ids)
        )

        gt_text_ids = np.asarray(
            img2txt[image_id],
            dtype=np.int64,
        )

        # ------------------------------------------
        # One image has multiple correct captions.
        #
        # Take the BEST rank.
        # ------------------------------------------

        gt_ranks = rank_lookup[
            gt_text_ids
        ]

        ranks[image_id] = (
            gt_ranks.min()
        )

    return ranks


def compute_t2i_ranks(
    scores_i2t,
    txt2img,
):
    """
    Text-to-Image ranking.

    Args:
        scores_i2t:
            numpy array [N_image, N_text]

        txt2img:
            mapping:
                text_id -> GT image id

    Returns:
        ranks:
            numpy array [N_text]
    """

    # [N_image, N_text]
    #
    # ->
    #
    # [N_text, N_image]

    scores_t2i = scores_i2t.T

    num_texts = scores_t2i.shape[0]

    ranks = np.zeros(
        num_texts,
        dtype=np.int64,
    )

    for text_id in range(
        num_texts
    ):

        scores = scores_t2i[
            text_id
        ]

        sorted_image_ids = np.argsort(
            scores
        )[::-1]

        rank_lookup = np.empty_like(
            sorted_image_ids
        )

        rank_lookup[
            sorted_image_ids
        ] = np.arange(
            len(sorted_image_ids)
        )

        gt_image_id = int(
            txt2img[text_id]
        )

        ranks[text_id] = (
            rank_lookup[
                gt_image_id
            ]
        )

    return ranks


def compute_retrieval_metrics(
    scores_i2t,
    img2txt,
    txt2img,
):
    """
    Compute standard RSITR retrieval metrics.

    Returns:
        dictionary containing:

        I2T:
            R@1
            R@5
            R@10

        T2I:
            R@1
            R@5
            R@10

        mR
    """

    # ==========================================
    # Basic checks
    # ==========================================

    if scores_i2t.ndim != 2:
        raise ValueError(
            "scores_i2t must be a 2-D matrix."
        )

    num_images, num_texts = (
        scores_i2t.shape
    )

    if len(img2txt) != num_images:
        raise ValueError(
            "img2txt size does not match "
            "number of images."
        )

    if len(txt2img) != num_texts:
        raise ValueError(
            "txt2img size does not match "
            "number of captions."
        )

    # ==========================================
    # Image -> Text
    # ==========================================

    i2t_ranks = compute_i2t_ranks(
        scores_i2t,
        img2txt,
    )

    i2t_r1 = _recall_at_k(
        i2t_ranks,
        1,
    )

    i2t_r5 = _recall_at_k(
        i2t_ranks,
        5,
    )

    i2t_r10 = _recall_at_k(
        i2t_ranks,
        10,
    )

    # ==========================================
    # Text -> Image
    # ==========================================

    t2i_ranks = compute_t2i_ranks(
        scores_i2t,
        txt2img,
    )

    t2i_r1 = _recall_at_k(
        t2i_ranks,
        1,
    )

    t2i_r5 = _recall_at_k(
        t2i_ranks,
        5,
    )

    t2i_r10 = _recall_at_k(
        t2i_ranks,
        10,
    )

    # ==========================================
    # Mean Recall
    # ==========================================

    i2t_mean = (
        i2t_r1
        + i2t_r5
        + i2t_r10
    ) / 3.0

    t2i_mean = (
        t2i_r1
        + t2i_r5
        + t2i_r10
    ) / 3.0

    mean_recall = (
        i2t_mean
        +
        t2i_mean
    ) / 2.0

    # ==========================================
    # Optional rank statistics
    #
    # +1 converts zero-based ranks
    # to human-readable rank numbers.
    # ==========================================

    i2t_medr = (
        np.median(i2t_ranks)
        + 1
    )

    t2i_medr = (
        np.median(t2i_ranks)
        + 1
    )

    i2t_meanr = (
        np.mean(i2t_ranks)
        + 1
    )

    t2i_meanr = (
        np.mean(t2i_ranks)
        + 1
    )

    return {
        "i2t_r1": float(i2t_r1),
        "i2t_r5": float(i2t_r5),
        "i2t_r10": float(i2t_r10),

        "t2i_r1": float(t2i_r1),
        "t2i_r5": float(t2i_r5),
        "t2i_r10": float(t2i_r10),

        "i2t_mean": float(i2t_mean),
        "t2i_mean": float(t2i_mean),

        "mR": float(mean_recall),

        "i2t_medr": float(i2t_medr),
        "t2i_medr": float(t2i_medr),

        "i2t_meanr": float(i2t_meanr),
        "t2i_meanr": float(t2i_meanr),
    }