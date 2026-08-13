import re


def pre_caption(caption: str, max_words: int = 30) -> str:
    """
    Normalize and truncate a caption.

    Args:
        caption: Raw caption string.
        max_words: Maximum number of words to keep.

    Returns:
        Cleaned caption string.
    """
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        "",
        caption.lower(),
    )

    caption = (
        caption
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.strip()

    words = caption.split(" ")

    if len(words) > max_words:
        caption = " ".join(words[:max_words])

    if not caption:
        raise ValueError("pre_caption yields invalid text")

    return caption
