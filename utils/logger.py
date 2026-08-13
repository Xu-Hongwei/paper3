import json
import sys
from datetime import datetime
from pathlib import Path


class TeeStream:
    """
    Duplicate stdout/stderr to both:

        1. terminal
        2. log file

    This allows all existing print(...) statements
    to be automatically written into the log file.
    """

    def __init__(
        self,
        terminal,
        log_file,
    ):
        self.terminal = terminal
        self.log_file = log_file

    def write(
        self,
        message,
    ):
        self.terminal.write(
            message
        )

        self.log_file.write(
            message
        )

        return len(message)

    def flush(
        self,
    ):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(
        self,
    ):
        return self.terminal.isatty()

    @property
    def encoding(
        self,
    ):
        return getattr(
            self.terminal,
            "encoding",
            "utf-8",
        )


def setup_logger(
    output_dir,
    prefix="train",
):
    """
    Create a timestamped log file and redirect
    stdout / stderr to both terminal and log file.

    Example:

        outputs/clip_rsicd_fullft_30ep/
            train_20260813_163900.log

    Returns:
        log_path
        log_file
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        output_dir
        / f"{prefix}_{timestamp}.log"
    )

    log_file = open(
        log_path,
        mode="a",
        encoding="utf-8",
        buffering=1,
    )

    # Save original terminal streams.
    stdout_terminal = sys.stdout
    stderr_terminal = sys.stderr

    sys.stdout = TeeStream(
        stdout_terminal,
        log_file,
    )

    sys.stderr = TeeStream(
        stderr_terminal,
        log_file,
    )

    return (
        log_path,
        log_file,
    )


def append_jsonl(
    path,
    data,
):
    """
    Append one dictionary as a JSON line.

    Each epoch corresponds to one line.

    Example:

        {"epoch": 1, "train_loss": 0.9, "val_mR": 30.2}
        {"epoch": 2, "train_loss": 0.6, "val_mR": 32.4}
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        mode="a",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
        )

        f.write("\n")