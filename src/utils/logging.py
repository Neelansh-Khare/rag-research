from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(output_dir: Optional[str | Path] = None) -> None:
    """Configure root logging with a console handler (and optional file handler)."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if output_dir is not None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        log_path = p / "run.log"
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

