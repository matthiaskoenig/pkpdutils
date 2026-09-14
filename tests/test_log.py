import io
import logging

from rich.console import Console
from rich.logging import RichHandler

from pkpdutils import log


def test_enable_rich_logging_adds_one_handler() -> None:
    console = Console(file=io.StringIO(), width=80)
    logger = log.enable_rich_logging(level=logging.DEBUG, console=console)
    log.enable_rich_logging(level=logging.DEBUG, console=console)
    handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
    assert logger.name == "pkpdutils"
    assert len(handlers) == 1
    assert logger.level == logging.DEBUG
    for handler in handlers:
        logger.removeHandler(handler)
