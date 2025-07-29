import logging
import os
from datetime import datetime
from pathlib import Path

from shared_config import Config

# --------------------------
# Configurable constants
# --------------------------
LOG_LEVEL = logging.INFO
LOG_DIR = "logs"
ENABLE_RANK_LOGS = False  # Whether to keep a work process independent log


# --------------------------
# Log system initialisation
# --------------------------
def get_timestamp():
    """Return normalised timestamp (YYYYMMDD_HHMMSS)"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_loggers():
    """Initialising a multi-process secure logging system"""
    if not hasattr(Config, "script_args") or Config.script_args is None:
        raise ValueError("Config.script_args must be set before calling setup_loggers()")

    project_name = Config.script_args.project_name
    eval_mode = Config.script_args.eval_mode
    timestamp = get_timestamp()
    rank = os.getenv("LOCAL_RANK", "0")

    # Set log directory, prefer SCRATCH if available
    scratch_dir = os.environ.get("SCRATCH")
    base_log_dir = Path(scratch_dir) / LOG_DIR if scratch_dir else Path(LOG_DIR)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger setup
    root_logger = logging.getLogger(project_name)
    if root_logger.handlers:
        return  # Logger already configured
    root_logger.setLevel(LOG_LEVEL)
    root_logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] [Rank:%(rank)s] %(message)s", defaults={"rank": rank}
    )

    # --------------------------
    # Console Handler (all processes)
    # --------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --------------------------
    # File Handler (master process only)
    # --------------------------
    if rank == "0":
        # Main log file
        main_path = base_log_dir / f"{project_name}_main_{timestamp}.log"
        main_handler = logging.FileHandler(main_path)
        main_handler.setFormatter(formatter)
        root_logger.addHandler(main_handler)

    # Module-specific logger: Rewards or Evaluation
    if eval_mode:
        _setup_module_logger(
            f"{project_name}.Evaluation", base_log_dir / f"{project_name}_evaluation_{timestamp}.log", formatter
        )
    else:
        _setup_module_logger(
            f"{project_name}.Rewards", base_log_dir / f"{project_name}_rewards_{timestamp}.log", formatter
        )

    # --------------------------
    # Work process log (optional saving)
    # --------------------------
    if ENABLE_RANK_LOGS and rank != "0":
        rank_handler = logging.FileHandler(f"{LOG_DIR}/{project_name}_rank{rank}_{timestamp}.log")
        rank_handler.setFormatter(formatter)
        root_logger.addHandler(rank_handler)


def _setup_module_logger(name: str, filepath: Path, formatter: logging.Formatter):
    """Helper function to set up a module-specific standalone logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(filepath)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger for a specific module."""
    if not hasattr(Config, "script_args") or Config.script_args is None:
        raise ValueError("Config.script_args must be set before calling get_logger()")
    return logging.getLogger(f"{Config.script_args.project_name}.{module_name}")
