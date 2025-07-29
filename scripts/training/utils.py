import logging
import os
from datetime import datetime

import psutil
import torch
from comet_ml import start
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)


def log_system_stats(logger: logging.Logger, device_idx: int = None, log_gpu: bool = True, log_memory: bool = True):
    """
    Logging System Resource Usage

    Args.
        logger: Logger
        device_idx: Specify GPU device ID (None means all GPUs)
        log_gpu: whether to log GPU information or not
        log_memory: Whether to log memory information
    """
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")

        # GPU Monitoring
        if log_gpu and torch.cuda.is_available():
            nvmlInit()
            devices = [device_idx] if device_idx is not None else range(nvmlDeviceGetCount())

            for i in devices:
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                utilization = nvmlDeviceGetUtilizationRates(handle)

                logger.info(
                    f"[{timestamp}] GPU {i} Status | "
                    f"Mem: Alloc={torch.cuda.memory_allocated(i)/1024**3:.2f}GB/"
                    f"Cache={torch.cuda.memory_reserved(i)/1024**3:.2f}GB/"
                    f"Free={info.free/1024**3:.2f}GB | "
                    f"Util: GPU={utilization.gpu}% Mem={utilization.memory}%"
                )

        # process memory
        if log_memory:
            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            logger.info(
                f"[{timestamp}] Process Memory | " f"RSS={mem.rss/1024**2:.2f}MB " f"VMS={mem.vms/1024**2:.2f}MB"
            )

    except Exception as e:
        logger.warning(f"System monitoring failed: {str(e)}")


def train_init(project_name):
    """Initialize the training process."""
    # Comet login
    experiment = start(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", ""),
        workspace=os.getenv("COMET_WORKSPACE", ""),
    )
    experiment.set_name(project_name)

    os.environ["COMET_LOG_ASSETS"] = "True"

    scratch_dir = os.environ.get("SCRATCH")
    success_dir = os.path.join(scratch_dir, "completion_samples")
    os.makedirs(success_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(success_dir, f"success_completion_samples_{timestamp}.log")

    error_dir = os.path.join(scratch_dir, "error_samples")
    os.makedirs(error_dir, exist_ok=True)
    error_file = os.path.join(error_dir, f"error_completion_samples_{timestamp}.log")

    os.environ["ERROR_LOG_FILE"] = error_file
    os.environ["COMPLETION_LOG_FILE"] = log_file


def test_init(project_name):
    # Comet login
    experiment = start(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", ""),
        workspace=os.getenv("COMET_WORKSPACE", ""),
    )
    experiment.set_name(project_name)

    os.environ["COMET_LOG_ASSETS"] = "True"

    scratch_dir = os.environ.get("SCRATCH")
    target_dir = os.path.join(scratch_dir, "evaluation_results")
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(target_dir, f"evaluation_results_{timestamp}.txt")
    os.environ["EVALUATION_LOG_FILE"] = log_file
