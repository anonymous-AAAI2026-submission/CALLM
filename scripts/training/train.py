import logging
import comet_ml
import os
import random
from datetime import datetime

import numpy
import numpy as np
import torch
from accelerate.utils import set_seed
from comet_ml import start
from dataset_utils import (
    AIMSDataset,
    get_AIMS_dataset_sentence,
    get_tokenizer,
)
from llm_judge import LLMJudgeConfig
from logging_config import get_logger, setup_loggers
from model_utils import get_checkpoint, grpo_function
from reward_functions import default_reward_funcs, llmjudge_reward_func
from shared_config import Config
from trl import GRPOConfig, ModelConfig, TrlParser
from utils import train_init


def main():
    """Train model using GRPO on multiple GPUs.
    Example usage on 2 GPUs:
    accelerate launch --num_processes 1 --config_file configs/deepspeed_zero3.yaml train.py --config configs/grpo.yaml

    num_processes: Number of GPUs to use. If num_processes=1, then the model will be trained on a single GPU.
    config_file: Path to the configuration file.
    Set the num_processes to the number of GPUs you have - 1 as the last one will be used with vLLM for Generation.
    If you are using more GPUS you need to change the vllm_device in the config file to last index GPU.
    E.g If you have 8 GPUs you need to set vllm_device=7 and your num_processes to 7.

    """

    # parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    parser = TrlParser((ModelConfig, AIMSDataset, GRPOConfig, LLMJudgeConfig))
    model_args, script_args, training_args, judge_config_args = parser.parse_args_and_config()
    Config.model_args = model_args
    Config.script_args = script_args
    Config.training_args = training_args
    Config.judge_config_args = judge_config_args
    train_init(script_args.project_name)
    # Setup logging
    setup_loggers()
    logger = get_logger("Training")
    logger.info("Initializing experiment...")
    logger.debug(f"Parsed arguments: {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if (
        (last_checkpoint is not None)
        and (training_args.resume_from_checkpoint is None)
        and (not script_args.tokenizer_name_or_path)
    ):
        logger.info(f"Checkpoint detected, resuming training tokenizer at {last_checkpoint}.")
        script_args.tokenizer_name_or_path = last_checkpoint

    logger.info(
        f"Loading tokenizer from {(script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path)}"
    )
    tokenizer = get_tokenizer(script_args, model_args)
    logger.info("Loading datasets...")
    interest = (
        script_args.class_of_interest
        if script_args.class_of_interest
        else ValueError("Class of interest not specified.")
    )
    logger.info(f"Class of interest: {interest}")
    os.environ["CLASS_OF_INTEREST"] = interest
    train_dataset, valid_dataset, test_dataset = get_AIMS_dataset_sentence(
        script_args.dataset_id_or_path, tokenizer, interest, script_args.eval_mode
    )

    # Run the main training loop
    logger.info(f"🚀 Starting training on rank {os.getenv('LOCAL_RANK', '0')}")

    SEED = training_args.seed  # =42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    reward_funcs = default_reward_funcs()
    if script_args.use_judge:
        reward_funcs.append(llmjudge_reward_func)

    grpo_function(model_args, training_args, logger, train_dataset, test_dataset, tokenizer, reward_funcs)
    logger.info("🚀 Training completed successfully.")


if __name__ == "__main__":
    main()
