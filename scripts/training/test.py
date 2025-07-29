import os
from datetime import datetime

from comet_ml import start
from dataset_utils import (
    AIMSDataset,
    ScriptArguments,
    get_AIMS_dataset_sentence,
    get_tokenizer,
)
from evaluate import evaluate_model  # your canvas function
from logging_config import get_logger, setup_loggers
from model_utils import get_checkpoint
from shared_config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, ModelConfig, TrlParser
from utils import test_init


def main():
    parser = TrlParser((ModelConfig, AIMSDataset, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    Config.model_args = model_args
    Config.script_args = script_args
    Config.training_args = training_args
    test_init(script_args.project_name)
    # Setup logging
    setup_loggers()
    logger = get_logger("Testing")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
        model_path = last_checkpoint
    else:
        model_path = model_args.model_name_or_path
        logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Loading datasets...")
    interest = (
        script_args.class_of_interest
        if script_args.class_of_interest
        else ValueError("Class of interest not specified.")
    )
    logger.info(f"Class of interest: {interest}")
    _, _, test_dataset = get_AIMS_dataset_sentence(
        script_args.dataset_id_or_path, tokenizer, interest, script_args.eval_mode
    )

    logger.info("🚀 Starting model evaluation...")
    evaluate_model(model_args, test_dataset, tokenizer, script_args, training_args, accelerator=None)
    logger.info("🚀 Model evaluation completed.")


if __name__ == "__main__":
    main()
