import logging
import os
from datetime import datetime
from types import MethodType
from typing import List, Optional

from logging_config import get_logger
from reward_functions import (
    contains_match_reward_func,
    correctness_reward_func,
    equation_reward_func,
    format_reward_func,
    llmjudge_reward_func,
    reason_penalty_func,
    sentence_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config
from vllm import LLM, SamplingParams


class PrintStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        reward_logger = get_logger("Rewards")
        reward_logger.info(f"Step {state.global_step}")


def attach_defaults(client, **defaults):
    orig_generate = client.generate

    def generate_with_defaults(*args, **kwargs):
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        return orig_generate(*args, **kwargs)

    client.generate = generate_with_defaults

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig,
    training_args: GRPOConfig,
    logger: logging.Logger,
    train_dataset,
    test_dataset,
    tokenizer,
    reward_funcs: Optional[List] = None,
):
    #########################
    # Log parameters
    #########################
    logger = get_logger("GRPO")
    logger.info("Initializing GRPO trainer")
    if not reward_funcs:
        logger.warning("No reward functions provided, using defaults")
        from reward_functions import default_reward_funcs

        reward_funcs = default_reward_funcs()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    #########################
    # Instantiate GRPO trainer
    #########################
    resume_from_checkpoint = False
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
        model_path = last_checkpoint
        resume_from_checkpoint = True
    else:
        model_path = model_args.model_name_or_path
        logger.info(f"Loading model from {model_path}")

    try:
        trainer = GRPOTrainer(
            model=model_path,
            # reward_funcs=[format_reward_func, equation_reward_func],
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=get_peft_config(model_args),
        )
        # logger.debug("Trainer config: %s", trainer.config)
    except Exception as e:
        logger.critical("Trainer initialization failed", exc_info=True)
        raise

    ###############
    # Training loop
    ###############
    trainer.add_callback(PrintStepCallback())
    attach_defaults(
        trainer.vllm_client,
        # seed=trainer.args.seed,
        temperature=trainer.args.temperature,
        top_p=trainer.args.top_p,
        top_k=trainer.args.top_k,
    )
    # trainer.sampling_params = SamplingParams(
    #     seed = training_args.seed,
    #     temperature = training_args.temperature,
    #     top_k = training_args.top_k,
    #     top_p = training_args.top_p,
    #     max_tokens = training_args.max_completion_length,
    #     n = training_args.num_generations,
    # )
    # Train the model
    # logger.info(f'Using Trainer: {type(trainer)}')
    # logger.info(f'LLM engine seed: {trainer.llm.config.seed}')
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training completed. Metrics: %s", train_result.metrics)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed with error: %s", str(e), exc_info=True)
        raise
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    # if int(os.getenv("LOCAL_RANK", "0")) == 0:
    #     logger.info("Saving model...")
    #     trainer.save_model(training_args.output_dir)
    #     logger.info(f"Model saved to {training_args.output_dir}")
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["grpo", "compliance_verifier"]})
    # push to hub if needed
    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub()

    logger.info("*** Training complete! ***")
