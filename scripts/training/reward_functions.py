import ast
import os
import random
import re
from collections import Counter
from datetime import datetime
from typing import List

import requests
import yaml
from logging_config import get_logger
from prompts.get_prompt import get_rules
from shared_config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import log_system_stats

IDX_LOG = 0
# Global judge cache
LLM_JUDGE_CACHE = {}


def extract_xml_answer(text: str) -> str:
    """Extracts content between <answer> tags"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_xml_reasoning(text: str) -> str:
    """Extracts content between <think> tags"""
    answer = text.split("<think>")[-1]
    answer = answer.split("</think>")[0]
    return answer.strip()

def extract_target_sentence(text: str) -> str:
    """
    Extracts the target sentence using regex.
    """
    match = re.search(
        r"The target sentence to classify is the following:\s*-{12,}\s*([\s\S]+?)\s*-{12,}",
        text
    )
    if match:
        return match.group(1).strip()
    return ""


def count_xml(text) -> float:
    """Count the number of XML tags in the text

    Args:
        text (str): The text to be checked

    Returns:
        float: The count of XML tags
    """
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>") == 1:
        count += 0.125
        # count -= len(text.split("\n<answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("</answer>")[-1]) - 1) * 0.001

    # Make sure count is between 0 and 1
    count = max(0.0, min(count, 1.0))
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Calculate the reward based on the number of XML tags in the completion

    Args:
        completions (str): Model completions

    Returns:
        list[float]: List of rewards for each completion
    """
    func_logger = get_logger("Rewards")
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion
            rewards.append(count_xml(completion))
        except Exception:
            rewards.append(0.0)

    func_logger.info("-" * 35)
    func_logger.info(f"XML Rewards: {rewards}")
    return rewards


def format_reward_func(completions, **kwargs):
    """Calculate the reward based on the format of the completion.
        Format: <think>...</think><answer>...</answer>

    Args:
        completions (str): Model completions
        target (str): Ground truth completions
    """
    func_logger = get_logger("Rewards")
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion
            # if random.random() < 0.1:
            #     os.makedirs("completion_samples", exist_ok=True)
            #     with open("completion_samples/completion_samples.txt", "a") as f:
            #         f.write("\n\n==============\n")
            #         f.write(completion)
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            rewards.append(1.0 if match and len(match.groups()) == 2 else 0.0)
        except Exception:
            rewards.append(0.0)

    func_logger.info("-" * 35)
    func_logger.info(f"Format Rewards: {rewards}")
    return rewards



def llmjudge_reward_func(completions, answer, **kwargs):
    """
    Calculate reward using API-based LLM judge server.
    """
    func_logger = get_logger("Rewards")
    # Fallback logger if not provided
    import logging

    if func_logger is None:
        func_logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    api_url = Config.judge_config_args.api_url
    criteria = Config.script_args.class_of_interest

    if not api_url:
        raise ValueError("`api_url` must be set in llmjudge_config.yaml for API-based judge.")

    rules = get_rules(criteria)

    # Prepare payload for /evaluate_batch
    items = []
    for completion in completions:
        full_text = "<think>" + completion.strip()
        extracted_answer = extract_xml_answer(full_text) or answer
        extracted_reasoning = extract_xml_reasoning(full_text) or ""

        items.append(
            {
                "rules": rules,
                "reasoning": extracted_reasoning,
                "final_answer": extracted_answer,
            }
        )

    payload = {"items": items}

    try:
        func_logger.info(f"[LLMJudge API] Calling {api_url}/evaluate_batch for {len(items)} items")
        response = requests.post(f"{api_url}/evaluate_batch", json=payload)
        response.raise_for_status()

        results = response.json()
        rewards = [r.get("score", 0.0) for r in results]
        response_texts = [r.get("response", "") for r in results]

    except Exception as e:
        func_logger.exception(f"[LLMJudge API] Error during judging: {e}")
        rewards = [0.0] * len(completions)
        response_texts = ["API error"] * len(completions)

    func_logger.info(f"[LLMJudge API] Rewards: {rewards}")
    func_logger.info(f"[LLMJudge API] Responses: {response_texts[0]}")

    return rewards


CLASS_NORM_WEIGHTS = {
    "Approval": {0: 0.408451, 1: 4.047088},
    "Signature": {0: 0.408354, 1: 4.079409},
    "C2 (structure)": {0: 0.426354, 1: 2.240249},
    "C2 (operations)": {0: 0.438174, 1: 1.873727},
    "C2 (supply chains)": {0: 0.429971, 1: 2.106346},
    "C3 (risk description)": {0: 0.430092, 1: 2.102257},
    "C4 (risk mitigation)": {0: 0.485718, 1: 1.251208},
    "C4 (remediation)": {0: 0.407926, 1: 4.236070},
    "C5 (effectiveness)": {0: 0.424686, 1: 2.311366},
}


def correctness_reward_func_sentence_classic(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculate the correctness reward for a list of completions.

    Args:
        prompts (str): Prompt
        completions (str): Model completions
        answer (str): Ground truth answer

    Returns:
        list[float]: _description_
    """
    func_logger = get_logger("Rewards")

    rewards = []
    extracted_responses = []

    for idx, completion in enumerate(completions):
        # Processing completion
        completion = "<think>" + completion
        extracted_response = extract_xml_answer(completion)

        if idx == IDX_LOG:
            func_logger.info("=" * 50)
            # func_logger.info(f"Step {step}/{max_steps} ({step/max_steps:.1%}) | Processing completion {idx}/{len(completions)}")
            log_system_stats(logger=func_logger, device_idx=None, log_gpu=True, log_memory=True)
            func_logger.info(f"Prompt len: {len(prompts)}")
            func_logger.info(f"Question: {prompts[idx]}")
            func_logger.info(f"Ground Truth Answer: {answer[idx]}")
            func_logger.info(f"Full Response: {completion}")
            func_logger.info(f"Parsed Answer: {extracted_response}")

        pred = extracted_response.strip().upper()
        true = answer[idx].strip().upper()
        if idx == IDX_LOG:
            func_logger.info(f"Pred: {pred} | True: {true}")

        if pred == true:  # TP
            reward = 1.0
        else:
            reward = 0.0

        if reward > 0 and random.random() < 0.30:  # 30% chance to write fully successful samples into a file
            log_file = os.environ.get("COMPLETION_LOG_FILE")
            with open(log_file, "a") as f:
                f.write("\n\n==============\n")
                # print timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(completion)
                f.write(f"\nPred: {pred} | True: {true}")

        if reward == 0.0 and random.random() < 0.50:
            err_log_file = os.environ.get("ERROR_LOG_FILE")
            with open(err_log_file, "a") as f:
                f.write("\n\n==============\n")
                # print timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Prompt: {prompts[idx]}\n")
                f.write(f"Completion: {completion}\n")
                f.write(f"\nTrue: {true} | Pred: {pred}\n")

        rewards.append(reward)
        extracted_responses.append(pred)

    func_logger.info(f"Ground Truth Answer: {answer}")
    func_logger.info(f"Extracted responses: {extracted_responses}")
    func_logger.info(f"Rewards: {rewards}")

    return rewards


def correctness_reward_func_sentence(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculate the correctness reward for a list of completions.

    Args:
        prompts (str): Prompt
        completions (str): Model completions
        answer (str): Ground truth answer

    Returns:
        list[float]: _description_
    """
    func_logger = get_logger("Rewards")

    rewards = []
    extracted_responses = []

    class_of_interest = Config.script_args.class_of_interest
    if class_of_interest not in CLASS_NORM_WEIGHTS:
        raise ValueError(f"Unknown class_of_interest: {class_of_interest}")

    w0 = CLASS_NORM_WEIGHTS[class_of_interest][0]
    w1 = CLASS_NORM_WEIGHTS[class_of_interest][1]
    tp_base = w1 / (w0 + w1)
    tn_base = w0 / (w0 + w1)

    scale = 1.0 / tp_base
    tp_score = 1.0
    tn_score = tn_base * scale

    # Clip for safety
    tp_score = min(max(tp_score, 0.0), 1.0)
    tn_score = min(max(tn_score, 0.0), 1.0)

    for idx, completion in enumerate(completions):
        # Processing completion
        completion = "<think>" + completion
        extracted_response = extract_xml_answer(completion)

        if idx == IDX_LOG:
            func_logger.info("=" * 50)
            # func_logger.info(f"Step {step}/{max_steps} ({step/max_steps:.1%}) | Processing completion {idx}/{len(completions)}")
            log_system_stats(logger=func_logger, device_idx=None, log_gpu=True, log_memory=True)
            func_logger.info(f"Prompt len: {len(prompts)}")
            func_logger.info(f"Question: {prompts[idx]}")
            func_logger.info(f"Ground Truth Answer: {answer[idx]}")
            func_logger.info(f"Full Response: {completion}")
            func_logger.info(f"Parsed Answer: {extracted_response}")

        pred = extracted_response.strip().upper()
        true = answer[idx].strip().upper()
        if idx == IDX_LOG:
            func_logger.info(f"Pred: {pred} | True: {true}")

        if pred == "YES" and true == "YES":  # TP
            reward = tp_score
        elif pred == "NO" and true == "NO":  # TN
            reward = tn_score
        else:
            reward = 0.0

        if reward > 0 and random.random() < 0.30:  # 30% chance to write fully successful samples into a file
            log_file = os.environ.get("COMPLETION_LOG_FILE")
            with open(log_file, "a") as f:
                f.write("\n\n==============\n")
                # print timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(completion)
                f.write(f"\nPred: {pred} | True: {true}")

        if reward == 0.0 and random.random() < 0.50:
            err_log_file = os.environ.get("ERROR_LOG_FILE")
            with open(err_log_file, "a") as f:
                f.write("\n\n==============\n")
                # print timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Prompt: {prompts[idx]}\n")
                f.write(f"Completion: {completion}\n")
                f.write(f"\nTrue: {true} | Pred: {pred}\n")

        rewards.append(reward)
        extracted_responses.append(pred)

    func_logger.info(f"Ground Truth Answer: {answer}")
    func_logger.info(f"Extracted responses: {extracted_responses}")
    func_logger.info(f"Rewards: {rewards}")

    return rewards



def default_reward_funcs():
    """Returns a list of default reward functions"""
    return [
        # correctness_reward_func_sentence,
        correctness_reward_func_sentence_classic,
        format_reward_func,
        xmlcount_reward_func,
    ]
