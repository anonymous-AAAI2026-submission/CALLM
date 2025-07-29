from dataclasses import dataclass
from typing import List

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from logging_config import get_logger
from prompts.get_prompt import get_grpo_prompt
from prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SENTENCE
from shared_config import Config
from tqdm import tqdm
from transformers import AutoTokenizer

CONTEXT_SIZE = 50

@dataclass
class AIMSDataset:
    dataset_id_or_path: str = "mila-ai4h/AIMS.au"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    class_of_interest: str = "Approval"
    eval_mode: bool = False
    project_name: str = "Approval-qwen"
    use_judge: bool = False
    downsample_ratio: float = 1


def get_tokenizer(script_args, model_args):
    """Load tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def is_Qwen3(model_path: str) -> bool:
    """Check if the model is Qwen3"""
    return "Qwen-3" in model_path or "Qwen3" in model_path


def generate_r1_prompt_AIMS_sentence(sentence, sentence_with_context, target, tokenizer, class_of_interest):
    r1_prefix = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_SENTENCE,
        },
        {
            "role": "user",
            "content": get_grpo_prompt(class_of_interest)
            + f"""Now classify the following target sentence:

The target sentence to classify is the following:
------------
{sentence}
------------

The same target sentence inside its original block of text:
------------
{sentence_with_context}
------------

### Question:
Is the target sentence compliant? (YES/NO)

# Answer: Lets think step-by-step. In order to provide the correct answer, you need to check if the target sentence matches the requirements. Provide the reasoning and the final answer (YES or NO) in <answer> tags.

Reasoning:

<answer>YES/NO</answer>""",
        },
        # {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    if not is_Qwen3(Config.model_args.model_name_or_path):
        r1_prefix.append({"role": "assistant", "content": "Let me solve this step by step.\n<think>"})
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True, truncation=True
            ),
            "answer": target,
        }
    else:
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix,
                tokenize=False,
                # continue_final_message=True,
                add_generation_prompt=True,
                enable_thinking=True,
                truncation=True,
            ),
            "answer": target,
        }


def add_text_with_context(ds: Dataset, group_column="statement_id", text_column="sentence") -> Dataset:
    # Convert to pandas
    df = ds.to_pandas()

    # Group by the given group column
    grouped = df.groupby(group_column, group_keys=False)

    all_outputs = []
    logger = get_logger("Training")

    logger.info("Adding context to text...")
    for sentence_statement_id, subdf in tqdm(grouped, desc="Adding context", total=len(grouped)):
        # Sort the rows in the correct reading order (by sentence_orig_idxs, presumably ascending):
        subdf_sorted = subdf.sort_values("sentence_orig_idxs").reset_index(drop=True)

        all_words = []
        offsets = []
        current_offset = 0

        for text in subdf_sorted[text_column]:
            words = str(text).split()
            start_offset = current_offset
            end_offset = start_offset + len(words)
            offsets.append((start_offset, end_offset))
            all_words.extend(words)
            current_offset = end_offset

        for i in range(len(subdf_sorted)):
            start_off, end_off = offsets[i]
            context_start = max(0, start_off - CONTEXT_SIZE)
            context_end = min(len(all_words), end_off + CONTEXT_SIZE)
            context_words = all_words[context_start:context_end]
            subdf_sorted.loc[i, "text_with_context"] = " ".join(context_words)

        all_outputs.append(subdf_sorted)

    # Combine all groups
    df_combined = pd.concat(all_outputs, ignore_index=True)

    # Rebuild dataset
    return Dataset.from_pandas(df_combined)


def get_AIMS_dataset_sentence(dataset_id_or_path, tokenizer, class_of_interest, eval_mode=False) -> Dataset:
    """
    Load dataset from Hugging Face Hub (with splits), add context, filter and tokenize.
    """
    dataset = load_dataset(dataset_id_or_path)
    script_args = Config.script_args
    sample_ratio = script_args.downsample_ratio
    logger = get_logger("Training")

    def process_split(split_dataset, do_downsample=False):
        # Step 1: Add context
        split_dataset = add_text_with_context(split_dataset)

        # Step 2: Filter out examples with label == -1
        split_dataset = split_dataset.filter(lambda x: x[class_of_interest] != -1)

        # Step 2.5: Automatically compute downsampling ratio
        if do_downsample:

            def is_positive(x):
                return x[class_of_interest] == 1

            def is_negative(x):
                return x[class_of_interest] == 0

            positives = split_dataset.filter(is_positive)
            negatives = split_dataset.filter(is_negative)

            num_pos = len(positives)
            num_neg = len(negatives)

            if num_pos == 0 or num_neg == 0:
                logger.info(f"[INFO] Skipping downsampling for {class_of_interest}: pos={num_pos}, neg={num_neg}")
            else:
                neg_to_pos_ratio = num_neg / num_pos
                if neg_to_pos_ratio > sample_ratio:
                    # Milder downsampling: 2 negatives per positive
                    logger.info(
                        f"[INFO] Downsampling {class_of_interest}: pos={num_pos}, neg={num_neg} (ratio={neg_to_pos_ratio})"
                    )
                    max_negatives = int(num_pos * sample_ratio)
                    negatives = negatives.shuffle(seed=42).select(range(min(max_negatives, num_neg)))
                    split_dataset = concatenate_datasets([positives, negatives]).shuffle(seed=42)
                else:
                    # No downsampling needed
                    logger.info(
                        f"[INFO] No downsampling needed for {class_of_interest}: pos={num_pos}, neg={num_neg} (ratio={neg_to_pos_ratio})"
                    )

        logger.info(f"[INFO] Dataset size: {len(split_dataset)}")

        # Step 3: Tokenize
        def process_example(x):
            str_label = "YES" if x[class_of_interest] == 1 else "NO"
            return generate_r1_prompt_AIMS_sentence(
                x["sentence"], x["text_with_context"], str_label, tokenizer, class_of_interest
            )

        return split_dataset.map(process_example, remove_columns=split_dataset.column_names)

    if not eval_mode:
        train_dataset = process_split(dataset["train"], do_downsample=True)
        val_dataset = process_split(dataset["validation"], do_downsample=False)
        test_dataset = process_split(dataset["test"], do_downsample=False)
        return train_dataset, val_dataset, test_dataset
    else:
        test_dataset = process_split(dataset["test"], do_downsample=False)
        return None, None, test_dataset
