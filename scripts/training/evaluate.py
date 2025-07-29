import datetime
import json
import os

import numpy as np
import pandas as pd
from logging_config import get_logger, setup_loggers
from model_utils import get_checkpoint
from reward_functions import extract_xml_answer, extract_xml_reasoning, extract_target_sentence
from shared_config import Config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from vllm import LLM, SamplingParams


def evaluate_model(model_args, test_dataset, tokenizer, script_args, training_args, accelerator):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = get_logger("Evaluation")

    logger.info("Looking for last checkpoint...")
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming from {last_checkpoint}.")
        model_path = last_checkpoint
    else:
        model_path = model_args.model_name_or_path
        logger.info(f"Loading model from {model_path}")

    logger.info("Loading vLLM model")
    llm = LLM(model=model_path, seed=training_args.seed)

    all_preds = []
    all_labels = []
    all_reasonings = []
    all_target_sentences = []
    logger.info("Starting batched evaluation loop with vLLM")

    batch_size = 8  # Tune this based on GPU capacity

    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Batch size: {batch_size}")

    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch = test_dataset[i : i + batch_size]
        prompts = batch["prompt"]
        labels = batch["answer"]

        sampling_params = SamplingParams(
            temperature=training_args.temperature,
            seed=training_args.seed,
            top_p=training_args.top_p,
            top_k=training_args.top_k,
            max_tokens=training_args.max_completion_length,
        )

        outputs = llm.generate(prompts, sampling_params)
        logger.info(f"Evaluation arguments: {sampling_params}")

        for prompt, output, label in zip(prompts, outputs, labels):
            answers = [extract_xml_answer(o.text) for o in output.outputs]
            logger.info("-" * 30)
            logger.info(f"Prompt: {prompt}")
            for idx, answer in enumerate(answers):
                # Log the unextracted output first, then the extracted one
                raw_text = output.outputs[idx].text
                logger.info(f"Raw Output {idx+1}: {raw_text}")
                logger.info(f"Extracted Completion {idx+1}: {answer}")
                
                if idx == 0:
                    try:
                        reasoning = extract_xml_reasoning(raw_text)
                        if not reasoning:
                            reasoning = raw_text
                    except Exception:
                        reasoning = raw_text
                    extracted_sentence = extract_target_sentence(prompt)
                    logger.info(f"Extracted Target Sentence: {extracted_sentence}")
                    all_reasonings.append(reasoning)
                    all_target_sentences.append(extracted_sentence)

            logger.info(f"Ground Truth: {label}")
            all_preds.append(answers[0])  # Use first as default for scoring
            all_labels.append(label)

    valid_labels = {"YES", "NO"}

    # Convert to binary if needed (assuming YES/NO labels)
    # y_true = [label if label in valid_labels else 'NO' for label in all_labels]
    # y_pred = [pred if pred in valid_labels else 'NO' for pred in all_preds]

    # Filter out and remove invalid predictions
    df = pd.DataFrame({"target_sentences": all_target_sentences, "labels": all_labels, "predictions": all_preds, "reasonings": all_reasonings})
    original_length = len(df)
    df = df[df["predictions"].isin(valid_labels) & df["labels"].isin(valid_labels)]
    filtered_length = len(df)
    if original_length != filtered_length:
        logger.warning(f"Filtered out {original_length - filtered_length} invalid predictions.")
    y_true = df["labels"].tolist()
    y_pred = df["predictions"].tolist()
    all_reasonings = df["reasonings"].tolist()
    all_target_sentences = df["target_sentences"].tolist()
    logger.info(f"Filtered dataset size: {filtered_length}")

    # Calculate all metrics with zero_division handling
    try:
        precision = precision_score(y_true, y_pred, pos_label="YES", zero_division=np.nan)
        recall = recall_score(y_true, y_pred, pos_label="YES", zero_division=np.nan)
        f1 = f1_score(y_true, y_pred, pos_label="YES", zero_division=np.nan)
        accuracy = accuracy_score(y_true, y_pred)
    except Exception as e:
        logger.error(f"Metric calculation failed: {str(e)}")
        precision = recall = f1 = accuracy = np.nan

    # Create comprehensive results dictionary
    results = {
        "model_name": model_path,
        "class_of_interest": script_args.class_of_interest,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "positive_class_ratio": sum(1 for x in y_true if x == "YES") / len(y_true),
        "invalid_predictions": sum(1 for x in all_preds if x not in valid_labels),
        "total_samples": len(y_true),
    }

    # Log results
    logger.info("\n" + "=" * 40)
    logger.info("Evaluation Results:")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall:    {results['recall']:.4f}")
    logger.info(f"F1 Score:  {results['f1_score']:.4f}")
    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"Positive Class Ratio: {results['positive_class_ratio']:.2%}")
    logger.info(f"Invalid Predictions: {results['invalid_predictions']}")
    logger.info("=" * 40 + "\n")

    # Save predictions and metrics
    results_dir = os.path.join(training_args.output_dir, f"results_{Config.script_args.project_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions CSV
    df = pd.DataFrame(
        {
            "target_sentences": all_target_sentences,
            "labels": y_true,
            "predictions": y_pred,
            "reasonings": all_reasonings,
        }
    )
    df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save human-readable report
    with open(os.path.join(results_dir, "report.txt"), "w") as f:
        f.write("Evaluation Report\n")
        f.write("=" * 40 + "\n")
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"{k:<20}: {v:.4f}\n")
            else:
                f.write(f"{k:<20}: {v}\n")

    logger.info(f"Results saved to {results_dir}")

    return results
