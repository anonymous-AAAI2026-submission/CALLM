import os
import re
import argparse
import pandas as pd
from prompts.get_prompt import get_rules
from compliance_verifier import ComplianceVerifier, LLMJudgeConfig
from tqdm import tqdm

class_names_map = {
    "approval": "Approval", 
    "signature": "Signature", 
    "c1": "C1 (reporting entity)",
    "c2_str": "C2 (structure)", 
    "c2_op": "C2 (operations)", 
    "c2_supp": "C2 (supply chains)",
    "c3_risk": "C3 (risk description)", 
    "c4_miti": "C4 (risk mitigation)",
    "c4_rem": "C4 (remediation)",
    "c5": "C5 (effectiveness)", 
    "c6": "C6 (consultation)",
}

def get_gpt_results_filename(class_name):
    # get values of class_names_map as class_names
    class_names = list(class_names_map.values())

    if class_name not in class_names:
        raise ValueError(f"Unknown class name: {class_name}")
    
    index = class_names.index(class_name)
    file_name = f"gpt_4o_results/result_for_class_index_test{index}.csv"
    df = pd.read_csv(file_name)
    df['reasoning'] = (
        df['reasoning']
        .str.replace(r'^Reasoning:\s*', '', regex=True)
        .str.replace(r'Final Answer:\s*.*$', '', regex=True)
        .str.strip()
    )
    df = df.rename(columns={"reasoning": "reasonings", "answer": "predictions"})
    return df

def get_cv_results_filename(class_name):
    # get values of class_names_map as class_names
    file_name = f"cvllm_results/predictions_{class_name}.csv"
    df = pd.read_csv(file_name)
    return df

def remove_before_assistant(text):
    """
    Removes everything before and including the line containing 'assistant' (case-insensitive).
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'on the reasoning and the final answer' in line.strip().lower():
            return '\n'.join(lines[i+1:]).strip()
    return text.strip()  # fallback if 'assistant' not found


def local_llmjudge_reward_func(reasonings, predictions, class_of_interest, output_df, save_path, batch_size=4, save_every=5):
    config = LLMJudgeConfig(
        provider_type="local",
        judge_model_name="nuojohnchen/JudgeLRM-7B",
        judge_model_torch_dtype="bfloat16",
        model_cache_dir="", # 👈 Add this
    )
    verifier = ComplianceVerifier(config)
    rules = get_rules(class_of_interest)

    scores, responses = [], []
    total = len(predictions)

    for i in tqdm(range(0, total, batch_size), desc="Batched judgment"):
        end = min(i + batch_size, total)

        # Skip already processed rows
        if "judge_score" in output_df.columns and not pd.isna(output_df.loc[i:end-1, "judge_score"]).any():
            scores.extend(output_df.loc[i:end-1, "judge_score"])
            responses.extend(output_df.loc[i:end-1, "cleaned_judge_reasoning"])
            continue

        batch_preds = predictions[i:end]
        batch_reasonings = reasonings[i:end]

        try:
            # Run evaluation
            batch_scores, batch_outputs = verifier.evaluate_reasoning_batch(
                rules, batch_reasonings, batch_preds
            )

            # Save raw outputs
            output_df.loc[i:end-1, "judge_score"] = batch_scores
            scores.extend(batch_scores)
            responses.extend(batch_outputs)

            # Clean the model outputs (judge_reasoning) and save in new column
            batch_outputs_cleaned = [remove_before_assistant(r) for r in batch_outputs]
            output_df.loc[i:end-1, "cleaned_judge_reasoning"] = batch_outputs_cleaned

        except Exception as e:
            print(f"⚠️ Error in batch {i}–{end}: {e}")
            for idx in range(i, end):
                output_df.loc[idx, "judge_score"] = None
                output_df.loc[idx, "cleaned_judge_reasoning"] = None
            continue  # skip to next batch

        # Save every N batches or at the end
        batch_number = i // batch_size + 1
        if (batch_number % save_every == 0) or (end == total):
            try:
                output_df.to_csv(save_path, index=False)
                print(f"💾 Saved results after batch {batch_number} to: {save_path}")
            except Exception as save_error:
                print(f"❌ Failed to save at batch {batch_number}: {save_error}")

    return scores, responses


def main():
    parser = argparse.ArgumentParser(description="Run ComplianceVerifier on model completions from a CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV file with judge results")
    parser.add_argument("--class", dest="class_of_interest", type=str, default="approval", help="Compliance rule class")
    args = parser.parse_args()

    # === Load input ===
    input_path = args.input_csv
    output_path = args.output_csv

    class_of_interest = class_names_map[args.class_of_interest]
    df = get_gpt_results_filename(class_of_interest) if "gpt_4o_results" in input_path else get_cv_results_filename(args.class_of_interest)

    # === Resume from saved file if available ===
    if os.path.exists(output_path):
        print(f"🔁 Resuming from existing output file: {output_path}")
        df_existing = pd.read_csv(output_path)
        if len(df_existing) == len(df):
            df = df_existing
        else:
            print("⚠️ Existing file has mismatched row count — starting fresh.")

    # === Run LLM Judge with checkpointing ===
    scores, judge_outputs = local_llmjudge_reward_func(
        reasonings=df["reasonings"].tolist(),
        predictions=df["predictions"].tolist(),
        class_of_interest=class_of_interest,
        output_df=df,
        save_path=output_path,
    )

    print(f"✅ Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
