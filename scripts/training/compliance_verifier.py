import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from prompts.compliance_verifier_prompt import get_compliance_verifier_prompt


@dataclass
class LLMJudgeConfig:
    provider_type: str = "local"
    judge_model_name: str = "nuojohnchen/JudgeLRM-7B"
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    judge_top_p: float = 0.9
    judge_model_max_tokens: int = 2048
    judge_model_device: Optional[str] = None  # not needed if using device_map=auto
    judge_model_torch_dtype: str = "bfloat16"
    model_cache_dir: Optional[str] = ""  # 👈 Add this


class ComplianceVerifier:
    def __init__(self, config: LLMJudgeConfig):
        self.config = config
        self.provider_type = config.provider_type.lower()
        self.model_name = config.judge_model_name
        self.top_p = config.judge_top_p
        self.max_tokens = config.judge_model_max_tokens

        if self.provider_type == "local":
            self._init_local_model()
        else:
            raise ValueError(f"Unsupported provider_type: {self.provider_type}")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.is_chat_model = hasattr(self.tokenizer, "apply_chat_template")

        try:
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            pass

    def _resolve_dtype(self, dtype_str):
        if dtype_str == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        return getattr(torch, dtype_str)

    def _init_local_model(self):
        dtype = self._resolve_dtype(self.config.judge_model_torch_dtype)
        cache_path = self.config.model_cache_dir
        torch.manual_seed(42)
        print("Cache path is set to:", cache_path)
        print(f"[LLMJudge] Loading model on multiple GPUs with dtype: {dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,cache_dir=cache_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"

        # Load model using device_map to utilize multiple GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            cache_dir=cache_path,
            device_map="auto"
        ).eval()

        self.is_chat_model = hasattr(self.tokenizer, "apply_chat_template")

    def extract_score(self, text: str) -> float:
        """
        Extracts the correctness score from the **last** line that contains:
        'The correctness score: [[<score>]]'
        This avoids matching example lines earlier in the reasoning.

        Returns:
            float: extracted score, or 0.0 if not found
        """
        lines = text.strip().splitlines()

        for line in reversed(lines):
            line_clean = line.replace("`", "").replace("*", "").strip()
            match = re.search(
                r"(?:correctness\s*score)[^0-9\-]*[:\s]*\[*\[*\s*([0-9]*\.?[0-9]+)\s*\]*\]*",
                line_clean,
                re.IGNORECASE,
            )
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return 0.0

        return 0.0

    def _evaluate_local(self, prompt):
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        score = self.extract_score(response_text)
        return score, response_text

    def _evaluate_local_batch(self, prompts):
        if self.is_chat_model:
            messages_batch = [[{"role": "user", "content": p}] for p in prompts]
            texts = [
                self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
        else:
            texts = prompts

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        scores = [self.extract_score(text) for text in decoded]

        return scores, decoded

    def evaluate_reasoning_batch(self, rules, reasonings, final_answers):
        prompts = [get_compliance_verifier_prompt(rules, r, a) for r, a in zip(reasonings, final_answers)]
        return self._evaluate_local_batch(prompts)
