import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
import torch
from prompts.llm_judge_prompt import get_judge_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

_SCORE_PATTERN = re.compile(
    r"(?:correctness\s*score)[^0-9\-]*[:\s]*\[*\[*\s*([0-9]*\.?[0-9]+)\s*\]*\]*",
    re.IGNORECASE,
)


@dataclass
class LLMJudgeConfig:
    provider_type: str = "local"  # "local" or "api"
    judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: Optional[str] = None  # for API
    api_url: Optional[str] = None  # for API
    judge_top_p: float = 0.9
    judge_model_max_tokens: int = 2048
    judge_model_device: Optional[str] = None  # for local
    judge_model_torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", etc.


class BaseLLMJudge:
    def __init__(self, config: LLMJudgeConfig):
        self.config = config
        self.provider_type = config.provider_type.lower()
        self.model_name = config.judge_model_name
        self.top_p = config.judge_top_p
        self.max_tokens = config.judge_model_max_tokens

        if self.provider_type == "local":
            self._init_local_model()
        elif self.provider_type == "api":
            self._init_api_client()
        else:
            raise ValueError(f"Unknown provider_type: {self.provider_type}")

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
        device_str = self.config.judge_model_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        dtype = self._resolve_dtype(self.config.judge_model_torch_dtype)

        print(f"[LLMJudge] Loading model on device: {self.device}, dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Fix pad_token if missing
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.vocab_size - 1)
        self.tokenizer.padding_side = "left"

        # Load model WITHOUT device_map or low_cpu_mem_usage (not supported in ZeRO-3)
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

        self.is_chat_model = hasattr(self.tokenizer, "apply_chat_template")

        # Set generation config safely
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            pass  # ignore if not needed

        if not self.is_chat_model:
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device.index if self.device.type == "cuda" else -1,
            )

    def _init_api_client(self):
        self.api_key = self.config.api_key or os.getenv("API_KEY")
        self.api_url = self.config.api_url
        if not self.api_url or not self.api_key:
            raise ValueError("Both `api_url` and `api_key` must be set for API-based LLM judge.")

    def _extract_score_batch(self, responses):
        scores = []
        for text in responses:
            lines = text.strip().splitlines()
            for line in reversed(lines):
                line_clean = line.replace("`", "").replace("*", "").strip()
                match = _SCORE_PATTERN.search(line_clean)
                if match:
                    try:
                        scores.append(float(match.group(1)))
                        break
                    except ValueError:
                        break
            else:
                scores.append(0.0)
        return scores

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

    def evaluate_reasoning(self, rules, reasoning, final_answer):
        prompt = get_judge_prompt(rules, reasoning, final_answer)

        if self.provider_type == "local":
            return self._evaluate_local(prompt)
        else:
            return self._evaluate_api(prompt)

    def _evaluate_local(self, prompt):
        # Optional cache to avoid re-evaluating the same prompt
        if not hasattr(self, "_reward_cache"):
            self._reward_cache = {}

        if prompt in self._reward_cache:
            return self._reward_cache[prompt]

        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)

            # Try to access model.module if DeepSpeed is wrapping it
            model = self.model.module if hasattr(self.model, "module") else self.model

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                )

            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:
            # fallback to pipeline-based eval
            outputs = self.pipe(prompt, max_new_tokens=self.max_tokens)
            response_text = outputs[0]["generated_text"].split("Reward:")[-1].strip()

        score = self.extract_score(response_text)
        self._reward_cache[prompt] = (score, response_text)
        return score, response_text

    def _evaluate_api(self, prompt):
        if "openai.com" in self.api_url:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
        elif "generativelanguage.googleapis.com" in self.api_url:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
        else:
            payload = {"prompt": prompt, "max_tokens": self.max_tokens}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            if "openai.com" in self.api_url:
                response_text = response.json()["choices"][0]["message"]["content"]
            elif "generativelanguage.googleapis.com" in self.api_url:
                response_text = (
                    response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                )
            else:
                response_text = response.text

            return self.extract_score(response_text), response_text
        else:
            print("API Error:", response.text)
            return 0.0, f"API Error: {response.text}"

    def _evaluate_local_batch(self, prompts):
        # Optional cache
        if not hasattr(self, "_reward_cache"):
            self._reward_cache = {}

        scores = []
        responses = []
        uncached_prompts = []
        uncached_indices = []

        # Check cache
        for idx, prompt in enumerate(prompts):
            if prompt in self._reward_cache:
                s, r = self._reward_cache[prompt]
                scores.append(s)
                responses.append(r)
            else:
                scores.append(None)  # fill with None to preserve index
                responses.append(None)
                uncached_prompts.append(prompt)
                uncached_indices.append(idx)

        if not uncached_prompts:
            return scores, responses

        if self.is_chat_model:
            # Prepare chat messages for all prompts
            messages_batch = [[{"role": "user", "content": p}] for p in uncached_prompts]
            texts = [
                self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            model = self.model.module if hasattr(self.model, "module") else self.model
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        else:
            # batch fallback using pipeline
            outputs = self.pipe(
                uncached_prompts,
                max_new_tokens=self.max_tokens,
                return_full_text=True,
                batch_size=8,  # adjust this based on your memory
            )
            decoded = [
                out[0]["generated_text"].split("Reward:")[-1].strip()
                if isinstance(out, list)
                else out["generated_text"].split("Reward:")[-1].strip()
                for out in outputs
            ]

        batch_scores = self._extract_score_batch(decoded)

        for idx, prompt, score, response in zip(uncached_indices, uncached_prompts, batch_scores, decoded):
            self._reward_cache[prompt] = (score, response)
            scores[idx] = score
            responses[idx] = response

        return scores, responses

    def evaluate_reasoning_batch(self, rules, reasonings, final_answers):
        prompts = [get_judge_prompt(rules, r, a) for r, a in zip(reasonings, final_answers)]
        return self._evaluate_local_batch(prompts)
