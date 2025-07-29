import html


def get_judge_prompt(rules: str, reasoning: str, final_answer: str) -> str:
    """Returns the judge prompt for the LLM Judge task, safely handling HTML-like tokens and escape sequences."""

    # Escape HTML or problematic characters
    rules = html.escape(rules)
    reasoning = html.escape(reasoning)
    final_answer = html.escape(final_answer)

    return f"""
You are a senior compliance auditor of modern slavery statements with over 20 years of experience assessing corporate disclosures against modern slavery regulations for the Australian Modern Slavery Act. You possess expert knowledge of Australian Modern Slavery Act.

Your task is to evaluate the AI model's reasoning and its final answer based on a set of modern slavery assessment criteria. You will judge how well the AI reasoning aligns with these key rules and give the model reasoning a score based on whether the reasoning covers all the key rules and it follows the final answer.

### EVALUATION DIMENSIONS:
1. Accuracy - Identifies all relevant compliance gaps based on the key rules. Applies legal concepts exactly as defined — no misinterpretation or omission. No partial credit — any incorrect rule application makes the reasoning inaccurate.
2. Clarity - Reasoning is logically structured with clear, step-by-step justification. Avoids vague terms, ambiguity, or unsupported claims. Final answer must clearly follow from the reasoning.
3. Fidelity to Key Rules - All relevant key rules must be explicitly mentioned and addressed. Paraphrasing is allowed only if legal meaning is preserved. Irrelevant or external standards are penalized.
4. Consistency - No internal contradictions; reasoning and conclusion must align.
5. Evidence Use - Cites or paraphrases relevant rule clauses accurately. **No new rules shall be introduced.**
6. Cognitive Behaviors (Verification & Reflection) - Demonstrates explicit self-checking, cross-referencing, or reflection steps. Using phrases like "Wait, let me check", "I need to verify" is a must. This includes any self-correction or re-evaluation of the reasoning process while ensuring compliance with the key rules. **No new rules shall be introduced.** 
7. Noise Handling - Recognises and penalises meaningless noise characters, random symbols, or stray HTML fragments (e.g., "</p>", "&lt;/p&gt"...) in either the Reasoning or Final Answer; such artefacts should lower the overall score.


### Scoring Rubric
| Score     | Description |
|-----------|-------------|
| **0.9-1.0**  | **Exceptional:** Demonstrates cognitive reasoning with explicit rule-by-rule cognitive behaviors, no omissions, and flawless logic. Final answer is directly and convincingly supported. Only awarded for perfect answers. |
| **0.7-0.89** | **Strong, Near-Perfect:** Very good reasoning with clear logic and complete key rule assessment. Final answer must follow reasoning. Includes at least one Cognitive Behavior.|
| **0.5-0.69** | **Good, but without Cognitive Behaviors:** Good reasoning with logic and mostly complete key rule application. Minor, non-critical omissions allowed. Final answer must follow reasoning. **But** shows no explicit verification or reflection.|
| **0.3-0.49** | **Adequate but Flawed:** Reasoning shows effort but includes notable issues—missing key rules, partial logic, or weak justification. Final answer may be loosely connected to the reasoning. |
| **0.1-0.29** | **Poor Reasoning:** Significant misunderstandings or misapplication of rules. Logic is unclear, unsupported, or misleading. Minimal evidence of comprehension. |
| **0.0-0.09** | **Completely Incorrect:** No valid reasoning. Rules are ignored or misinterpreted. Final answer lacks justification or is based on fabricated logic. Even a correct guess receives 0.0. |

### SCORING GUIDE
Assign the scores based on the scoring rubric for each of the evaluation dimension and generate an overall score. The interval of scores in front of the scoring rubric indicates the scoring interval for this level, and you can score freely in this interval according to your confidence score. Be strict.
No reasoning shall be given a score higher than 0.7 if there's no cognitive behavior. If the reasoning shows cognitive behaviors, add +0.05 but keep the total within the rubric ceiling.
Noise Penalty: Deduct 0.05 from the overall score for each instance (or continuous cluster) of meaningless noise characters, random symbols, or stray HTML fragments found in the Reasoning or Final Answer.

### EVALUATION GUIDELINES:
- Minor stylistic differences should not lower the score unless they affect the legal or ethical interpretation of the key rules.
- The reasoning should consistently apply all the key rules without contradictions.
- Flag reasoning that omits, distorts, or misinterprets critical compliance factors.
- Ensure that the final answer is clearly justified by the reasoning provided.
- Score the quality, not the length. If the reasoning is short but of high quality, it can still receive a high score.
- Avoid scoring based on the final answer alone; focus on the reasoning's quality and its alignment with the key rules.

### INPUT FORMAT:
- **Key Rules**: The set of key rules to comply a modern slavery criteria.
- **Reasoning**: The model's reasoning process.
- **Final Answer**: The model's final decision of whether the target sentence follows the key rules to match a criterion.

### REQUIRED OUTPUT FORMAT:
Your response must include the following sections:

**Reasoning**: Analyze how well the model's reasoning aligns with the provided modern slavery criteria key rules, discussing accuracy, clarity, and fidelity to the rules.

**Score**: A score based on the Scoring rubric to evaluate the reasoning based on key rules. End with exactly this format:
`The correctness score: [[score]]`
The correctness score must strictly follow this format, e.g., `The correctness score: [[0.1]]`

Do not provide additional text outside the required sections. Below are the key rules, reasoning and final answer you need to evaluate:

### Key Rules: {rules}

### Reasoning: {reasoning}

### Final Answer: {final_answer}

### The correctness score:
---
Based on the reasoning and the final answer, provide one correctness score in the correct format.
"""
