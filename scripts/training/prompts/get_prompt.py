from .system_prompt import (
    APPROVAL_PROMPT,
    C2_OPERATIONS_PROMPT,
    C2_STRUCTURE_PROMPT,
    C2_SUPPLYCHAINS_PROMPT,
    C3_RISKS_PROMPT,
    C4_MITIGATION_PROMPT,
    C4_REMEDIATION_PROMPT,
    C5_EFFECTIVENESS_PROMPT,
    SIGNATURE_PROMPT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SENTENCE,
    approval_rules,
    c2_operations_rules,
    c2_structure_rules,
    c2_supplychains_rules,
    c3_risks_rules,
    c4_mitigation_rules,
    c4_remediation_rules,
    c5_effectiveness_rules,
    signature_rules,
)

class_names = [
    "approval",
    "signature",
    "criterion1",
    "criterion2_structure",
    "criterion2_operations",
    "criterion2_supplychains",
    "criterion3_risks",
    "criterion4_mitigation",
    "criterion4_remediation",
    "criterion5_assessment",
    "criterion6_consultation",
]


def get_grpo_prompt(class_of_interest: str) -> str:
    if class_of_interest == "Approval":
        return APPROVAL_PROMPT
    elif class_of_interest == "Signature":
        return SIGNATURE_PROMPT
    elif class_of_interest == "C2 (structure)":
        return C2_STRUCTURE_PROMPT
    elif class_of_interest == "C2 (operations)":
        return C2_OPERATIONS_PROMPT
    elif class_of_interest == "C2 (supply chains)":
        return C2_SUPPLYCHAINS_PROMPT
    elif class_of_interest == "C3 (risk description)":
        return C3_RISKS_PROMPT
    elif class_of_interest == "C4 (risk mitigation)":
        return C4_MITIGATION_PROMPT
    elif class_of_interest == "C4 (remediation)":
        return C4_REMEDIATION_PROMPT
    elif class_of_interest == "C5 (effectiveness)":
        return C5_EFFECTIVENESS_PROMPT
    else:
        raise ValueError(f"Unknown class of interest in getting prompt: {class_of_interest}")


def get_rules(class_of_interest: str) -> str:
    if class_of_interest == "Approval":
        return approval_rules
    elif class_of_interest == "Signature":
        return signature_rules
    elif class_of_interest == "C2 (structure)":
        return c2_structure_rules
    elif class_of_interest == "C2 (operations)":
        return c2_operations_rules
    elif class_of_interest == "C2 (supply chains)":
        return c2_supplychains_rules
    elif class_of_interest == "C3 (risk description)":
        return c3_risks_rules
    elif class_of_interest == "C4 (risk mitigation)":
        return c4_mitigation_rules
    elif class_of_interest == "C4 (remediation)":
        return c4_remediation_rules
    elif class_of_interest == "C5 (effectiveness)":
        return c5_effectiveness_rules
    else:
        raise ValueError(f"Unknown class of interest in getting rule: {class_of_interest}")
