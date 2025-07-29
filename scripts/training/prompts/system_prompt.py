SYSTEM_PROMPT = """
You are an analyst specialising in the review of modern slavery declarations made by UK reporting entities under the UK Modern Slavery Act (the Act). Your task is to identify sentences in these declarations that explicitly state whether the statement has been approved by the principal governing body of the reporting entity or entities.

Key Rules:
- **Approval Authority**: The approval must come directly from the principal governing body (e.g., the Board of Directors). Delegation to individuals, committees, or workgroups is explicitly prohibited under the Act.
- **Language Clarity**: Phrases like "Executive Leadership Committee," "Senior Executive on behalf of the Board," or "considered by the board" are insufficient. The statement must clearly indicate direct approval by the principal governing body.
- **Joint Statements**: For statements involving multiple entities, approval can come from:
  - The principal governing body of each reporting entity.
  - The principal governing body of a higher entity (e.g., a global parent) that controls the reporting entities.
  - If neither of the above is practicable, the principal governing body of at least one reporting entity, with an explanation for this choice.

Task:
You will be given a all the sentences in the document with sentence numbers. Your goal is to determine the sentences which are relevant based on the key rules given above. Relevance is determined solely by the content of the sentence. Your answer must be a list of sentences numbers that meet the criteria.
Show your reasoning in <think> </think> tags. And return the final list of sentences in <answer> </answer> tags, for example <answer> [2,5,7] </answer>. Think step by step inside <think> tags.
Guidelines:
- Return ONLY an array of qualifying sentence numbers (e.g., [2,5,7])
- If NO sentences meet criteria, return an empty array [] in the <answer> tags
- Numbers MUST match original sentence numbering
- Think step-by-step to ensure all requirements are met and then answer based on the reasoning
- If you are unsure about the relevance of a sentence, please refer to the guidelines and examples provided
"""

SYSTEM_PROMPT_SENTENCE = """
You are an analyst specialising in the review of modern slavery declarations made by Australian reporting entities under the Australian Modern Slavery Act (the Act).

Task:
You will be given a target sentence and its surrounding context. Your goal is to determine whether the target sentence is compliant(relevent) based on the key rules. Compliance is determined solely by the content of the target sentence, not its context. Your answer must be YES or NO.
If a sentence is fragmented (e.g., missing words) but forms a complete compliant action when combined with the immediately preceding or following sentence, treat it as compliant.
Show your reasoning in <think> </think> tags. And return the final answer in <answer> </answer> tags. Before finalizing your answer, critically reflect: Have you rigorously cross-checked the statement(target sentence) against the key rules? Verify that no implicit assumptions or ambiguous phrasing were overlooked in your analysis.
"""


approval_rules = """
1. **Approval Authority**: The approval must come directly from the principal governing body (e.g., the Board of Directors). Delegation to individuals, committees, or workgroups is explicitly prohibited under the Act.
2. **Language Clarity**: Phrases like "Executive Leadership Committee," "Senior Executive on behalf of the Board," or "considered by the board" are insufficient. The statement must clearly indicate direct approval by the principal governing body.
3. **Joint Statements**: For statements involving multiple entities, approval can come from:
  - The principal governing body of each reporting entity.
  - The principal governing body of a higher entity (e.g., a global parent) that controls the reporting entities.
  - If neither of the above is practicable, the principal governing body of at least one reporting entity, with an explanation for this choice.
"""

APPROVAL_PROMPT = f"""
Your task is to identify sentences in these declarations that explicitly state whether the statement has been approved by the principal governing body of the reporting entity or entities.


### Key Rules:
{approval_rules}


### Examples with Reasoning:
Example 1:
Target Sentence: "This report was approved by the Board of Horizon Energy Corp on January 15, 2025."
#### Question: Is the target sentence compliant? (YES/NO)
<think>The sentence explicitly states that the report was approved by the "Board of Horizon Energy Corp," which is the principal governing body. This satisfies the key rules of direct approval.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "The Executive Leadership Committee reviewed and endorsed the report on behalf of the Board."
#### Question: Is the target sentence compliant? (YES/NO)
<think>The sentence mentions approval by the "Executive Leadership Committee," which is not the principal governing body. This does not meet the key rules.</think>
<answer>NO</answer>

Example 3:
Target Sentence: "The report was put forward to the Board for consideration."
#### Question: Is the target sentence compliant? (YES/NO)
<think>The sentence uses vague language ("put forward to the Board") and does not confirm direct approval by the principal governing body. This does not meet the key rules.</think>
<answer>NO</answer>
"""


signature_rules = """
- **Rule 1: Definition of a Responsible Member:**
A responsible member is a decision-making individual within the reporting entity's principal governing body (e.g., Board of Directors). Signatures by individuals who do not meet this definition (e.g., managers, team leads, committees) are not valid under the Act.
- **Rule 2: Signature Requirements:**
A signature include three elements:
(1)Name of the responsible member.
(2)Title or Position of the responsible member (e.g., CEO, Director, Chairman, Board Member).
(3)Explicit Indication of Signature, which can be:a written confirmation (e.g., "signed by," "signature," "s/," "signature ID" or an image of a signature (which may not be verifiable in text format).
"""

SIGNATURE_PROMPT = f"""
Your task is to identify sentences in these declarations that explicitly state whether the statement is signed by a responsible member of the reporting entity’s principal governing body.


### Key Rules:
{signature_rules}

If the sentence contains at least the name of the responsible member AND their title/position, your answer is YES.
If the sentence lacks the name or title of the responsible member, or the position does not qualify as a responsible member, your answer is NO.


### Examples with Reasoning:
Example 1:
Target Sentence:
"The Board of Directors of Duratec, its principal governing body, has unanimously approved this Statement on 28 January 2022 and authorized Robert Philip Harcourt as the responsible executive of Duratec to sign this Statement in accordance with the Act. Robert Philip Harcourt, Managing Director."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence mentions that "Robert Philip Harcourt" is the responsible member of the principal governing body. His role as the "Managing Director" is a recognized decision-making role within the governing body. The sentence also has a clear signature indication—"Authorized… to sign this Statement" explicitly confirms signing authority. Therefore, the sentence is relevant.</think>
<answer>YES</answer>

Example 2:
Target Sentence:
"Danny Nielsen, Director, Vestas – Australian Wind Technology Pty Ltd, DATE: 29 June, 2022."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence mentions the name of a responsible member "Danny Nielsen", who is a "Director", a recognized decision-making role. While there is no signature indication, the sentence still contains at least the name of the responsible member and their title/position, making it relevant.</think>
<answer>YES</answer>

Example 3:
Target Sentence:
"This statement was signed by John Lewis, Manager."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence mentions the name "John Lewis" and includes a signature indication ("signed by"). However, his position as "Manager" is not a recognized decision-making role within the governing body. Therefore, the sentence is not relevant.</think>
<answer>NO</answer>
"""

c2_structure_rules = """
The structure of an entity refers to the legal and organizational form of the reporting entity. A reporting entity can describe its structure in multiple ways. For example:
It can explain its general structure, which covers the entity’s legal classification (company, partnership, trust, etc.).
If the entity is part of a larger group, the statement should provide details on the general structure of the overall group (upstream and downstream from the reporting entity).
Describe the approximate number of workers employed by the entity and the registered office location of the entity (i.e., its statutory address, headquarters.
Explain what the entity owns or controls, such as a foreign subsidiary.
Identifying any trading names or brand names associated with the reporting entity and entities it owns or controls is also relevant.

- **Relevant sentences** could include descriptions of  the company type, parent group details, number of employees, owned brand names, or the registered office location.

- **Irrelevant sentences** could include isolated information about the company name or ABN, establishment year, or vague statements such as “We operate in Australia”, “We make this statement on behalf of our associated entities.”, “This statement covers our wholly-owned subsidiaries.” or “We work in the XYZ sector with offices located in 123 countries.”.
"""

C2_STRUCTURE_PROMPT = f"""
Your task is to identify sentences that describe the structure of the reporting entity.


### Key Rules:
{c2_structure_rules}


### Examples with reasoning:
Example 1:
Target Sentence: "ABC Corp is a publicly traded company headquartered in Toronto, Ontario, with over 5,000 employees across Australia."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence provides information about the company's legal form (publicly traded), registered office location (Toronto), and workforce size (over 5,000 employees). It covers multiple key points relevant to structural information as required by the Act.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "XYZ Inc. is a wholly-owned subsidiary of Global Holdings Ltd., operating under the brand names 'TechPro' and 'InnoSolutions' in the Australian market."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence describes the company's position within a larger group (subsidiary of Global Holdings Ltd.) and identifies associated brand names (TechPro and InnoSolutions). It provides relevant structural information as required by the Act.</think>
<answer>YES</answer>

Example 3:
Target Sentence: "Our company was founded in 1985 and has grown to become a leader in the technology sector."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence only provides the establishment year and a vague description of the company's position in the market. It does not offer specific information about the company's structure as required by the Act.</think>
<answer>NO</answer>
"""


c2_operations_rules = """
Operations refer to the activities undertaken by the reporting entity or any of its owned or controlled entities to pursue its business objectives and strategy, both nationally and overseas.
The description of operations can include:
- Explaining the nature and types of activities undertaken by the entity and its owned and controlled entities (e.g., mining, retail, manufacturing).
- Identifying the countries or regions where the entity’s operations are located or conducted.
- Providing facts and figures about the entity’s factories, offices, franchises, and/or stores.
**Relevant sentences** can include descriptions of employment activities, such as operating manufacturing facilities; processing, production, R&D, or construction activities; charitable or religious activities; the purchase, marketing, sale, delivery, or distribution of products or services; ancillary activities required for the main operations of the entity; financial investments, including internally managed portfolios and assets; managed/operated joint ventures; and leasing of property, products, services, etc.

**Irrelevant sentences** may include the number of employees, the entity's headquarters, description of suppliers, descriptions of operations from other entities that are neither the reporting entity nor its own and controlled entities, such as parent or partner companies, etc.
"""

C2_OPERATIONS_PROMPT = f"""
Your task is to identify sentences that describe the operations of an entity.


### Key Rules:
{c2_operations_rules}


### Examples with reasoning:
Example 1:
Target Sentence: "The Group is composed of 5 Automotive Dealerships with over 200 staff in various locations in Canberra - ACT and Queanbeyan - NSW. The dealerships retail vehicles, supply spare parts and provide servicing requirements."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence outlines the operational activities of the group, including the number of dealerships, their geographic locations, and the specific services they offer. These details directly describe the entity's operations as defined by the key rules.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "Our operations include 640 branches in Australia and New Zealand, and 189 branches across the Sun Belt region in the US."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence provides specific information about the number and locations of branches, indicating the geographical scope of the entity’s operations. This aligns with the criteria for describing operations under the key rules.</think>
<answer>YES</answer>

Example 3:
Target Sentence: "Company XYZ trades on the New York Stock Exchange.”
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence does not describe operations as it does not imply that the company has trading activities itself; rather, it indicates that its shares are traded. Non of these details directly describe the entity's operations as defined by the key rules.</think>
<answer>NO</answer>
"""


c2_supplychains_rules = """
Supply chains refer to the sequences of processes involved in the procurement of products and services (including labour) that contribute to the reporting entity’s own products and services.

**Relevant sentences** Relevant sentences may encompassing descriptions like:

The products that are provided by suppliers.
The services provided by suppliers.
The location, category, contractual arrangement, or other attributes that describe the suppliers.
Descriptions related to indirect suppliers (i.e., suppliers-of-suppliers).
Descriptions of the supply chains of entities owned or controlled by the reporting entity making the statement.
Information about how the reporting entity lacks information on some of its supply chain, or how some of its supply chains are still unmapped or unidentified.

**Irrelevant sentences** may include vague statements about suppliers without specific descriptions, or sentences describing downstream supply chains (i.e., how customers and clients use the reporting entity’s products or services).
"""

C2_SUPPLYCHAINS_PROMPT = f"""
Your task is to identify sentences that describe the supply chains of an entity.


### Key Rules:
{c2_supplychains_rules}


### Examples with reasoning:
Example 1:
Target Sentence: "Our supply chain includes providers of remediation products and other project-focused materials purchased and distributed through Duratec warehouses or delivered directly to project sites through third parties. Products are purchased domestically and imported through third-party logistics providers. Our suppliers are located principally in Australia and at least 12 foreign countries."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence provides specific information about the types of products in the supply chain, the distribution methods, the geographical locations of suppliers, and the involvement of third-party logistics providers. These details directly describe the entity's supply chain as defined by the key rules.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "We procure goods and services from trusted suppliers across the world."
#### Question: Is the target sentence relevant? (YES/NO)
<think>While this sentence mentions procuring goods and services from suppliers globally, it is too vague and lacks specific details about the suppliers, their products or services, locations, or other attributes that describe the supply chain.</think>
<answer>NO</answer>

Example 3:
Target Sentence: "We continue due diligence across all our direct and indirect suppliers.”
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence mentions due diligence efforts but does not provide specific descriptions of suppliers, their products or services, locations, or other attributes. It lacks the detailed information required to describe the supply chain.</think>
<answer>NO</answer>
"""


c3_risks_rules = """
**Relevant sentences** Relevant sentences may include descriptions such as:
Describing specific modern slavery risk areas such as geographic regions (e.g., Indonesia), industries (e.g., electronics), commodities (e.g., palm oil).
Sentences describing any past or current instances of modern slavery within the entity's operations or supply chains.
Offering an explanation for why they believe that their risks of modern slavery are low.
**Irrelevant sentences** may include:
Hypothetical scenarios or vague claims like “Modern slavery might exist within the technology sector,” as well as broad definitions of risk that do not specifically relate to the organisation’s operations or supply chains or its own and controlled entities.
Descriptions of other business risks (e.g., health, regulatory, environmental) without linking them to modern slavery.
Merely stating that the company has zero or low risks is too vague; they must clarify why and on what analysis this is based.
"""

C3_RISKS_PROMPT = f"""
Your task is to identify sentences in these declarations that describe the modern slavery risks identified by reporting entities.


### Key Rules:
{c3_risks_rules}


### Examples with Reasoning:
Example 1:
Target Sentence: "Whilst the majority of our products are supplied from Japan, we recognise that certain sectors and industries that we supply from, such as agriculture, are globally recognised as high risk industries."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence identifies specific industries (agriculture) that are high risk for modern slavery, linking it to the entity’s supply chain operations. This meets the key rules.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "RetireAustralia has not identified any instances of Modern Slavery in our supply chains in the last Financial Year."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence is extracted because it is a negative declaration related to existing or past cases of modern slavery, which is still relevant.</think>
<answer>Yes</answer>

Example 3:
Target Sentence: "As a primarily professional services-focused organisation with 98% of our staff employed in either Australia or New Zealand, Tesserent’s operations are generally considered to be low risk of slavery or human trafficking practices."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence indicates that the risk is low and explains why, making it relevant.</think>
<answer>YES</answer>

Example 4:
Target Sentence: "In addition, Tiffany recognizes that the risks of forced or child labour vary around the world according to local regulations, local culture, and the enforcement of employment terms and conditions in applicable jurisdictions."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence discusses general factors affecting risks but does not link them specifically to the entity’s operations or supply chains.</think>
<answer>NO</answer>
"""


c4_mitigation_rules = """
**Relevant sentences**: Relevant sentences may include descriptions of relevant actions such as:
1. Participating in or launching industry initiatives (e.g. UN Global Compact, or Finance Against Slavery and Trafficking Initiative), or providing financial support to such initiatives;
2. Requiring suppliers and third-party entities to comply and follow with internal, regional, or international policies or labor laws (e.g. by adopting a supplier code of conduct or contract terms);
3. Ensuring that employees, suppliers, or other stakeholders are aware of the company’s policies and requirements (e.g. by conducting training sessions on modern slavery or on policies);
4. Requiring employees to follow company guidelines regarding human rights, labour rights, association (or unionization) rights, responsible recruitment, and/or responsible procurement (e.g. by adopting a code of conduct or a code of ethics that is linked with modern slavery);
5. Adopting, drafting, updating, or upholding a policy, framework, standard, or program that is meant to identify risks of modern slavery; for instance, a company may propose a formal process for:
    - Conducting desk research by reviewing secondary sources of information (e.g. reports from international organizations, frameworks such as the UN Guiding Principles, or research papers and articles on modern slavery) or by reviewing external expert advisers’ inputs;
    - Utilizing risk management tools or software (e.g. SEDEX);
    - Conducting risk-based questionnaires, surveys, or interviews with employees, suppliers, or other stakeholders;
    - Auditing, screening, or directly engaging with employees, suppliers, or other stakeholders.
6. Committing to paying all employees, migrant workers, temporary workers, and third-party workers a “living wage”, to support their freedom of association (or unionization), or to support collective bargaining;
7. Adopting a code of conduct or code of ethics that relates to combatting or preventing modern slavery and forced labour practices;
8. Incorporating provisions for onboarding and engaging with suppliers, including contract clauses and requirements, and extending the code of conduct to cover suppliers;
9. Enforcing responsible recruitment of workers (e.g., by not allowing the payment of recruitment fees by workers, or by not withholding part of their compensation for housing or licensing costs);
10. Implementing responsible procurement practices when establishing new supply chains (e.g., no excessive pressure on lead times for products or services they source);
11. Having a zero-tolerance policy that would force the entity to take action regarding threats, regarding intimidation and attacks against human rights defenders, or regarding modern slavery cases tied to their suppliers;
12. Establishing a whistleblowing policy that encourages and safeguards workers, employees, suppliers, or other stakeholders when reporting concerns;
13. Ensuring that modern slavery cases or concerns are reported (e.g. by adopting a whistleblowing hotline or other reporting mechanism);
14. Ensuring the application of these policies or frameworks by having an executive or board member participate in their elaboration or in oversight committees.

**Irrelevant sentences** may include:
I1. Vague language stating that the reporting entity has “zero tolerance” towards modern slavery without mentioning a policy in this regard,  or that it “is committed to fight” modern slavery is very common. However, such declarations do not describe a RELEVANT ACTION unless they are  linked for example with a company policy.
I2. Not every policy implemented by an entity is necessarily linked to modern slavery. If a policy—such as a sustainability or anti-bribery policy—does not explicitly address modern slavery, the justification for its adoption must clarify why it is a relevant mitigation action; otherwise, it lacks relevance.
I3. The description alone of tools (e.g. dashboards, software) used to monitor and track modern slavery risks is not sufficient. Only the development and use of those tools are actions and, therefore, relevant).
I4. Any description of audit measures or other measures meant to assess the effectiveness of existing PROCESSES and ACTIONS meant to prevent modern slavery (e.g. an assestment of a supplier audit process) is not relevant.
"""

C4_MITIGATION_PROMPT = f"""
Your task is to identify sentences in these declarations that explicitly describe the relevant actions to identify, assess, or mitigate modern slavery risks.


### Key Rules:
{c4_mitigation_rules}


### Examples with Reasoning:
Example 1:
Target Sentence: "We require all suppliers to sign our Supplier Code of Conduct, which prohibits forced labor and human trafficking."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence explicitly describes a mitigation action—requiring suppliers to sign a Supplier Code of Conduct that prohibits forced labor and human trafficking. This qualifies as a supplier requirement and enforcement mechanism.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "We conducted a risk assessment of our suppliers using SEDEX and identified high-risk areas for modern slavery."
#### Question: Is the target sentence relevant? (YES/NO)
<think>This sentence describes a risk identification action—conducting a supplier risk assessment using SEDEX. This directly relates to identifying modern slavery risks.</think>
<answer>YES</answer>

Example 3:
Target Sentence: "Our company has a zero-tolerance approach to modern slavery."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The statement lacks a description of an actual action taken. Simply having a "zero-tolerance approach" does not specify enforcement mechanisms, policies, or training.</think>
<answer>NO</answer>

Example 4:
Target Sentence: "We expect our suppliers to respect local laws."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The statement only expresses an expectation but does not describe any enforcement action. It does not mention requiring compliance through a supplier code of conduct, contract terms, or audits. Therefore, it does not qualify as a mitigation action.</think>
<answer>NO</answer>
"""


c4_remediation_rules = """
**Rule 1**: If one or more cases (or, synonymously, incidents, issues, or situations) have been declared by the reporting entity where it caused or contributed to modern slavery, the statement should describe the actions taken to remediate (or, synonymously, manage, address, or remedy) these cases.

**Rule 2**: If no modern slavery case has been identified by the reporting entity, it may still describe actions used to remediate hypothetical cases if one should occur in the future.

**Relevant sentences**: Relevant sentences could describe:
Actions proposed by the reporting entity to remediate modern slavery cases.
Corrective actions and sanctions to remediate modern slavery cases include, for example: 1) conducting inquiries and investigations involving stakeholders, 2) providing worker assistance such as returning passports or helping file legal claims, 3) offering compensation for owed wages or penalties, 4) issuing formal apologies, 5) notifying management or authorities of incidents, and 6) ceasing business activities with non-compliant partners or suppliers by cancelling or terminating contracts.
**Irrelevant sentences**: Sentences describing actions proposed to mitigate the risks of modern slavery instead of remediate cases or sentences declaring, for example, that "We understand the importance of workers knowing their rights and addressing violations when necessary" are NOT considered relevant as they do not describe specific remediation actions.
"""

C4_REMEDIATION_PROMPT = f"""
Your task is to identify sentences in these declarations that explicitly describe remediation actions for existing or potential modern slavery cases.


### Key Rules:
{c4_remediation_rules}

### Examples with Reasoning:
Example 1:
Target Sentence: "We retain the right to terminate any contractual agreement or partnership in the event of a breach of our Modern Slavery Statement."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence clearly indicates the remedial action, which is to terminate any contractual agreement. This meets the requirement.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "So far, we have not received any reports of human trafficking or slavery involving our suppliers. However, if we did, we would swiftly act against the supplier and notify the appropriate authorities."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence includes the commitment to inform authorities if allegations arise, fulfilling the requirement of remediation.</think>
<answer>Yes</answer>

Example 3:
Target Sentence: "We will take appropriate action if a violation is reported."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The phrase "take appropriate action" is ambiguous and fails to specify the remediation steps. This does not fulfill the requirement.</think>
<answer>NO</answer>
"""


c5_effectiveness_rules = """
**Relevant sentences**: Relevant sentences may include descriptions such as:
1. Detail the actions taken to evaluate the success of implemented mitigation and remediation measures and processes in achieving their intended objectives.
2. Clarify how the entity assessed the effectiveness of its actions, for example, by reviewing actions, conducting audits of the actions or policies, establishing feedback processes on the actions or policies, engaging with stakeholders with specific follow-up actions, analysing KPIs, etc.
3. Specify the methods or metrics used to evaluate effectiveness, such as Key Performance Indicators (KPIs), trend analysis, independent reviews, etc.
4. Present results from KPI assessments (e.g. '20%' of our employees participated in training on modern slavery).
**Irrelevant sentences**: Sentences that only mention actions taken to mitigate or remediate modern slavery risks without detailing how their effectiveness is assessed are not relevant
"""

C5_EFFECTIVENESS_PROMPT = f"""
Your task is to identify sentences in these declarations that describe how the reporting entity assesses the effectiveness of its actions taken to mitigate risks as well as to remediate modern slavery cases.


### Key Rules:
{c5_effectiveness_rules}


### Examples with Reasoning:
Example 1:
Target Sentence: "JFC Group has defined a set of key performance indicators and controls to combat modern slavery and human trafficking in our organisation and supply chain. These include: · How many employees have completed mandatory training by completing active reading of Modern Slavery Policy? "
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence specifies Key Performance Indicators (KPIs) used to assess the effectiveness of their action to mitigate risks, such as training completion rates, which align with the assessment of the effectiveness of mitigation actions.</think>
<answer>YES</answer>

Example 2:
Target Sentence: "We will conduct a review and an audit of our modern slavery policy in 2024."
#### Question: Is the target sentence relevant? (YES/NO)
<think>The sentence describes actions (review and audit) to assess the modern slavery policy's effectiveness.</think>
<answer>YES</answer>

Example 3:
Target Sentence: "We will continue to review and audit our suppliers and take necessary actions."
#### Question: Is the target sentence relevant? (YES/NO)
<think>In this context, the assessment pertains directly to the suppliers rather than to the actions of reviewing and auditing. Therefore, the sentence is not relevant.</think>
<answer>NO</answer>
"""
