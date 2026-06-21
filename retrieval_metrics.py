"""Prompts for each reasoning node. Kept here (not inline) because prompt quality is
a big part of agent quality — these were iterated against real failures:
  - DECOMPOSE: tightened with a rule + few-shot examples to stop over-decomposition
    (it was splitting simple single-topic questions into 4 sub-questions).
  - GRADE: reframed from "is this exhaustive?" to "can the question be answered?" —
    the original was too strict and caused unnecessary retrieval loops.
  - GENERATE: strict grounding ("ONLY the passages") to prevent hallucination.
"""

DECOMPOSE = """You decide whether a question needs to be broken into sub-questions, \
and if so, break it into the MINIMAL set.

RULE: Only decompose if answering requires finding separate facts in sequence — for \
example, identifying one entity and then asking something about that entity ("Who \
founded the company that acquired X?" -> find the company, then its founder).

If the question is about a SINGLE topic — even a broad one — return it UNCHANGED as a \
single-item list. Do NOT split a topic into aspects or sub-themes.

Examples:
- "How did orchestras change in the Romantic period?" -> ["How did orchestras change in the Romantic period?"]
- "Who founded the company that acquired DeepMind?" -> ["What company acquired DeepMind?", "Who founded that company?"]

Question: {question}

Return ONLY a JSON list of strings, nothing else."""

GRADE = """You judge whether the question can be ANSWERED from the retrieved passages. \
The passages do NOT need to cover the topic exhaustively — they only need to contain \
enough to answer THIS specific question.

Mark "sufficient": true if a correct answer can be derived from the passages, even if \
they don't cover every aspect of the topic.
Mark "sufficient": false ONLY if the core information needed is genuinely absent or \
contradictory.

Question: {question}

Passages:
{passages}

Return ONLY a JSON object: {{"sufficient": true or false, "missing": "what core fact is missing, or null"}}"""

REWRITE = """The question '{question}' is missing: {missing}. Write ONE focused search \
query to find it. Return only the query."""

GENERATE = """Answer the question using ONLY the passages below. Do not use outside \
knowledge. If the passages don't contain the answer, say exactly: "I cannot answer \
this from the provided sources."

Cite the passage numbers you used in square brackets, e.g. [1][3].

Question: {question}

Passages:
{passages}

Answer:"""
