"""
Quality Evaluation — Measure Gemma 4 31B agent output quality across
3 dimensions using DeepSeek V4 Flash Free (OpenCode Zen) as LLM judge.

Usage:
    python tests/evaluate_quality.py

Requires:
    LANGSMITH_API_KEY, ZEN_API_KEY, GOOGLE_API_KEY in .env
"""

import os
import re
import sys
import time
import uuid
from dotenv import load_dotenv
from langsmith import Client, evaluate, traceable
from langchain_core.messages import HumanMessage
from openai import OpenAI

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JUDGE_MODEL = "deepseek-v4-flash-free"
DATASET_NAME = "Customer Support Evaluation Dataset"
ZEN_BASE_URL = "https://opencode.ai/zen/v1"

ANSWER_RELEVANCE_PROMPT = """You are an expert evaluator assessing whether outputs are relevant to the given input. Your task is to determine whether EACH statement appropriately addresses what was asked.

<Rubric>
  A relevant output:
  - Directly answers the question or addresses the request
  - Provides information specifically asked for
  - Stays on topic with the input's intent
  - Contributes meaningfully to fulfilling the request

  An irrelevant output:
  - Discusses topics not requested or implied by the input
  - Provides unnecessary tangents or digressions
  - Includes information that doesn't answer the question
  - Addresses a different question than what was asked
</Rubric>

Now, grade the following example according to the above instructions.

Output your score as a single number 0-10 with no other text.

Input: {inputs}

Output: {outputs}
Score:"""

CORRECTNESS_PROMPT = """You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  When scoring, you should penalize:
  - Factual errors or inaccuracies
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing key information
</Rubric>

Now, grade the following example according to the above instructions.

Output your score as a single number 0-10 with no other text.

Input: {inputs}

Output: {outputs}

Reference output: {reference_outputs}
Score:"""


def _call_judge(prompt: str) -> float | None:
    client = OpenAI(base_url=ZEN_BASE_URL, api_key=os.environ.get("ZEN_API_KEY", ""))
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=60,
            )
            text = r.choices[0].message.content.strip()
            nums = re.findall(r'\d+\.?\d*', text)
            return float(nums[-1]) if nums else None
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                delay = min(3.0 * (2 ** attempt), 30.0)
                print(f"  [RETRY] 429 on judge, retry in {delay:.1f}s (attempt {attempt+1}/5)")
                time.sleep(delay)
            else:
                print(f"  [JUDGE ERROR] {e}")
                return None
    return None


def _relevance_judge(inputs, outputs, reference_outputs):
    score = _call_judge(ANSWER_RELEVANCE_PROMPT.format(
        inputs=inputs.get("question", ""),
        outputs=outputs.get("resolution", ""),
    ))
    return {"score": (score or 0) / 10, "key": "relevance"}


def _correctness_judge(inputs, outputs, reference_outputs):
    score = _call_judge(CORRECTNESS_PROMPT.format(
        inputs=inputs.get("question", ""),
        outputs=outputs.get("resolution", ""),
        reference_outputs=reference_outputs.get("expected_resolution", ""),
    ))
    return {"score": (score or 0) / 10, "key": "correctness"}


relevance_eval = _relevance_judge
correctness_eval = _correctness_judge


def intent_accuracy(inputs, outputs, reference_outputs):
    prediction = outputs.get("intent")
    reference = reference_outputs.get("expected_intent")
    return {"score": 1 if prediction == reference else 0, "key": "intent_accuracy"}


@traceable(run_type="chain", name="Support Agent (Gemma 4 31B)")
def predict_support_response(inputs: dict):
    import main
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "messages": [HumanMessage(content=inputs["question"])],
        "customer_id": "EVAL-QUALITY",
        "ticket_id": f"QUAL-{uuid.uuid4().hex[:6]}",
    }
    final_state = main.app.invoke(initial_state, config=config)
    return {
        "intent": final_state.get("intent"),
        "sentiment": final_state.get("sentiment"),
        "resolution": final_state.get("resolution"),
    }


if __name__ == "__main__":
    client = Client()

    print("=" * 65)
    print("  QUALITY EVALUATION")
    print(f"  Agent: Gemma 4 31B (Google AI Studio)")
    print(f"  Judge: {JUDGE_MODEL} (OpenCode Zen)")
    print(f"  Dataset: {DATASET_NAME}")
    print("=" * 65)

    results = client.evaluate(
        predict_support_response,
        data=DATASET_NAME,
        evaluators=[
            relevance_eval,
            correctness_eval,
            intent_accuracy,
        ],
        experiment_prefix="Gemma4B-Quality-ZenJudge",
        max_concurrency=2,
    )

    print(f"\nDone! View results: {results.experiment_name}")
