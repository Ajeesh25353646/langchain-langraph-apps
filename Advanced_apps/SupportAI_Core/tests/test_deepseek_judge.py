"""
TDD Step 1 (RED): Test that DeepSeek V4 Flash Free can parse the exact
openevals ANSWER_RELEVANCE_PROMPT and CORRECTNESS_PROMPT templates
and return a parseable numeric score.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from openai import OpenAI

JUDGE_MODEL = "deepseek-v4-flash-free"
ZEN_BASE_URL = "https://opencode.ai/zen/v1"
ZEN_API_KEY = os.environ.get("ZEN_API_KEY", "")

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

<Instructions>
  For each output:
  - Read the original input carefully to understand what was asked
  - Examine the output and identify its core claim or purpose
  - Determine if the output directly addresses the input's request
  - Assess whether the information helps fulfill what was asked
  - Determine the answer relevancy of output and output a score

  Focus on whether each statement helps answer the specific input question, not whether the statement is true or well-written. A statement can be factually correct but still irrelevant if it doesn't address what was asked.
</Instructions>

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

<Instructions>
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity
</Instructions>

The goal is to evaluate factual correctness and completeness of the response.

Input: {inputs}

Output: {outputs}

Use the reference outputs below to help you evaluate the correctness of the response.

Output your score as a single number 0-10 with no other text.

Reference output: {reference_outputs}
Score:"""


def _call_judge(prompt: str) -> float | None:
    client = OpenAI(base_url=ZEN_BASE_URL, api_key=ZEN_API_KEY)
    r = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=60,
    )
    text = r.choices[0].message.content.strip()
    nums = re.findall(r'\d+\.?\d*', text)
    return float(nums[-1]) if nums else None


def test_relevance_returns_score():
    """RED: This test will fail until we implement the full pipeline."""
    prompt = ANSWER_RELEVANCE_PROMPT.format(
        inputs="What is your return policy?",
        outputs="Our policy allows returns within 30 days.",
    )
    score = _call_judge(prompt)
    print(f"  Relevance prompt returned score: {score}")
    assert score is not None, f"Expected a number, got None"
    assert 0 <= score <= 10, f"Expected score 0-10, got {score}"
    print("  PASS: relevance_returns_score")


def test_correctness_returns_score():
    """RED: This test will fail until we implement the full pipeline."""
    prompt = CORRECTNESS_PROMPT.format(
        inputs="What is your return policy?",
        outputs="Our policy allows returns within 30 days.",
        reference_outputs="Returns are accepted within 30 days of purchase with original receipt.",
    )
    score = _call_judge(prompt)
    print(f"  Correctness prompt returned score: {score}")
    assert score is not None, f"Expected a number, got None"
    assert 0 <= score <= 10, f"Expected score 0-10, got {score}"
    print("  PASS: correctness_returns_score")


def test_score_high_for_good_answer():
    """A good answer should get a high relevance score."""
    prompt = ANSWER_RELEVANCE_PROMPT.format(
        inputs="What is your return policy?",
        outputs="Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
    )
    score = _call_judge(prompt)
    print(f"  Good answer relevance score: {score}")
    assert score is not None and score >= 5, f"Expected high score (>=5), got {score}"
    print("  PASS: score_high_for_good_answer")


def test_score_low_for_bad_answer():
    """A bad/irrelevant answer should get a low relevance score."""
    prompt = ANSWER_RELEVANCE_PROMPT.format(
        inputs="What is your return policy?",
        outputs="The weather is nice today. I like sunny days.",
    )
    score = _call_judge(prompt)
    print(f"  Bad answer relevance score: {score}")
    assert score is not None and score <= 5, f"Expected low score (<=5), got {score}"
    print("  PASS: score_low_for_bad_answer")


if __name__ == "__main__":
    print("=" * 60)
    print("  TDD Step 1: DeepSeek Judge Tests")
    print("=" * 60)
    
    tests = [
        test_relevance_returns_score,
        test_correctness_returns_score,
        test_score_high_for_good_answer,
        test_score_low_for_bad_answer,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
    
    print(f"\n  {passed}/{len(tests)} tests passed")
