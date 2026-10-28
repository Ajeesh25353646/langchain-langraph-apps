# Plan & Outcome: LLM Judge Evaluation for Guardrail Gap Attacks

## Outcome

This evaluation closed the loop on the layered defense. 319 attacks that passed the
deterministic guardrail (44% block rate, 0% false positives on 70 benign queries) went
through the full agent — and all 319 were **refused by the prompt-based defense layer**:
**100% agent refusal rate, 100% combined block rate**. The deterministic layer alone was
never the whole story; the gap attacks are where the prompt layer proves itself.

Authoritative numbers: `Evals/guardrail_gap_results.json` (this evaluation, with
per-category refusal breakdown) and `Evals/guardrail_eval_results.json` (guardrail layer,
with per-category block rates). The complete layered-defense narrative — deterministic
guardrail, PII anonymization, `<USER_INPUT>` instruction-data separation, LLM-as-a-Judge —
is in the README, section "Two-Layer Evaluation".

## What this evaluation was for

The deterministic guardrail (regex, keyword, URL, token, and Unicode checks — implemented
in `main.py`'s `Guardrail` class) is layer 1 only. The question this evaluation answers:
what happens to the attacks it misses? Layer 3 is the agent prompt, which wraps user input
in `<USER_INPUT>` tags and instructs the model to treat their contents as DATA ONLY. Every
guardrail-passing attack was sent through the full LangGraph agent, and an LLM judge
classified each response as "complied" vs "refused", yielding the combined block rate.

## Plan (as executed)

1. Create `config/models.yaml` — judge model registry across Google, OpenRouter, Nvidia
2. Install `langchain-nvidia-ai-endpoints`
3. Write `tests/judge_latency_test.py` — 10 sample inputs, latency table per provider
4. Run the latency test → **Nemotron 3 Super chosen** (`nvidia/nemotron-3-super-120b-a12b`)
5. Write `tests/evaluate_guardrail_gaps.py` — run all 319 gap attacks through
   agent + judge (checkpointed, resumable)
6. Run the evaluation → 319/319 processed, 0 errors, 319 refused

Done when: every gap attack has a judged verdict and the summary in
`Evals/guardrail_gap_results.json` carries `combined_block_rate_pct`. That criterion is met.
