"""
DeepSeek Judge Benchmark — Test v4-flash and v4-pro at multiple concurrency levels.

Measures avg/p95 latency, throughput, error rate, and output validity.

Usage:
    python tests/judge_deepseek_benchmark.py
"""

import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODELS = [
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
]
CONCURRENCIES = [1, 2, 4, 8]
SAMPLES = 5
SAMPLE_QUERIES = [
    "My order hasn't arrived yet, where is it?",
    "I need to reset my password, can you help?",
    "How do I return a defective item?",
    "What is your refund policy?",
    "I was charged twice for the same order",
    "Can I change my shipping address?",
    "How long does standard shipping take?",
    "Do you offer international shipping?",
    "My account is locked, what should I do?",
    "Can I get a copy of my invoice?",
]

JUDGE_PROMPT = """You are an expert quality evaluator for customer support responses.

User question: {question}
AI response: {response}

Grade the AI response on two dimensions:
1. correctness: Does the response accurately address the question? (0.0 to 1.0)
2. helpfulness: Is the response helpful and clear? (0.0 to 1.0)

Output ONLY a JSON object with no other text:
{{"correctness": <float>, "helpfulness": <float>, "reasoning": "<brief explanation>"}}
"""


def make_client():
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
    )


def single_judge_call(client, model: str, question: str, response: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, response=response)
    start = time.time()
    ttft = None
    chunks = []

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
            timeout=120,
        )
        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = time.time() - start
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        total = time.time() - start
        text = "".join(chunks)

        # Validate output
        valid = False
        try:
            parsed = json.loads(text)
            valid = isinstance(parsed, dict) and "correctness" in parsed and "helpfulness" in parsed
        except (json.JSONDecodeError, ValueError):
            # Try to find JSON in the response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    valid = isinstance(parsed, dict) and "correctness" in parsed and "helpfulness" in parsed
                except (json.JSONDecodeError, ValueError):
                    pass

        return {
            "success": True,
            "latency_s": round(total, 2),
            "ttft_s": round(ttft, 2) if ttft else None,
            "output_valid": valid,
            "output": text[:100],
        }
    except Exception as e:
        return {
            "success": False,
            "latency_s": round(time.time() - start, 2),
            "ttft_s": None,
            "output_valid": False,
            "error": str(e)[:100],
        }


def run_batch(model: str, concurrency: int, queries: list) -> dict:
    client = make_client()
    responses = ["Thank you for your inquiry. Our team will assist you shortly."] * len(queries)

    print(f"  {model} @ concurrency={concurrency}: running...", flush=True)
    start = time.time()
    results = [None] * len(queries)
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(single_judge_call, client, model, queries[i], responses[i]): i
            for i in range(len(queries))
        }
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()
            completed += 1
            r = results[idx]
            status = "OK" if r["success"] else "ERR"
            val = "V" if r.get("output_valid") else "X"
            lat = r["latency_s"]
            print(f"    [{completed}/{len(queries)}] req#{idx} {status} lat={lat}s valid={val}", flush=True)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    valid_outputs = [r for r in successful if r["output_valid"]]
    latencies = [r["latency_s"] for r in successful]
    ttfts = [r["ttft_s"] for r in successful if r.get("ttft_s")]

    elapsed = time.time() - start
    stats = {
        "model": model,
        "concurrency": concurrency,
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "error_rate_pct": round(len(failed) / len(results) * 100, 1) if results else 0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "min_latency_s": round(min(latencies), 2) if latencies else None,
        "max_latency_s": round(max(latencies), 2) if latencies else None,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) >= 20 else (round(sorted(latencies)[-1], 2) if latencies else None),
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "output_valid_pct": round(len(valid_outputs) / len(successful) * 100, 1) if successful else 0,
        "total_time_s": round(elapsed, 1),
        "throughput_req_per_min": round(len(results) / elapsed * 60, 1) if elapsed > 0 else 0,
    }

    print(f"  -> avg={stats['avg_latency_s']}s p95={stats['p95_latency_s']}s err={stats['error_rate_pct']}% valid={stats['output_valid_pct']}% tput={stats['throughput_req_per_min']}/min", flush=True)
    return stats


def main():
    print("=" * 65)
    print("  DEEPSEEK JUDGE BENCHMARK")
    print("=" * 65)

    import random
    random.seed(42)
    experiments = [(m, c) for m in MODELS for c in CONCURRENCIES]
    random.shuffle(experiments)

    all_results = []
    for model, concurrency in experiments:
        print(f"\n--- {model} @ concurrency={concurrency} ---")
        stats = run_batch(model, concurrency, SAMPLE_QUERIES * (SAMPLES // len(SAMPLE_QUERIES) + 1))
        all_results.append(stats)
        time.sleep(2)  # cooldown between tests

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Model':<35} {'Conc':>4} {'Avg':>7} {'P95':>7} {'TTFT':>7} {'Err%':>6} {'Valid%':>7} {'Req/min':>8}"
    print(header)
    print("-" * 80)
    for r in all_results:
        model_short = r["model"].replace("deepseek-ai/", "")
        avg = f"{r['avg_latency_s']}s" if r['avg_latency_s'] else "N/A"
        p95 = f"{r['p95_latency_s']}s" if r['p95_latency_s'] else "N/A"
        ttft = f"{r['avg_ttft_s']}s" if r['avg_ttft_s'] else "N/A"
        err = f"{r['error_rate_pct']}%"
        valid = f"{r['output_valid_pct']}%"
        tput = f"{r['throughput_req_per_min']}"
        print(f"{model_short:<35} {r['concurrency']:>4} {avg:>7} {p95:>7} {ttft:>7} {err:>6} {valid:>7} {tput:>8}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "deepseek_judge_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
