"""
Latency Benchmark — Compare v1 (Gemma 4 31B) vs v2 (DeepSeek V4 Flash)
for portfolio latency comparison table.

Usage:
    python tests/benchmark_latency.py

Requires:
    GOOGLE_API_KEY, ZEN_API_KEY in .env
"""

import importlib
import os
import sys
import time
import json
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAMPLE_QUERIES = [
    "My phone battery dies within 2 hours of charging, what should I do?",
    "I want to return a product I bought 45 days ago",
    "How do I reset my account password?",
    "What is your refund policy for defective items?",
    "I was charged twice for the same order, can you help?",
]


def reroute_get_llm(original_fn):
    """Wraps get_llm to measure creation + first-invoke latency."""
    import functools
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        llm = original_fn(*args, **kwargs)
        original_invoke = llm.invoke
        invoke_times = []
        def timed_invoke(*iargs, **ikwargs):
            t0 = time.time()
            result = original_invoke(*iargs, **ikwargs)
            invoke_times.append(time.time() - t0)
            return result
        llm.invoke = timed_invoke
        llm._timing_invoke_times = invoke_times
        return llm
    return wrapper


def run_benchmark(label: str, main_module_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"{'='*60}")

    # Fresh import of the module
    if "main" in sys.modules:
        del sys.modules["main"]
    # Clean up any cached test artifacts
    spec = importlib.util.spec_from_file_location("main", main_module_path)
    main = importlib.util.module_from_spec(spec)
    # Patch log_timing to also capture to list
    timing_log = []
    original_log_timing_path = os.path.join(os.path.dirname(main_module_path), "main.py")
    
    sys.modules["main"] = main
    spec.loader.exec_module(main)

    # Collect timing data from the module's log_timing output
    app = main.app

    per_query = []
    for q_idx, query in enumerate(SAMPLE_QUERIES):
        print(f"\n  --- Query {q_idx+1}/{len(SAMPLE_QUERIES)}: {query[:50]}... ---")
        config = {"configurable": {"thread_id": f"bench-{q_idx}"}}
        initial_state = {
            "messages": [main.HumanMessage(content=query)],
            "customer_id": "BENCH",
            "ticket_id": f"BENCH-{q_idx}",
        }

        t0 = time.time()
        try:
            final_state = app.invoke(initial_state, config=config)
            total = time.time() - t0
            print(f"  Total: {total:.2f}s | Intent: {final_state.get('intent')} | Resolution: {str(final_state.get('resolution', ''))[:80]}")
            per_query.append({
                "query": query,
                "total_s": round(total, 2),
                "intent": final_state.get("intent"),
                "resolution": final_state.get("resolution", "")[:200],
            })
        except Exception as e:
            total = time.time() - t0
            print(f"  ERROR after {total:.2f}s: {str(e)[:100]}")
            per_query.append({
                "query": query,
                "total_s": round(total, 2),
                "error": str(e)[:200],
            })

    totals = [q["total_s"] for q in per_query if "error" not in q]
    errors = [q for q in per_query if "error" in q]

    stats = {
        "label": label,
        "module": main_module_path,
        "queries": len(per_query),
        "successful": len(totals),
        "failed": len(errors),
        "avg_latency_s": round(sum(totals) / len(totals), 2) if totals else None,
        "min_latency_s": round(min(totals), 2) if totals else None,
        "max_latency_s": round(max(totals), 2) if totals else None,
        "total_time_s": round(sum(totals), 2) if totals else None,
        "per_query": per_query,
        "error_details": [e["error"] for e in errors],
    }
    return stats


def print_comparison(results: list[dict]):
    print(f"\n{'='*65}")
    print(f"  LATENCY COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"{'Model':<35} {'Avg':>8} {'Min':>8} {'Max':>8} {'Success':>8} {'Fail':>6}")
    print("-" * 65)
    for r in results:
        avg = f"{r['avg_latency_s']}s" if r.get('avg_latency_s') else "N/A"
        mn = f"{r['min_latency_s']}s" if r.get('min_latency_s') else "N/A"
        mx = f"{r['max_latency_s']}s" if r.get('max_latency_s') else "N/A"
        success = f"{r['successful']}/{r['queries']}"
        fail = str(r['failed'])
        label_short = r["label"][:34]
        print(f"{label_short:<35} {avg:>8} {mn:>8} {mx:>8} {success:>8} {fail:>6}")

    if len(results) == 2:
        r1, r2 = results
        if r1.get('avg_latency_s') and r2.get('avg_latency_s'):
            improvement = (r1['avg_latency_s'] - r2['avg_latency_s']) / r1['avg_latency_s'] * 100
            print("-" * 65)
            print(f"{'Improvement':<35} {f'{improvement:.1f}% faster':>42}")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = []

    # Run v1: Gemma 4 31B (original)
    v1_path = os.path.join(base, "main_gemma_31b_v1.py")
    if os.path.exists(v1_path):
        stats = run_benchmark("Gemma 4 31B (Google AI Studio)", v1_path)
        results.append(stats)
    else:
        print(f"WARNING: {v1_path} not found, skipping v1 benchmark")

    # Run v2: DeepSeek V4 Flash (current)
    v2_path = os.path.join(base, "main_deepseek_v2.py")
    if os.path.exists(v2_path):
        stats = run_benchmark("DeepSeek V4 Flash Free (OpenCode Zen)", v2_path)
        results.append(stats)
    else:
        print(f"WARNING: {v2_path} not found, skipping v2 benchmark")

    if results:
        print_comparison(results)

    # Save results
    out_path = os.path.join(base, "data", "latency_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
