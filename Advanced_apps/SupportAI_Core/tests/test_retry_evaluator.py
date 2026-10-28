"""
Test: Retry logic for openevals evaluators on 429 rate limits.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return '429' in msg or 'rate limit' in msg or 'too many requests' in msg


def retry_evaluator(evaluator, max_retries=5, base_delay=1.0):
    """Wrap an openevals evaluator with retry logic for 429 errors."""
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(max_retries):
            try:
                return evaluator(*args, **kwargs)
            except Exception as e:
                if not is_rate_limit_error(e):
                    raise
                last_error = e
                delay = base_delay * (2 ** attempt)
                print(f"  [RETRY] 429 rate limit, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
        raise last_error
    return wrapper


# --- Failing test ---
class _Mock429(Exception):
    pass

def test_retry_succeeds_after_429s():
    call_count = [0]

    def mock_evaluator(inputs, outputs, reference_outputs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise _Mock429("429 Too Many Requests")
        return {"score": 1, "key": "test"}

    wrapped = retry_evaluator(mock_evaluator, max_retries=5, base_delay=0.01)
    result = wrapped(inputs={}, outputs={}, reference_outputs={})
    
    assert result["score"] == 1, f"Expected score=1, got {result}"
    assert call_count[0] == 3, f"Expected 3 calls, got {call_count[0]}"
    print(f"  PASS: retry succeeded after {call_count[0]} attempts")


def test_retry_fails_after_exhaustion():
    call_count = [0]

    def always_failing(inputs, outputs, reference_outputs):
        call_count[0] += 1
        raise _Mock429("429 Too Many Requests")

    wrapped = retry_evaluator(always_failing, max_retries=3, base_delay=0.01)
    
    try:
        wrapped(inputs={}, outputs={}, reference_outputs={})
        assert False, "Expected RateLimitError"
    except _Mock429:
        assert call_count[0] == 3, f"Expected 3 calls, got {call_count[0]}"
        print(f"  PASS: correctly failed after {call_count[0]} attempts")


def test_retry_passes_through_other_errors():
    def broken(inputs, outputs, reference_outputs):
        raise ValueError("Something else broke")

    wrapped = retry_evaluator(broken, max_retries=3, base_delay=0.01)
    
    try:
        wrapped(inputs={}, outputs={}, reference_outputs={})
        assert False, "Expected ValueError"
    except ValueError:
        print(f"  PASS: non-429 errors pass through immediately")


if __name__ == "__main__":
    test_retry_succeeds_after_429s()
    test_retry_fails_after_exhaustion()
    test_retry_passes_through_other_errors()
    print("\nAll tests passed!")
