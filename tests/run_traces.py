"""
Replays the provided sample conversation traces (GenAI_SampleConversations/*.md)
against a running /chat endpoint.

This does NOT reproduce the real evaluator (which uses an LLM-simulated user
that can answer out of order / refuse / correct itself). It replays the
literal user turns recorded in each trace, which is a reasonable proxy for
"does my agent behave sanely on realistic conversations" during development.

Usage:
  python3 tests/run_traces.py --base-url http://localhost:8000 --traces-dir sample_conversations/GenAI_SampleConversations
"""
import argparse
import re
import sys
import time
from pathlib import Path
import requests


def parse_trace(path: Path):
    """Extract (user_turns, expected_names_by_turn) from a trace .md file."""
    text = path.read_text(encoding="utf-8")
    turns = re.split(r"### Turn \d+", text)[1:]

    user_turns = []
    expected_names_last_turn = []

    for turn in turns:
        user_match = re.search(r"\*\*User\*\*\s*\n\s*>\s*(.+)", turn)
        if user_match:
            user_turns.append(user_match.group(1).strip())

        # expected assessment names from the markdown table, if present
        table_rows = re.findall(r"\|\s*\d+\s*\|\s*(.+?)\s*\|", turn)
        if table_rows:
            expected_names_last_turn = [r.strip() for r in table_rows]

    return user_turns, expected_names_last_turn


def run_trace(base_url, user_turns, delay_seconds=0):
    messages = []
    last_response = None
    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        resp = requests.post(f"{base_url}/chat", json={"messages": messages}, timeout=30)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        last_response = data
        messages.append({"role": "assistant", "content": data.get("reply", "")})
        if delay_seconds:
            time.sleep(delay_seconds)
    return last_response, None


def score_recall(expected_names, got_recommendations):
    if not expected_names:
        return None
    got_names_lower = {r["name"].lower() for r in got_recommendations}
    hits = sum(1 for name in expected_names if name.lower() in got_names_lower)
    return hits / len(expected_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--traces-dir", default="sample_conversations/GenAI_SampleConversations")
    ap.add_argument("--delay", type=float, default=2.0,
                     help="Seconds to sleep between /chat calls, to stay under free-tier RPM limits")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    trace_files = sorted(traces_dir.glob("*.md"))
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        sys.exit(1)

    print(f"Health check: ", end="")
    try:
        h = requests.get(f"{args.base_url}/health", timeout=10)
        print(h.status_code, h.json())
    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)

    recalls = []
    for tf in trace_files:
        user_turns, expected_names = parse_trace(tf)
        response, error = run_trace(args.base_url, user_turns, delay_seconds=args.delay)
        if error:
            print(f"{tf.name}: ERROR - {error}")
            time.sleep(args.delay)
            continue

        recs = response.get("recommendations", [])
        recall = score_recall(expected_names, recs)
        if recall is not None:
            recalls.append(recall)

        print(f"{tf.name}: recs={len(recs)} eoc={response.get('end_of_conversation')} "
              f"recall={f'{recall:.2f}' if recall is not None else 'n/a'}")
        print(f"   reply: {response.get('reply', '')[:150]}")
        if recs:
            print(f"   got:      {[r['name'] for r in recs]}")
        if expected_names:
            print(f"   expected: {expected_names}")

    if recalls:
        print(f"\nMean recall across {len(recalls)} traces: {sum(recalls)/len(recalls):.3f}")


if __name__ == "__main__":
    main()
