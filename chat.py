#!/usr/bin/env python3
"""
Interactive query shell for MedRAG.
Loads the pipeline once, then accepts questions until you type 'exit'.

Usage:
    set -a && source .env && set +a
    python chat.py
"""

import sys
import textwrap

def print_result(result: dict) -> None:
    width = 72
    print("\n" + "─" * width)
    print(f"  Answer     : {result['answer']}")
    print(f"  Confidence : {result.get('confidence', 'N/A')}")
    print(f"  Support    : {result['support_score']:.3f}")
    print(f"  Iterations : {result['iterations']}")
    print(f"  Latency    : {result['latency']:.2f}s")

    rationale = result.get("rationale", [])
    if rationale:
        print(f"\n  Rationale  :")
        for i, step in enumerate(rationale, 1):
            wrapped = textwrap.fill(step, width=width - 16,
                                    subsequent_indent=" " * 16)
            print(f"    {i}. {wrapped}")

    citations = result.get("citations", [])
    if citations:
        print(f"\n  Citations  :")
        for c in citations[:3]:
            wrapped = textwrap.fill(str(c)[:160], width=width - 16,
                                    subsequent_indent=" " * 16)
            print(f"    • {wrapped}")
    print("─" * width + "\n")


def main() -> None:
    print("Loading MedRAG pipeline (this takes ~10s after first run)...")
    try:
        from src.pipeline import MedRAGPipeline
        pipeline = MedRAGPipeline()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed to load: {e}")
        sys.exit(1)

    print("\nMedRAG ready. Type your medical question, or:")
    print("  'exit' / 'quit' / Ctrl-C  →  quit")
    print("  'detail'                  →  toggle detailed history on/off\n")

    show_detail = False

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if question.lower() == "detail":
            show_detail = not show_detail
            print(f"  Detail mode {'ON' if show_detail else 'OFF'}")
            continue

        try:
            result = pipeline.query(question, return_details=show_detail)
            print_result(result)
        except Exception as e:
            print(f"  [ERROR] {e}\n")


if __name__ == "__main__":
    main()
