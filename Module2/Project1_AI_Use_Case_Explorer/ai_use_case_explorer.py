"""AI Use Case Explorer.

A menu-driven tool that catalogs and prioritizes AI use cases using the
Impact-vs-Feasibility framework (Module 2, section 7.3). You record a use
case (industry, problem, AI type, impact 1-5, feasibility 1-5); the tool
ranks them by priority (impact x feasibility) and places each one in a
quadrant:

    Quick Win   = high impact + high feasibility (do these first)
    Big Bet     = high impact + low feasibility (plan carefully)
    Low Priority = low impact + high feasibility (easy but low value)
    Avoid       = low impact + low feasibility

The catalog is pre-loaded with 5 example use cases and is saved to
ai_use_cases.json.

Usage:
    python ai_use_case_explorer.py            interactive menu
    python ai_use_case_explorer.py --demo     non-interactive demo that
                                              prints the prioritized table
                                              of the 5 sample use cases
"""

import json
import os
import sys

HIGH = 4  # a score of 4 or 5 counts as "HIGH"

DATA_FILE = "ai_use_cases.json"

EXAMPLE_CASES = [
    {"industry": "Finance", "problem": "Flag fraudulent card transactions",
     "ai_type": "ML (classification / anomaly detection)", "impact": 5, "feasibility": 4},
    {"industry": "Retail", "problem": "Recommend products to shoppers",
     "ai_type": "ML (recommendation system)", "impact": 4, "feasibility": 5},
    {"industry": "Healthcare", "problem": "Detect tumors in X-ray/CT scans",
     "ai_type": "Deep Learning (computer vision)", "impact": 5, "feasibility": 3},
    {"industry": "Customer Service", "problem": "Answer FAQs with a chatbot",
     "ai_type": "Generative AI / NLP", "impact": 3, "feasibility": 5},
    {"industry": "Agriculture", "problem": "Predict crop disease from leaf photos",
     "ai_type": "Deep Learning (computer vision)", "impact": 4, "feasibility": 2},
]


def priority_label(impact: int, feasibility: int) -> str:
    """Return the Impact-vs-Feasibility quadrant name for a use case."""
    high_impact = impact >= HIGH
    high_feasibility = feasibility >= HIGH
    if high_impact and high_feasibility:
        return "Quick Win"        # do these first
    if high_impact and not high_feasibility:
        return "Big Bet"          # high value but hard - plan carefully
    if not high_impact and high_feasibility:
        return "Low Priority"     # easy but low value
    return "Avoid"                # low value AND hard


def priority_score(case: dict) -> int:
    """A single number used to rank use cases (higher = do sooner)."""
    return case["impact"] * case["feasibility"]


def prioritize(cases: list) -> None:
    """Print all use cases ranked by priority, highest first."""
    ranked = sorted(cases, key=priority_score, reverse=True)  # highest first
    print("=== AI USE CASES BY PRIORITY (highest first) ===")
    print("Rank | Score | Quadrant     | Use Case")
    print("----------------------------------------------------------------------")
    for rank, c in enumerate(ranked, start=1):
        label = priority_label(c["impact"], c["feasibility"])
        score = priority_score(c)
        print(f"{rank:<5}| {score:<6}| {label:<13}| {c['industry']} - {c['problem']}")


def get_int(prompt: str, low: int, high: int) -> int:
    """Ask for a whole number between low and high, re-prompting until valid."""
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print(f"[!] Please type a number between {low} and {high}.")
            continue
        if low <= value <= high:
            return value
        print(f"[!] Please type a number between {low} and {high}.")


def add_case(cases: list) -> None:
    """Prompt for one new use case and append it to the catalog."""
    print("\n--- ADD A USE CASE ---")
    industry = input("Industry (e.g. Healthcare): ").strip()
    problem = input("Problem to solve: ").strip()
    ai_type = input("AI type (e.g. ML, Deep Learning, GenAI): ").strip()
    impact = get_int("Impact (1-5): ", 1, 5)
    feasibility = get_int("Feasibility (1-5): ", 1, 5)
    if not industry or not problem:
        print("[!] Industry and problem cannot be empty - use case not added.")
        return
    cases.append({"industry": industry, "problem": problem,
                  "ai_type": ai_type, "impact": impact,
                  "feasibility": feasibility})
    print("[OK] Use case added.")


def list_cases(cases: list) -> None:
    """Print all use cases in the order they were entered."""
    print("\n=== ALL USE CASES ===")
    if not cases:
        print("(catalog is empty)")
        return
    for i, c in enumerate(cases, start=1):
        print(f"{i}. [{c['industry']}] {c['problem']} "
              f"(impact {c['impact']}, feasibility {c['feasibility']}, "
              f"type: {c['ai_type']})")


def save_cases(cases: list) -> None:
    """Save the catalog to ai_use_cases.json."""
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(cases, fh, indent=2)
    print(f"[OK] Saved {len(cases)} use case(s) to {DATA_FILE}.")


def load_cases() -> list:
    """Load the catalog from ai_use_cases.json, or pre-load the examples."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            print(f"[!] Could not read {DATA_FILE} - starting with examples.")
    return [dict(c) for c in EXAMPLE_CASES]


def run_menu(cases: list) -> None:
    """Show the interactive menu until the user quits."""
    while True:
        print("\n=== AI USE CASE EXPLORER ===")
        print("1. Add a use case")
        print("2. List all use cases")
        print("3. Prioritize (rank by impact x feasibility)")
        print("4. Save catalog to JSON")
        print("5. Exit")
        choice = input("Your choice (1-5): ").strip()
        if choice == "1":
            add_case(cases)
        elif choice == "2":
            list_cases(cases)
        elif choice == "3":
            print()
            prioritize(cases)
        elif choice == "4":
            save_cases(cases)
        elif choice == "5":
            save_cases(cases)
            print("[OK] Goodbye.")
            break
        else:
            print("[!] Please type a number from 1 to 5.")


def main() -> None:
    demo = "--demo" in sys.argv
    if demo:
        print("[!] Demo mode: showing the priority table for the 5 sample use cases.")
        prioritize(EXAMPLE_CASES)
        print("\n(Demo only - nothing was saved.)")
        return
    cases = load_cases()
    run_menu(cases)


if __name__ == "__main__":
    main()
