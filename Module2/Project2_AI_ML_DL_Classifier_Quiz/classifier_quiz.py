"""AI vs ML vs DL Classifier Quiz.

A short interactive quiz that hardens your understanding of Module 2's core
concepts (AI vs ML vs DL, the three learning paradigms, Generative vs
Predictive AI, ANI/AGI/ASI, and the 2026 AI landscape). Each question shows
a real-world scenario; you classify it and the quiz EXPLAINS the right
answer, so it teaches while it tests.

The 10 questions are shuffled on every run and you get a final score with
a rating.

Usage:
    python classifier_quiz.py            interactive quiz (type a/b/c/d)
    python classifier_quiz.py --demo     non-interactive demo: auto-answers
                                         every question correctly (10/10)
"""

import random
import sys

LETTERS = ["a", "b", "c", "d"]

# Each question: "scenario" is the real-world case, "options" is the
# choice list, "answer" is the 0-based index of the correct option,
# "why" is the teaching explanation shown after you answer.
QUESTIONS = [
    {
        "scenario": "A model is shown 50,000 emails already labelled 'spam' or "
                    "'not spam' and learns to label new emails on its own.",
        "options": ["Rule-based AI", "Supervised Machine Learning",
                    "Unsupervised Learning", "Reinforcement Learning"],
        "answer": 1,
        "why": "Learning from LABELLED examples (spam / not spam) is "
               "SUPERVISED Machine Learning - specifically classification.",
    },
    {
        "scenario": "A robot learns to walk by trying different movements, "
                    "falling, and receiving rewards whenever it stays upright.",
        "options": ["Supervised Learning", "Unsupervised Learning",
                    "Reinforcement Learning", "Rule-based AI"],
        "answer": 2,
        "why": "Trial and error plus rewards is REINFORCEMENT Learning - an "
               "agent learns by interacting with an environment.",
    },
    {
        "scenario": "A streaming service groups viewers with similar tastes "
                    "into segments, without being told any group names in "
                    "advance.",
        "options": ["Supervised Learning (classification)",
                    "Unsupervised Learning (clustering)",
                    "Reinforcement Learning", "Rule-based AI"],
        "answer": 1,
        "why": "Finding hidden structure in UNLABELED data is UNSUPERVISED "
               "Learning - specifically clustering.",
    },
    {
        "scenario": "An AI system writes a brand-new poem about the ocean "
                    "when you ask it to.",
        "options": ["Predictive AI", "Generative AI",
                    "Rule-based AI", "Reinforcement Learning"],
        "answer": 1,
        "why": "CREATING new content (text, images, code) is GENERATIVE AI. "
               "Predictive/discriminative AI classifies or predicts instead.",
    },
    {
        "scenario": "A thermostat turns the heating on whenever the "
                    "temperature drops below 18 degrees C, using a fixed "
                    "if/then rule written by a human.",
        "options": ["Deep Learning", "Unsupervised Learning",
                    "Rule-based AI", "Reinforcement Learning"],
        "answer": 2,
        "why": "A human-written fixed rule with no learning from data is "
               "RULE-BASED (classic) AI.",
    },
    {
        "scenario": "A many-layered neural network recognizes handwritten "
                    "digits from millions of labelled images.",
        "options": ["Rule-based AI", "Classic Machine Learning",
                    "Deep Learning", "Unsupervised Learning"],
        "answer": 2,
        "why": "A multi-layer neural network that learns the useful features "
               "by itself from images is DEEP Learning - a subset of ML.",
    },
    {
        "scenario": "A system predicts tomorrow's temperature as a NUMBER "
                    "from years of weather history.",
        "options": ["Unsupervised Learning",
                    "Supervised Learning (regression)",
                    "Reinforcement Learning", "Generative AI"],
        "answer": 1,
        "why": "Predicting a NUMBER from labelled history is SUPERVISED "
               "Learning - specifically regression. Classification predicts a "
               "category instead.",
    },
    {
        "scenario": "An LLM such as ChatGPT is best described, in terms of "
                    "capability, as:",
        "options": ["AGI (Artificial General Intelligence)",
                    "ASI (Artificial Super Intelligence)",
                    "ANI (Artificial Narrow Intelligence)",
                    "Self-aware AI"],
        "answer": 2,
        "why": "Every AI that exists in 2026 - including ChatGPT - is NARROW "
               "AI (ANI): astonishingly capable but specialized at particular "
               "tasks. AGI does not exist yet.",
    },
    {
        "scenario": "A bank scores a credit-card transaction as fraud or not "
                    "in milliseconds, based on historical labelled "
                    "transactions.",
        "options": ["Unsupervised Learning (clustering)",
                    "Generative AI",
                    "Supervised Learning (classification)",
                    "Reinforcement Learning"],
        "answer": 2,
        "why": "Assigning a CATEGORY (fraud / not fraud) from labelled "
               "examples is SUPERVISED Learning - classification.",
    },
    {
        "scenario": "A tool plans and executes multi-step tasks on its own, "
                    "using browsers, code, and APIs to complete a goal.",
        "options": ["AI Agent", "Rule-based system", "Chatbot",
                    "Clustering algorithm"],
        "answer": 0,
        "why": "An AI that TAKES ACTIONS with tools to complete multi-step "
               "goals is an AI AGENT - the biggest 2024-2026 shift from "
               "chatbots (which only talk) to AI that acts.",
    },
]


def ask_question(q: dict, number: int, total: int, demo: bool = False) -> bool:
    """Ask one question and return True if it was answered correctly."""
    print(f"\nQuestion {number} of {total}")
    print(q["scenario"])
    for i, option in enumerate(q["options"]):
        print(f"   {LETTERS[i]}) {option}")

    if demo:
        choice = LETTERS[q["answer"]]
        print(f"(demo) auto-answer: {choice}")
    else:
        while True:  # keep asking until valid input
            choice = input("Your answer (a/b/c/d): ").strip().lower()
            if choice in LETTERS:
                break
            print("[!] Please type one of: a, b, c, d.")

    picked = LETTERS.index(choice)  # letter -> index (a=0, b=1, ...)
    if picked == q["answer"]:
        print("[CORRECT]")
        correct = True
    else:
        print(f"[WRONG] Correct: {q['options'][q['answer']]}")
        correct = False
    print(f"Why: {q['why']}")
    return correct


def rating(score: int, total: int) -> str:
    """Return a one-line rating for a quiz score."""
    fraction = score / total
    if fraction >= 0.8:
        return "Excellent - you have mastered the AI vs ML vs DL concepts!"
    if fraction >= 0.6:
        return "Good - review the module notes for the questions you missed."
    return "Keep practicing - re-read the module notes and try again."


def main() -> None:
    demo = "--demo" in sys.argv
    if demo:
        print("[!] Demo mode: auto-answering every question correctly.")
    questions = list(QUESTIONS)
    random.shuffle(questions)  # shuffle the question bank each run
    total = len(questions)
    score = 0
    for number, q in enumerate(questions, start=1):
        if ask_question(q, number, total, demo=demo):
            score += 1
    print("\n=== FINAL SCORE ===")
    print(f"You scored {score} out of {total}.")
    print(rating(score, total))


if __name__ == "__main__":
    main()
