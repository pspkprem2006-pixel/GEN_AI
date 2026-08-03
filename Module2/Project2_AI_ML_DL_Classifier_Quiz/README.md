# Project 2 - AI vs ML vs DL Classifier Quiz

A short interactive quiz that hardens your understanding of the module's
core concepts. It shows a real-world scenario, you classify it, and it
**explains** the right answer - so it teaches while it tests.

## Files
- `classifier_quiz.py` - the full program

## How it works
A multiple-choice quiz that presents 10 scenarios and asks you to label
each as Rule-based AI, Supervised/Unsupervised/Reinforcement learning,
Deep Learning, Generative vs Predictive AI, ANI/AGI/ASI, or an AI Agent.

- The **question bank** is a list of dictionaries: `scenario`, `options`
  (a list), `answer` (the 0-based index of the correct option), and `why`
  (the teaching explanation shown after you answer).
- `ask_question()` validates input with a `while` loop, numbers options
  with `enumerate`, maps a letter to an index, and returns a Boolean.
- Questions are **shuffled on every run** with `random.shuffle()`, and you
  get a final score with a rating.

## How to run
```bash
python classifier_quiz.py
```
Type `a`, `b`, `c`, or `d` for each question.

Non-interactive demo (auto-answers every question correctly, 10/10):
```bash
python classifier_quiz.py --demo
```

## Challenges
1. Add **2 of your own questions** to the `QUESTIONS` list, following the
   existing dictionary format (practice exercise 20).
2. Aim for 8/10 or higher (practice exercise 19).
3. Extend `LETTERS` and the options to 5 choices per question.

## Sample interaction
```
Question 1 of 10
A model is shown 50,000 emails already labelled 'spam' or 'not spam'
and learns to label new emails on its own.

   a) Rule-based AI
   b) Supervised Machine Learning
   c) Unsupervised Learning
   d) Reinforcement Learning

Your answer (a/b/c/d): b
[CORRECT]
Why: Learning from LABELLED examples (spam / not spam) is SUPERVISED
Machine Learning - specifically classification.
```
