# Project 1 - Number Guessing Game

The computer secretly picks a random number between 1 and 100. The player
keeps guessing; after each guess the program hints "too low" or "too high"
and counts attempts. When the player guesses correctly it congratulates
them and shows how many tries it took.

This is the complete program from Module 1 notes section 12, plus the
extra touches the notes mention: a **best-score tracker** across rounds
and a graceful **Ctrl+C** exit. Console output is plain ASCII (no emoji)
so it runs on every computer.

**Concepts used:** `random` module, `while` loop, `if/elif/else`,
`input()`, type conversion, `try/except`, `break`, f-strings, counters.

## Files
- `number_guessing_game.py` - the full game

## How to run
```bash
python number_guessing_game.py
```

Type whole numbers between 1 and 100. Non-numeric input and out-of-range
guesses are handled gracefully and do not count as attempts.

### Auto demo (verification)
```bash
python number_guessing_game.py --demo
```
Plays one full round automatically with scripted answers and a fixed
secret number, then exits.

## Challenge extensions
1. Limit the player to 7 guesses (add an attempts cap tied to the `while`
   loop).
2. Add difficulty levels (Easy 1-50, Hard 1-500) using a `match`
   statement.
3. Track the best (lowest) score across rounds using a variable outside
   `play_game()` - already done here; try improving it (e.g. persist it
   to a file).

> Math insight: with a "always guess the middle" strategy (binary search),
> any number 1-100 can be found in at most 7 guesses. This halving idea is
> foundational to algorithms and decision trees in Machine Learning
> (Module 4)!
