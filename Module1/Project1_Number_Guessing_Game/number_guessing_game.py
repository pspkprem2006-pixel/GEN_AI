"""Number Guessing Game -- Module 1, Project 1.

The computer secretly picks a number between 1 and 100. The player keeps
guessing; after each guess the program hints "too high" or "too low" and
counts attempts. When the player guesses correctly it congratulates them
and shows how many tries it took.

Concepts: random module, while loop, if/elif/else, input(), type
conversion, try/except, break, f-strings, counters.

This is the complete program from module notes section 12.3, plus the
extra touches mentioned in the notes: a best-score tracker across rounds
and a graceful Ctrl+C exit. Console output is plain ASCII (no emoji) so
the game runs on every computer. Run with `--demo` for a fully automatic
sample game (used for verification).
"""
import random
import sys


def play_game():
    """Run one round of the Number Guessing Game. Returns the attempts used."""
    secret_number = random.randint(1, 100)
    attempts = 0

    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.\n")

    while True:
        try:
            guess = int(input("Enter your guess: "))
        except ValueError:
            print("[!] Please enter a valid whole number.\n")
            continue

        if guess < 1 or guess > 100:
            print("[!] Out of range! Guess between 1 and 100.\n")
            continue

        attempts += 1

        if guess < secret_number:
            print("Too low!  Try higher.\n")
        elif guess > secret_number:
            print("Too high! Try lower.\n")
        else:
            print(f"\nCorrect! The number was {secret_number}.")
            print(f"You guessed it in {attempts} attempts.")
            break

    return attempts


def main():
    """Main loop: play rounds, track the best score, allow replay."""
    best_score = None

    while True:
        try:
            attempts = play_game()
        except KeyboardInterrupt:
            print("\n[!] Game aborted. Thanks for playing!")
            sys.exit(0)

        if best_score is None or attempts < best_score:
            best_score = attempts
            print(f"[Best] New best score: {best_score} attempts.")
        else:
            print(f"[Best] Best score so far: {best_score} attempts.")

        try:
            again = input("\nPlay again? (yes/no): ").lower()
        except KeyboardInterrupt:
            print("\n[!] Game aborted. Thanks for playing!")
            sys.exit(0)

        if again != "yes":
            print("Thanks for playing!")
            break


def _run_demo():
    """Non-interactive demo: play one full round with auto-answers.

    `random.randint` is pinned to a fixed secret so the scripted guesses
    always produce a complete, deterministic sample run.
    """
    import builtins

    real_randint = random.randint
    random.randint = lambda a, b: 62

    answers = iter(["50", "75", "62", "no"])
    real_input = builtins.input

    def fake_input(prompt=""):
        try:
            value = next(answers)
        except StopIteration:
            value = "no"
        print(prompt + value)
        return value

    builtins.input = fake_input
    try:
        main()
    finally:
        builtins.input = real_input
        random.randint = real_randint


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _run_demo()
    else:
        main()
