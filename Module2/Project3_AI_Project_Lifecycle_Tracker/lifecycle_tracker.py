"""AI Project Lifecycle Tracker.

A tracker that makes the 7-stage AI Project Lifecycle (Module 2, section 6)
tangible: you pick a project, then for each of the 7 lifecycle stages you
set a status (Not Started / In Progress / Done) and notes. The tool shows a
text progress bar and a completion percentage, and saves the project to
ai_project.json.

The 7 stages:
    1. Problem Definition        5. Model Building & Training
    2. Data Collection           6. Model Evaluation
    3. Data Preparation & Cleaning  7. Deployment & Monitoring
    4. EDA & Feature Engineering

Usage:
    python lifecycle_tracker.py            interactive menu
    python lifecycle_tracker.py --demo     non-interactive demo that builds
                                           the sample "Churn Predictor"
                                           project and shows its progress
"""

import json
import os
import sys

STAGES = [
    "Problem Definition", "Data Collection", "Data Preparation & Cleaning",
    "EDA & Feature Engineering", "Model Building & Training",
    "Model Evaluation", "Deployment & Monitoring",
]

# Each status is worth a fraction of "done":
STATUS_WEIGHT = {"Not Started": 0.0, "In Progress": 0.5, "Done": 1.0}

STATUSES = ["Not Started", "In Progress", "Done"]

DATA_FILE = "ai_project.json"


def new_project(name: str) -> dict:
    """Create a project with every stage 'Not Started'."""
    return {
        "name": name,
        "stages": [
            {"stage": stage, "status": "Not Started", "notes": ""}
            for stage in STAGES                  # a list comprehension!
        ],
    }


def completion_percent(project: dict) -> int:
    """Overall completion as a whole-number percentage (0-100)."""
    weights = [STATUS_WEIGHT[s["status"]] for s in project["stages"]]
    return round(sum(weights) / len(weights) * 100)


def progress_bar(percent: int, width: int = 20) -> str:
    """Build a text bar like [####----------------] for a percentage."""
    filled = round(percent / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def show_project(project: dict) -> None:
    """Print the project's stage table and overall progress."""
    print("=" * 70)
    print(f"PROJECT: {project['name']}")
    print("=" * 70)
    print("#  | Stage                       | Status      | Notes")
    print("----------------------------------------------------------------------")
    for i, s in enumerate(project["stages"], start=1):
        notes = s["notes"] if s["notes"] else "-"
        print(f"{i:<3}| {s['stage']:<28}| {s['status']:<12}| {notes}")
    print("----------------------------------------------------------------------")
    percent = completion_percent(project)
    print(f"Overall progress: {progress_bar(percent)} {percent}%")


def get_choice(options: list, prompt: str) -> str:
    """Ask the user to pick one of the given options (case-insensitive)."""
    while True:
        choice = input(prompt).strip()
        if choice in options:
            return choice
        print(f"[!] Please type one of: {', '.join(options)}.")


def update_stage(project: dict) -> None:
    """Let the user set the status and notes for one lifecycle stage."""
    show_project(project)
    print("\n--- UPDATE A STAGE ---")
    try:
        num = int(input("Stage number (1-7): ").strip())
    except ValueError:
        print("[!] Please type a stage number between 1 and 7.")
        return
    if num < 1 or num > len(project["stages"]):
        print(f"[!] Stage number must be between 1 and {len(project['stages'])}.")
        return
    stage = project["stages"][num - 1]
    print(f"Stage: {stage['stage']}")
    status = get_choice(STATUSES, "Status (Not Started / In Progress / Done): ")
    notes = input("Notes (optional, Enter for none): ").strip()
    stage["status"] = status
    stage["notes"] = notes
    print("[OK] Stage updated.")


def save_project(project: dict) -> None:
    """Save the project to ai_project.json."""
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(project, fh, indent=2)
    print(f"[OK] Saved project '{project['name']}' to {DATA_FILE}.")


def load_project() -> dict:
    """Load the project from ai_project.json, or None if not present."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            print(f"[!] Could not read {DATA_FILE}.")
    return None


def run_menu() -> None:
    """Show the interactive menu until the user quits."""
    project = load_project()
    if project is None:
        name = input("Project name (e.g. Churn Predictor): ").strip() or "My AI Project"
        project = new_project(name)
        print(f"[OK] Created project '{project['name']}' with all stages Not Started.")
    while True:
        print("\n=== AI PROJECT LIFECYCLE TRACKER ===")
        print(f"Current project: {project['name']}")
        print("1. View project and progress")
        print("2. Update a stage (status + notes)")
        print("3. New project")
        print("4. Save project to JSON")
        print("5. Exit")
        choice = input("Your choice (1-5): ").strip()
        if choice == "1":
            print()
            show_project(project)
        elif choice == "2":
            update_stage(project)
        elif choice == "3":
            name = input("New project name: ").strip() or "My AI Project"
            project = new_project(name)
            print(f"[OK] Created project '{project['name']}'.")
        elif choice == "4":
            save_project(project)
        elif choice == "5":
            save_project(project)
            print("[OK] Goodbye.")
            break
        else:
            print("[!] Please type a number from 1 to 5.")


def demo() -> None:
    """Build the sample 'Churn Predictor' project and show its progress."""
    print("[!] Demo mode: building the sample 'Churn Predictor' project.")
    project = new_project("Churn Predictor")
    project["stages"][0]["status"] = "Done"
    project["stages"][0]["notes"] = "Defined goal: predict churn"
    project["stages"][1]["status"] = "In Progress"
    project["stages"][1]["notes"] = "Pulling 2yr data"
    show_project(project)
    print("\n(1 stage Done = 100% + 1 In Progress = 50%, out of 7 stages "
          "-> 1.5/7 = 21%.)")
    print("(Demo only - nothing was saved.)")


def main() -> None:
    if "--demo" in sys.argv:
        demo()
        return
    run_menu()


if __name__ == "__main__":
    main()
