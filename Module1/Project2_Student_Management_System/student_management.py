"""Student Management System -- Module 1, Project 2.

A menu-driven console app that lets a teacher Add, View, Search, Update,
and Delete student records, and saves everything to a JSON file so data
survives between runs.

Concepts: dictionaries, lists, functions, loops, file handling (JSON),
exception handling, `match` pattern matching, the
`if __name__ == "__main__":` guard.

This is the complete program from module notes sections 13.3 and 13.4,
plus the extra touches mentioned in the notes: a duplicate-roll check and
a clean Ctrl+C exit. Console output is plain ASCII. Run with `--demo` for
a fully automatic sample session (used for verification).
"""
import json
import sys

FILE = "students.json"


def load_students():
    """Load students from the JSON file, or start empty if none exists."""
    try:
        with open(FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []          # first run -- no file yet, so start with an empty list


def save_students(students):
    """Save the current list of students to the JSON file."""
    with open(FILE, "w") as f:
        json.dump(students, f, indent=4)
    print("[saved] Data saved.")


def add_student(students):
    """Add a new student record."""
    try:
        roll = int(input("Roll number: "))
        name = input("Name: ").strip()
        marks = float(input("Marks: "))
    except ValueError:
        print("[!] Invalid input. Roll and marks must be numbers.")
        return
    for s in students:
        if s["roll"] == roll:
            print(f"[X] Roll {roll} already exists. Student not added.")
            return
    students.append({"roll": roll, "name": name, "marks": marks})
    print(f"[OK] Added {name}.")


def view_students(students):
    """Display all students in a neat table."""
    if not students:                     # empty list is falsy
        print("No students yet.")
        return
    print("\nRoll  | Name            | Marks")
    print("-" * 35)
    for s in students:
        print(f"{s['roll']:<5} | {s['name']:<15} | {s['marks']}")


def search_student(students):
    """Find a student by roll number."""
    try:
        roll = int(input("Enter roll number to search: "))
    except ValueError:
        print("[!] Roll must be a number.")
        return
    for s in students:
        if s["roll"] == roll:
            print(f"Found -> Name: {s['name']}, Marks: {s['marks']}")
            return
    print("[X] Student not found.")


def update_student(students):
    """Update a student's marks."""
    try:
        roll = int(input("Enter roll number to update: "))
    except ValueError:
        print("[!] Roll must be a number.")
        return
    for s in students:
        if s["roll"] == roll:
            s["marks"] = float(input("New marks: "))
            print("[OK] Updated.")
            return
    print("[X] Student not found.")


def delete_student(students):
    """Delete a student by roll number."""
    try:
        roll = int(input("Enter roll number to delete: "))
    except ValueError:
        print("[!] Roll must be a number.")
        return
    for s in students:
        if s["roll"] == roll:
            students.remove(s)
            print("[OK] Deleted.")
            return
    print("[X] Student not found.")


def main():
    students = load_students()      # load saved data at startup

    menu = """
====== STUDENT MANAGEMENT SYSTEM ======
1. Add Student
2. View All Students
3. Search Student
4. Update Marks
5. Delete Student
6. Save & Exit
=======================================
"""
    while True:
        print(menu)
        try:
            choice = input("Choose an option (1-6): ").strip()
        except KeyboardInterrupt:
            print("\n[!] Exiting without saving.")
            break

        match choice:                          # modern pattern matching
            case "1": add_student(students)
            case "2": view_students(students)
            case "3": search_student(students)
            case "4": update_student(students)
            case "5": delete_student(students)
            case "6":
                save_students(students)
                print("Goodbye!")
                break
            case _:
                print("[!] Invalid choice. Pick 1-6.")


def _run_demo():
    """Non-interactive demo: walk through every menu option with auto-answers."""
    import builtins

    answers = iter([
        "1", "1", "Aarav", "92",          # add Aarav
        "1", "2", "Ayesha", "88",         # add Ayesha
        "2",                              # view all
        "3", "2",                         # search Ayesha
        "4", "1", "95",                   # update Aarav to 95
        "5", "2",                         # delete Ayesha
        "2",                              # view all again
        "6",                              # save & exit
    ])
    real_input = builtins.input

    def fake_input(prompt=""):
        try:
            value = next(answers)
        except StopIteration:
            value = "6"
        print(prompt + value)
        return value

    builtins.input = fake_input
    try:
        main()
    finally:
        builtins.input = real_input


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _run_demo()
    else:
        main()
