"""File Processing -- Module 1, Project 3.

Reads a file of student marks (CSV), computes statistics (total, average,
highest, lowest, pass/fail count), and writes a clean report to a new
file -- plus a simple text-based bar chart of grades.

Concepts: csv module, file reading/writing, `with open`, list
comprehensions, built-ins (sum, len, max, min), dictionaries, tuples.

This is the complete pipeline from module notes sections 14.2-14.6, with
the extra touch mentioned in the notes: the sample `marks.csv` is
auto-created (only when missing) so the program runs on a fresh checkout.
Console output is plain ASCII.
"""
import csv
import os


SAMPLE_DATA = [
    ["Name", "Marks"],
    ["Aarav", 92], ["Ayesha", 88], ["Rahul", 47],
    ["Sneha", 76], ["Arjun", 34], ["Priya", 90],
]


def ensure_sample_data(filename="marks.csv"):
    """Create the sample marks.csv if it does not exist yet."""
    if os.path.exists(filename):
        return
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(SAMPLE_DATA)   # writerows writes ALL rows at once
    print(f"[OK] Sample data file '{filename}' created.")


def read_marks(filename):
    """Read a CSV of (Name, Marks) -> return a list of (name, marks) tuples."""
    records = []
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader)                     # skip the header row
            for row in reader:
                name = row[0]
                marks = int(row[1])
                records.append((name, marks))
    except FileNotFoundError:
        print(f"[X] File '{filename}' not found.")
    return records


def analyze(records):
    """Compute statistics from the records."""
    if not records:
        return None

    marks_list = [marks for name, marks in records]   # list comprehension
    stats = {
        "count": len(marks_list),
        "total": sum(marks_list),
        "average": round(sum(marks_list) / len(marks_list), 2),
        "highest": max(marks_list),
        "lowest": min(marks_list),
        "passed": len([m for m in marks_list if m >= 40]),
        "failed": len([m for m in marks_list if m < 40]),
    }
    return stats


def write_report(records, stats, filename="report.txt"):
    """Write a formatted analysis report to a text file."""
    with open(filename, "w") as f:
        f.write("=" * 40 + "\n")
        f.write("       STUDENT MARKS REPORT\n")
        f.write("=" * 40 + "\n\n")

        # A simple text bar chart -- one * per 10 marks:
        f.write("Marks Chart (each * = 10 marks):\n")
        for name, marks in records:
            bar = "*" * (marks // 10)
            f.write(f"{name:<10} | {bar} {marks}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key.capitalize():<10}: {value}\n")

    print(f"[OK] Report written to '{filename}'.")


def main():
    ensure_sample_data("marks.csv")
    records = read_marks("marks.csv")
    if not records:
        return
    stats = analyze(records)
    write_report(records, stats)

    # Also print a quick summary to the screen:
    print(f"\nProcessed {stats['count']} students.")
    print(f"Average: {stats['average']} | Highest: {stats['highest']} | Lowest: {stats['lowest']}")
    print(f"Passed: {stats['passed']} | Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
