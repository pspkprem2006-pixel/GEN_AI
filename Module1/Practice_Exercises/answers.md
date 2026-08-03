# Practice Exercises & Self-Assessment - Module 1

Every exercise from section 17 of the module notes, with the question
written out and the answer taken from the module's 17.8 answer key
(verified code). There is often more than one correct way - if yours
works, it's right.

---

## 17.1 Warm-up (basics, variables, operators)

### Exercise 1
**Question:** Print your name, age, and favorite AI tool on three lines using one `print()` with `\n`.

**Answer:**
```python
print("Name: Aarav\nAge: 20\nFavorite AI tool: Claude")
```

### Exercise 2
**Question:** Ask the user for two numbers and print their sum, difference, product, and quotient.

**Answer:**
```python
a = float(input("First number: "))
b = float(input("Second number: "))
print("Sum:", a + b, "| Diff:", a - b, "| Product:", a * b, "| Quotient:", a / b)
```

### Exercise 3
**Question:** Given `radius = 7`, compute a circle's area (`π r²`) using `PI = 3.14159`, rounded to 2 decimals.

**Answer:**
```python
PI = 3.14159
radius = 7
print("Area:", round(PI * radius ** 2, 2))          # -> 153.94
```

### Exercise 4
**Question:** Ask for a temperature in Celsius and convert it to Fahrenheit (`F = C × 9/5 + 32`).

**Answer:**
```python
c = float(input("Celsius: "))
print("Fahrenheit:", c * 9 / 5 + 32)
```

### Exercise 5
**Question:** Swap the values of two variables `a` and `b` without a third variable. *(Hint: `a, b = b, a`.)*

**Answer:**
```python
a, b = 5, 9
a, b = b, a
print(a, b)                                          # -> 9 5
```

---

## 17.2 Conditions & loops

### Exercise 6
**Question:** Ask for a number and print whether it is positive, negative, or zero.

**Answer:**
```python
n = float(input("Number: "))
print("positive" if n > 0 else "negative" if n < 0 else "zero")
```

### Exercise 7
**Question:** Print all even numbers from 1 to 50 using a loop.

**Answer:**
```python
for n in range(2, 51, 2):
    print(n, end=" ")
```

### Exercise 8
**Question:** Take a number `n` and print its multiplication table (1-10).

**Answer:**
```python
n = int(input("Number: "))
for i in range(1, 11):
    print(f"{n} x {i} = {n * i}")
```

### Exercise 9
**Question:** Ask for a year and determine if it is a leap year. *(Divisible by 4, but not by 100 unless also by 400.)*

**Answer:**
```python
y = int(input("Year: "))
is_leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
print("Leap year" if is_leap else "Not a leap year")
```

### Exercise 10
**Question:** Print the first 15 numbers of the Fibonacci sequence (0, 1, 1, 2, 3, 5, ...).

**Answer:**
```python
a, b = 0, 1
for _ in range(15):
    print(a, end=" ")
    a, b = b, a + b        # -> 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377
```

### Exercise 11
**Question:** Ask for a password until the user enters `"python123"` (use a `while` loop).

**Answer:**
```python
while input("Password: ") != "python123":
    print("Wrong, try again.")
print("Access granted!")
```

### Exercise 12
**Question:** Count how many vowels are in a word the user types.

**Answer:**
```python
word = input("Word: ").lower()
count = sum(1 for ch in word if ch in "aeiou")
print("Vowels:", count)
```

---

## 17.3 Functions

### Exercise 13
**Question:** Write `is_prime(n)` that returns `True`/`False` for whether `n` is prime.

**Answer:**
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):   # only check up to sqrt(n)
        if n % i == 0:
            return False
    return True
```

### Exercise 14
**Question:** Write `factorial(n)` using a loop (`5 -> 120`).

**Answer:**
```python
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result                            # factorial(5) -> 120
```

### Exercise 15
**Question:** Write `max_of_three(a, b, c)` that returns the largest - without using `max()`.

**Answer:**
```python
def max_of_three(a, b, c):
    biggest = a
    if b > biggest: biggest = b
    if c > biggest: biggest = c
    return biggest
```

### Exercise 16
**Question:** Write `count_words(sentence)` that returns how many words a sentence has.

**Answer:**
```python
def count_words(sentence):
    return len(sentence.split())             # "the cat sat" -> 3
```

---

## 17.4 Collections

### Exercise 17
**Question:** Given `[4, 2, 8, 6, 2, 8, 4]`, print the list of unique values (use a set).

**Answer:**
```python
nums = [4, 2, 8, 6, 2, 8, 4]
print(sorted(set(nums)))                     # -> [2, 4, 6, 8]
```

### Exercise 18
**Question:** Build a dictionary mapping 5 students to their marks, then print the topper.

**Answer:**
```python
marks = {"Aarav": 90, "Diya": 85, "Kabir": 92, "Meera": 78, "Sam": 88}
topper = max(marks, key=marks.get)           # key with the highest value
print("Topper:", topper, "with", marks[topper])   # -> Kabir with 92
```

### Exercise 19
**Question:** Given two lists `names` and `ages`, combine them into one dictionary. *(Hint: `zip()`.)*

**Answer:**
```python
names = ["Ann", "Bob", "Cara"]
ages = [20, 21, 19]
print(dict(zip(names, ages)))                # -> {'Ann': 20, 'Bob': 21, 'Cara': 19}
```

### Exercise 20
**Question:** Count how many times each character appears in a string, using a dictionary.

**Answer:**
```python
text = "hello"
counts = {}
for ch in text:
    counts[ch] = counts.get(ch, 0) + 1
print(counts)                                # -> {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

---

## 17.5 File & exception handling

### Exercise 21
**Question:** Write your 5 favorite AI tools to a file, one per line, then read and print them.

**Answer:**
```python
tools = ["ChatGPT", "Claude", "Gemini", "Copilot", "Midjourney"]
with open("tools.txt", "w") as f:
    for t in tools:
        f.write(t + "\n")
with open("tools.txt", "r") as f:
    for line in f:
        print(line.strip())
```

### Exercise 22
**Question:** Modify the temperature converter to catch invalid (non-numeric) input.

**Answer:**
```python
try:
    c = float(input("Celsius: "))
    print("Fahrenheit:", c * 9 / 5 + 32)
except ValueError:
    print("Please enter a valid number.")
```

### Exercise 23
**Question:** Write a program that safely divides two user numbers, handling division by zero.

**Answer:**
```python
try:
    a = float(input("Numerator: "))
    b = float(input("Denominator: "))
    print("Result:", a / b)
except ZeroDivisionError:
    print("Cannot divide by zero.")
except ValueError:
    print("Please enter valid numbers.")
```

---

## 17.6 Mini-projects (integration)

### Exercise 24
**Question:** **To-Do List app**: add/view/remove tasks, saved to a text file (menu-driven).

**Answer:**
```python
import os
TODO_FILE = "todo.txt"

def load_tasks():
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

def save_tasks(tasks):
    with open(TODO_FILE, "w") as f:
        for t in tasks:
            f.write(t + "\n")

def todo_app():
    tasks = load_tasks()                       # remembers tasks between runs
    while True:
        print("\n1) Add  2) View  3) Remove  4) Quit")
        choice = input("Choose: ").strip()
        if choice == "1":
            tasks.append(input("New task: ").strip())
            save_tasks(tasks)
        elif choice == "2":
            for i, t in enumerate(tasks, 1):
                print(f"{i}. {t}")
            if not tasks:
                print("(no tasks yet)")
        elif choice == "3":
            num = int(input("Task number to remove: "))
            if 1 <= num <= len(tasks):
                tasks.pop(num - 1)
                save_tasks(tasks)
            else:
                print("No task with that number.")
        elif choice == "4":
            break

# todo_app()   # uncomment to run
```

### Exercise 25
**Question:** **Simple calculator**: menu with +, -, x, /, with error handling for /0.

**Answer:**
```python
def calculator():
    a = float(input("First number: "))
    op = input("Operator (+ - * /): ").strip()
    b = float(input("Second number: "))
    if op == "+": print(a + b)
    elif op == "-": print(a - b)
    elif op == "*": print(a * b)
    elif op == "/":
        print(a / b if b != 0 else "Error: cannot divide by zero")
    else: print("Unknown operator")
```

### Exercise 26
**Question:** **Word frequency counter**: read a text file, output the top 5 most common words.

**Answer:**
```python
from collections import Counter
with open("sample.txt", "r") as f:
    words = f.read().lower().split()
for word, freq in Counter(words).most_common(5):
    print(f"{word}: {freq}")
```

---

## 17.7 Quick self-check quiz

**Question:** What is printed by `print(7 // 2, 7 % 2)`?

**Answer:** `3 1` (floor division, then remainder).

**Question:** What type does `input()` always return?

**Answer:** `str` (string) - convert with `int()` / `float()` when needed.

**Question:** Which collection guarantees unique items?

**Answer:** set.

**Question:** What's the difference between `==` and `is`?

**Answer:** `==` compares **value** (are the contents equal?); `is` compares
**identity** (are they the same object in memory?).

**Question:** What does `range(2, 10, 2)` produce?

**Answer:** 2, 4, 6, 8 (start at 2, step 2, stop before 10).

**Question:** Why use `with open(...)` instead of `open(...)`?

**Answer:** `with` **auto-closes the file** when the block ends - even if an
exception is raised - so you can't leak open file handles.

**Question:** What runs in a `finally` block?

**Answer:** Always - whether or not an error occurred. It runs both on
success and on exception.

**Question:** What's the difference between `print` and `return`?

**Answer:** `print` **shows** a value on screen (and discards it); `return`
**gives the value back** to the caller so it can be stored and used.
