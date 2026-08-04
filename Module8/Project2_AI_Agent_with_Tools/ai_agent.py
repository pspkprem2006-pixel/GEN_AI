"""
Project 2: AI Agent with Tools
Module 8 - AI Agents & Automation

A single-agent loop that demonstrates think → act → observe → answer.
Uses rule-based routing for offline mode. Set USE_REAL_API = True for real LLM.
"""

import math
import datetime
import re

# Configuration
USE_REAL_API = False  # Set True + provide API key for real LLM agent

# Tool definitions
TOOLS = {}


def calculator(args):
    """Perform mathematical calculations."""
    expression = args.get("expression", "")
    if not expression:
        return "Error: No expression provided"
    
    try:
        # Clean up the expression
        expression = expression.strip()
        
        # Safe evaluation using math module
        allowed_names = {
            "abs": abs, "round": round,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "log": math.log, "pi": math.pi, "e": math.e
        }
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def clock(args):
    """Get current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def word_counter(args):
    """Count words in text."""
    text = args.get("text", "")
    words = text.split()
    return len(words)


def knowledge_lookup(args):
    """Look up knowledge (mock database)."""
    topic = args.get("topic", "").lower()
    knowledge_db = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "machine learning": "Machine Learning is a subset of AI that enables systems to learn from data.",
        "deep learning": "Deep Learning is a subset of ML using neural networks with many layers.",
        "nlp": "Natural Language Processing enables computers to understand and generate human language.",
    }
    for key, value in knowledge_db.items():
        if key in topic:
            return value
    return f"No information found for '{topic}'."


# Register tools
TOOLS["calculator"] = calculator
TOOLS["clock"] = clock
TOOLS["word_counter"] = word_counter
TOOLS["knowledge_lookup"] = knowledge_lookup


def choose_tool(goal):
    """Rule-based brain: choose which tool to use based on the goal."""
    goal_lower = goal.lower()
    
    # Check for word counting first (before other checks)
    if any(word in goal_lower for word in ["word count", "count words", "how many words"]):
        # Extract text to count - try different patterns
        text_match = re.search(r'"([^"]*)"', goal)
        if not text_match:
            text_match = re.search(r"'([^']*)'", goal)
        if not text_match:
            # Try to extract after "words in"
            text_match = re.search(r'words?\s+in\s+(.+)', goal_lower)
            if text_match:
                return "word_counter", {"text": text_match.group(1).strip()}
        if text_match:
            return "word_counter", {"text": text_match.group(1)}
    
    # Check for time/date queries
    if any(word in goal_lower for word in ["time", "date", "day", "clock", "now", "today"]):
        return "clock", {}
    
    # Check for knowledge lookup (but not if it contains math operators)
    if any(word in goal_lower for word in ["what is", "tell me about", "explain", "define"]):
        # Don't use knowledge lookup if the goal contains math operators
        if not re.search(r'[\+\-\*\/]', goal):
            # Extract topic
            topic_match = re.search(r'(?:what is|tell me about|explain|define)\s+(.+)', goal_lower)
            if topic_match:
                topic = topic_match.group(1).strip()
                # Remove question marks
                topic = topic.rstrip('?')
                return "knowledge_lookup", {"topic": topic}
    
    # Check for math operations - try to extract math expression
    # Look for patterns like "15 * 23 + 100" or "sqrt(144) + 10"
    
    # Try to find sqrt or other math functions first
    math_func_match = re.search(r'(sqrt\([\d\.\,]+\)(?:\s*[\+\-\*\/]\s*[\d\.\s\+\-\*\/\(\)]*)?)', goal)
    if math_func_match:
        expr = math_func_match.group(1).strip()
        return "calculator", {"expression": expr}
    
    # Try to find explicit math with operators
    math_match = re.search(r'(\d[\d\s\+\-\*\/\(\)\.\,]*[\+\-\*\/][\d\s\+\-\*\/\(\)\.\,]*\d)', goal)
    if math_match:
        expr = math_match.group(1).strip()
        return "calculator", {"expression": expr}
    
    # Default: try to calculate if it looks like math
    if re.search(r'\d\s*[\+\-\*\/]\s*\d', goal):
        return "calculator", {"expression": goal}
    
    return None, None


def run_agent(goal):
    """Run the agent loop with think → act → observe → answer."""
    print(f"GOAL: {goal}")
    
    # THINK: Choose a tool
    tool_name, tool_input = choose_tool(goal)
    
    if tool_name is None:
        print("  THINK  : I don't have a tool for this goal.")
        print("  ANSWER : Sorry, I cannot help with that goal.")
        return
    
    print(f"  THINK  : This needs the '{tool_name}' tool.")
    
    # ACT: Use the tool
    tool_fn = TOOLS[tool_name]
    print(f"  ACT    : {tool_name}({tool_input})")
    
    # OBSERVE: Get the result
    observation = tool_fn(tool_input)
    print(f"  OBSERVE: {observation}")
    
    # ANSWER
    print(f"  ANSWER : The answer is: {observation}")
    print()


def main():
    """Main agent loop with sample goals."""
    print("=" * 60)
    print("PROJECT 2: AI AGENT WITH TOOLS")
    print("=" * 60)
    print(f"Mode: {'Real API' if USE_REAL_API else 'Mock (rule-based)'}")
    print()
    
    # Sample goals to demonstrate the agent
    sample_goals = [
        "What is 15 * 23 + 100?",
        "What time is it now?",
        "How many words are in 'Hello world this is a test'?",
        "What is machine learning?",
        "Calculate sqrt(144) + 10",
    ]
    
    for goal in sample_goals:
        run_agent(goal)
    
    # Interactive mode
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("Enter your goals (type 'quit' to exit):")
    print("=" * 60)
    
    while True:
        try:
            goal = input("\nGoal: ").strip()
            if goal.lower() in ("quit", "exit", ""):
                break
            run_agent(goal)
        except KeyboardInterrupt:
            break
    
    print("\nAgent session ended.")


if __name__ == "__main__":
    main()