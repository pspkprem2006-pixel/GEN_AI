# Module 7 - Practice Exercises & Self-Assessment (Answers)

Answers for section 17 of the Module 7 notes (Generative AI & Prompt
Engineering). Prompting exercises are open-ended - these are example prompts
and answers; try yours in a real chatbot and compare.

## 17.1 Concept checks

1. **How does an LLM generate text?**
   It repeatedly predicts the most likely **next token**, appends it, and
   feeds its own output back in - one token at a time, until it stops.

2. **What is a token, and why does it matter for cost?**
   A token is a word-piece (roughly 3/4 of a word, about 4 characters). It
   matters because APIs bill **per token** (input + output), and the context
   window is measured in tokens.

3. **What is a context window? What happens to text outside it?**
   The maximum amount of text (in tokens) the model can consider at once -
   your prompt plus its reply. Anything outside it is "forgotten": in a long
   chat, the earliest messages can fall out of context.

4. **Name the 4 levers of a great prompt.**
   Role, Task, Examples, Format.

5. **When would you use few-shot over zero-shot?**
   When you need a specific style or format, or the task could be interpreted
   loosely. Showing 1-3 examples "teaches" the model the exact pattern you
   want.

6. **Why does chain-of-thought improve reasoning?**
   Generating the intermediate steps gives the model "room to reason" instead
   of blurting out a possibly-wrong final answer. It stops guessing and
   actually works through the problem.

7. **What is a hallucination, and give two ways to reduce it.**
   A hallucination is confident-but-false output (fake citations, made-up
   facts, non-existent code). To reduce it: (a) ground the model in provided
   facts / use RAG, and (b) ask for sources and verify them yourself. Also:
   lower the temperature, and prompt "say if you are not sure."

8. **Explain RAG in two sentences.**
   Retrieval-Augmented Generation finds the most relevant chunks of *your*
   documents (via embeddings) and pastes them into the prompt. The model then
   answers from that grounded context, which reduces hallucination and lets
   it use private or current data.

## 17.2 Prompting practice (example prompts)

9. **Weak prompt -> strong prompt**
   Weak: "Write about the ocean."
   Strong: "You are a marine biologist. Write a 120-word, engaging paragraph
   for 12-year-olds about why the ocean matters, ending with one surprising
   fact." (added role + audience + length + format + constraint)

10. **Few-shot: country -> capital in a fixed format**
    ```
    France -> Paris
    Japan -> Tokyo
    India -> ?
    ```

11. **Chain-of-thought on a word problem**
    "A train travels 60 km in 45 minutes. What is its speed in km/h? Think
    step by step, then give the final answer." Then try it *without* "step by
    step" and compare - CoT usually gets it right more reliably.

12. **Role prompting - recursion**
    (a) To a 10-year-old: "You are a fun kids' teacher. Explain recursion to
    a 10-year-old using a story."
    (b) To a CS student: "You are a CS professor. Explain recursion to a
    2nd-year student, with the base case, recursive case, and a code example."

13. **Same data, two formats**
    First: "List 3 planets and their moon-counts as JSON with keys name,
    moons." Then: "...now as a markdown table with columns Planet, Moons."

14. **Anti-hallucination prompt**
    "Answer ONLY using the text between the triple quotes. If the answer
    isn't in it, reply exactly 'Not stated.' """<text>"""

## 17.3 Project & API

15. **Run all three projects in mock mode** - done:
    - `Project1_AI_Resume_Generator/resume_generator.py`
    - `Project2_Research_Assistant/research_assistant.py`
    - `Project3_Prompt_Engineering_Lab/prompt_lab.py`

16. **Your own resume** - put your own details in the `PROFILE` dict of
    `resume_generator.py` and run it; the resume is saved to `resume.md`.

17. **(Optional) Real mode** - `pip install anthropic`, set
    `ANTHROPIC_API_KEY`, flip `USE_REAL_API = True`, and compare the AI
    output to the mock. The mock is representative; Claude's real prose is
    smarter and varies per run.

18. **Cost estimate: 1,000 words in + 1,000 words out.**
    1,000 words ~ 1,333 tokens (100 tokens ~ 75 words).

    | Model | Input (1,333 tok) | Output (1,333 tok) | Total per call |
    |---|---|---|---|
    | Opus 5 ($5 / $25)  | $0.0067 | $0.0333 | ~ $0.040 |
    | Sonnet 5 ($3 / $15) | $0.0040 | $0.0200 | ~ $0.024 |
    | Haiku 4.5 ($1 / $5) | $0.0013 | $0.0067 | ~ $0.008 |

    Takeaway: the same task costs ~5x more on Opus than on Haiku - pick the
    smallest model that does the job well.

## 17.4 Quick self-check quiz

1. What does an LLM predict, one step at a time? -> the **next token**
2. Which prompt technique adds examples? -> **few-shot**
3. Which adds "think step by step"? -> **chain-of-thought**
4. What is a confident false answer called? -> **hallucination**
5. What technique grounds answers in your own documents? -> **RAG**
6. Where should an API key live? -> an **environment variable**, never in code
7. Which Claude model is cheapest? -> **Haiku 4.5**
8. Role & rules go in which prompt? -> the **system prompt**
