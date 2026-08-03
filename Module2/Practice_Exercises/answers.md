# Module 2 - Practice Exercises & Self-Assessment (Answers)

Every question from section 17 of the module notes is reproduced verbatim
below, followed by the answer (from the module's 17.6 Solutions & Answer
Key, or the inline answers in 17.2 / 17.5).

---

## 17.1 Concept checks (write one or two sentences each)

**1. Define AI, ML, and DL, and explain how they are nested.**

> AI is any technique that makes machines act intelligently. ML is a
> *subset* of AI where machines learn patterns from data instead of being
> hand-coded. DL is a *subset* of ML that uses many-layered neural
> networks. So **DL is a subset of ML, which is a subset of AI** - every
> DL system is ML, every ML system is AI, but not the reverse.

**2. Give one everyday example each of classification, regression, and clustering.**

> *Classification* -> spam vs not-spam email. *Regression* -> predicting
> a house's price. *Clustering* -> grouping customers into segments by
> behavior.

**3. Why is *all* current AI considered "Narrow" (ANI)?**

> Every system today - even ChatGPT/Claude - is specialized at particular
> tasks and cannot generalize across *any* task the way a human can.
> General intelligence (AGI) does not exist yet.

**4. Explain the difference between supervised and unsupervised learning with your own example.**

> Supervised learns from **labeled** data (e.g., emails tagged
> spam/not-spam -> predict the label). Unsupervised finds structure in
> **unlabeled** data (e.g., group shoppers into segments with no
> predefined groups).

**5. What is the difference between Generative AI and Predictive AI?**

> Predictive/discriminative AI **classifies or predicts** ("is this
> spam?", "what's the price?"). Generative AI **creates new content**
> (text, images, code).

**6. Why is data quality more important than data quantity?**

> Models learn patterns *from* the data - "garbage in, garbage out." A
> huge but biased/incorrect dataset teaches wrong patterns, while a
> smaller clean, representative one teaches right ones.

**7. List the 7 stages of the AI project lifecycle in order.**

> (1) Problem Definition, (2) Data Collection, (3) Data
> Preparation/Cleaning, (4) EDA & Feature Engineering, (5) Model Building
> & Training, (6) Model Evaluation, (7) Deployment & Monitoring.

**8. Explain "Garbage In, Garbage Out" in the context of bias.**

> If training data reflects human bias (e.g., biased past hiring), the
> model *learns and amplifies* that bias at scale - with a false air of
> objectivity because "the computer decided."

---

## 17.2 Classify these scenarios (AI type / paradigm)

*For each, name the category (Rule-based AI, Supervised, Unsupervised,
Reinforcement, Deep Learning, Generative):*

**9. Netflix grouping viewers with similar taste. -> ?**

> **Unsupervised (clustering)** - it finds hidden groups in unlabeled
> viewing data with no predefined groups.

**10. Predicting tomorrow's temperature from weather history. -> ?**

> **Supervised (regression)** - it predicts a number (temperature) from
> labeled historical data.

**11. A robot learning to walk by trying and falling. -> ?**

> **Reinforcement** - it learns by trial and error with rewards/penalties
> from the environment.

**12. An AI writing a poem about the ocean. -> ?**

> **Generative** - it creates brand-new content (text) rather than
> classifying or predicting.

**13. A thermostat that turns on heating below 18 degrees C using a fixed rule. -> ?**

> **Rule-based AI** - a human-written fixed if/then rule with no learning
> from data.

**14. Recognizing handwritten digits with a neural network. -> ?**

> **Deep Learning** - a many-layered neural network learns the features
> by itself from images.

---

## 17.3 Applied / discussion tasks

**15. Pick an industry and list 3 realistic AI use cases; score each for impact & feasibility (use Project 1).**

> **Healthcare example:** (a) Detect tumors in scans - *impact 5,
> feasibility 3*; (b) Appointment no-show prediction - *impact 3,
> feasibility 4*; (c) Chatbot for FAQs - *impact 3, feasibility 5*.
> Priority score = impact x feasibility -> the FAQ chatbot (15) and
> no-show predictor (12) are "quick wins," tumor detection (15) is a
> "big bet." (Run `ai_use_case_explorer.py` to see the full ranking.)

**16. Choose one use case and map it through all 7 lifecycle stages (use Project 3).**

> **Lifecycle map (no-show prediction):** *Define* "predict which
> patients miss appointments" -> *Collect* 2 yrs of booking + attendance
> data -> *Prepare* clean dates, fill gaps -> *EDA* find that distance &
> prior no-shows matter -> *Build* a classifier -> *Evaluate* on unseen
> data (recall matters) -> *Deploy* a daily at-risk list + monitor.
> (Run `lifecycle_tracker.py` to track it stage by stage.)

**17. Find one news headline about AI this week and identify: Is it Narrow or General AI? Which paradigm? Any ethical concern?**

> Any current AI story describes **Narrow AI** (not AGI - AGI does not
> exist in 2026). Identify the paradigm (a chatbot = generative; a fraud
> detector = supervised classification) and one ethical angle (bias,
> privacy, misinformation, or job impact). Yours will differ - this is a
> discussion task.

**18. Describe one ethical risk of an AI you use daily, and how you'd reduce it.**

> **Everyday ethical risk (sample):** a recommendation feed can create
> filter bubbles/bias. *Reduce it* by adding diversity to
> recommendations, being transparent, and giving users control over what
> they see. Yours will differ - this is a discussion task.

---

## 17.4 Hands-on

**19. Complete all three module projects and get 8/10 or higher on the quiz (Project 2).**

> Completed against the module's three projects:
> 1. AI Use Case Explorer - `Project1_AI_Use_Case_Explorer/ai_use_case_explorer.py`
> 2. AI vs ML vs DL Classifier Quiz - `Project2_AI_ML_DL_Classifier_Quiz/classifier_quiz.py`
> 3. AI Project Lifecycle Tracker - `Project3_AI_Project_Lifecycle_Tracker/lifecycle_tracker.py`
>
> The quiz scores out of 10; 8/10 or higher is the target (run
> `python classifier_quiz.py --demo` to verify a full 10/10 run).

**20. Add 2 of your own questions to the quiz's question bank.**

> Done by adding new scenario questions to the `QUESTIONS` list in
> `classifier_quiz.py`, following the existing dictionary format:
> `scenario`, `options` (list), `answer` (0-based index), and `why`
> (explanation). Example additions:
>
> ```python
> {
>     "scenario": "A music app finds songs that often get played together "
>                 "after one another, with no labels telling it which songs "
>                 "belong together.",
>     "options": ["Supervised Learning (classification)",
>                 "Unsupervised Learning (association)",
>                 "Reinforcement Learning", "Rule-based AI"],
>     "answer": 1,
>     "why": "Finding 'items that go together' with no labels is "
>            "UNSUPERVISED Learning - association (people who buy bread "
>            "also buy butter).",
> },
> {
>     "scenario": "A self-driving car uses its recent camera frames and "
>                 "sensor readings to decide whether to brake now.",
>     "options": ["Reactive Machine", "Limited Memory AI",
>                 "Theory of Mind", "Self-aware AI"],
>     "answer": 1,
>     "why": "It uses RECENT past data (short-term memory) to make "
>            "decisions, so it is LIMITED MEMORY AI - where almost all "
>            "modern AI lives.",
> },
> ```

---

## 17.5 Quick self-check quiz

**1. Which is broadest: AI, ML, or DL?**

> **AI**

**2. Labeled data is used in which paradigm?**

> **Supervised**

**3. Does AGI exist in 2026?**

> **No**

**4. Predicting a house price is classification or regression?**

> **Regression**

**5. What architecture powers modern LLMs?**

> **Transformer**

**6. What does "hallucination" mean for an AI?**

> **Confidently stating false info**

**7. Roughly what % of data-science time is data cleaning?**

> **~80%**

**8. Name the last stage of the AI lifecycle.**

> **Deployment & Monitoring**
