# Module 6 - Natural Language Processing

Completed hands-on tasks for Module 6 of the AI Powered Engineering
Upskilling Program. The module covers the NLP pipeline (preprocess ->
vectorize -> model), Bag-of-Words / TF-IDF, word embeddings, text
classification, sequence models, Transformers, and BERT.

## Task Files (one file per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_Spam_Detection/` | Spam Detection - TF-IDF + Multinomial Naive Bayes classifies messages as spam/ham (40 embedded messages), prints the top SPAM-signal words, predicts on new messages, saves a chart. |
| Project 2 | `Project2_Sentiment_Analysis/` | Sentiment Analysis - TF-IDF (unigrams + bigrams, negation words KEPT) + Logistic Regression(C=10) classifies reviews as positive/negative, with a negation test ("not good" vs "not bad"), saves a chart. |
| Project 3 | `Project3_Text_Preprocessing_Toolkit/` | Text Preprocessing Toolkit - every preprocessing step with before/after output (lowercase, clean, tokenize, stop words, stem), then Bag-of-Words and TF-IDF, saves a word-frequency chart. |
| Practice | `Practice_Exercises/answers.md` | Answers for ALL of section 17: 17.1 concept checks, 17.2-17.4 coding exercises (runnable code), 17.5 integrative tasks, and the 17.6 self-check quiz. |

## How to run

Each project runs OFFLINE with only the installed libraries (numpy,
pandas, scikit-learn, matplotlib) - no external dataset files, no API
keys, no downloads. Stop words come from scikit-learn and Project 3's
stemmer is a small built-in one, so no NLTK data is needed.

```bash
# Project 1 - Spam Detection (syllabus activity)
cd Project1_Spam_Detection && python spam_detection.py

# Project 2 - Sentiment Analysis
cd Project2_Sentiment_Analysis && python sentiment_analysis.py

# Project 3 - Text Preprocessing Toolkit
cd Project3_Text_Preprocessing_Toolkit && python preprocessing_toolkit.py
```

Do them **3 -> 1 -> 2** (Module 6, section 14.2): understand how text
becomes numbers, then build classifiers on top.

## Key techniques demonstrated

- The NLP pipeline: preprocess -> vectorize -> model (section 1.4)
- Bag-of-Words and TF-IDF weighting (sections 3.1-3.2)
- N-grams and the negation lesson for sentiment (sections 3.3, 6.3)
- Multinomial Naive Bayes and Logistic Regression text classifiers
- Interpretability: reading the model's top words as "spam signals"
- Zero-download preprocessing (sklearn stop words + built-in stemmer)
