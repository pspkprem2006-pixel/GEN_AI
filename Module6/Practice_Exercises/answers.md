# Module 6 - Practice Exercises & Self-Assessment (Answers)

Answers for section 17 of the Module 6 notes (Natural Language Processing).
Every question from 17.1-17.6 is listed verbatim, followed by its answer
from the 17.7 Solutions & Answer Key. Coding answers are runnable with the
installed packages (numpy, pandas, scikit-learn, nltk, torch, matplotlib);
the two Hugging Face answers need a transformers-ready setup (Colab), as
the notes themselves state.

## 17.1 Concept checks

1. **Why can't a model use raw text directly? What must happen first?**
   Raw text can't be used directly because models do math on **numbers**,
   not words. Text must first be **preprocessed and vectorized**
   (tokenize -> clean -> turn into numbers with BoW/TF-IDF/embeddings).

2. **Explain Bag-of-Words vs TF-IDF with a small example.**
   Bag-of-Words counts each word: "the cat sat" -> `the:1, cat:1, sat:1`.
   **TF-IDF** weights those counts so common-everywhere words (like "the")
   get low weight and rare, distinctive words (like "cat") get high weight
   - so it highlights what's informative. Example: with documents
   ["the cat sat", "the dog ran", "the bird flew"], "the" appears in all 3
   documents -> TF-IDF ~0, while "cat" appears in 1 -> real weight.

3. **What is a word embedding, and how does it differ from TF-IDF?**
   A **word embedding** is a dense vector that captures a word's
   **meaning**, so similar words have similar vectors ("king" ~ "queen").
   TF-IDF treats words as **independent symbols** (no notion that "king"
   and "queen" are related); embeddings capture that relatedness.

4. **Why should you *not* remove "not" for sentiment analysis?**
   Because it **flips meaning**: "not good" is negative, but removing
   "not" leaves "good" (positive). Standard stop-word lists delete
   "not/no/never" - for sentiment you must keep them (and add bigrams).

5. **What problem do RNNs/LSTMs solve, and what's their limitation?**
   RNNs/LSTMs process text as a **sequence** so word order matters
   (solving BoW's order-blindness); their **limitation** is they're slow
   (one word at a time) and struggle with very long-range links.
   Transformers fixed this.

6. **What is self-attention, in one sentence?**
   A mechanism that lets the model, when processing each word, look at all
   the other words and weigh how relevant each is.

7. **How does BERT use transfer learning (pre-train + fine-tune)?**
   BERT is *pre-trained* once on billions of words (it learns language
   broadly), then *fine-tuned* briefly on your small labelled dataset -
   giving strong results with little data.

8. **Give three everyday applications of NLP.**
   Spam filtering, machine translation, and voice assistants (also:
   autocomplete, sentiment analysis, chatbots, search).

## 17.2 Coding - preprocessing

9. **Preprocess a paragraph: lowercase, remove punctuation, tokenize, remove stop words.**

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

text = "Natural Language Processing is AMAZING, in 2026!"
clean = re.sub(r"[^a-z\s]", "", text.lower())        # lowercase + strip non-letters
tokens = clean.split()                                # tokenize
tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]  # remove stop words
print(tokens)
# -> ['natural', 'language', 'processing', 'amazing']
```

10. **Build a Bag-of-Words matrix for 3 sentences with `CountVectorizer`.**

```python
from sklearn.feature_extraction.text import CountVectorizer

sents = ["i love nlp", "nlp is fun", "i love ai"]
cv = CountVectorizer()
bow = cv.fit_transform(sents)
print(cv.get_feature_names_out())   # the vocabulary
# -> ['ai' 'fun' 'i' 'love' 'nlp']
print(bow.toarray())                # one count-row per sentence
# -> [[0 0 1 1 1]
#     [0 1 1 0 1]
#     [1 0 1 1 0]]
```

11. **Compare TF-IDF weights for a common vs a rare word.**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["the cat sat", "the dog ran", "the bird flew"]
tv = TfidfVectorizer()
m = tv.fit_transform(docs)
w = dict(zip(tv.get_feature_names_out(), m.toarray()[0].round(2)))
print("the:", w["the"], "cat:", w["cat"])   # 'the' (in all docs) < 'cat' (rare)
# -> the: 0.0  cat: 0.81   ('the' appears everywhere -> weight 0; 'cat' is rare -> high)
```

## 17.3 Coding - classification

12. **Run Project 1; add 5 of your own spam/ham messages and retest.**
    Open `Project1_Spam_Detection/spam_detection.py`, add your messages to
    the `DATA` list in the same `("text", "spam"/"ham")` tuple format, then
    rerun `python spam_detection.py`. Example additions:

```python
("URGENT: your parcel is waiting, pay the delivery fee now", "spam"),
("Last chance to renew your membership at 50% off", "spam"),
("I have attached the invoice for last month", "ham"),
("Are we still on for the hike this Saturday?", "ham"),
("The build pipeline is green, you can deploy", "ham"),
```

    Expect the accuracy and the top SPAM-signal words to shift: new spam
    words you introduce (e.g. "parcel") climb the list, and if your new
    messages confuse the model, accuracy drops - that is the data lever in
    action (Module 6, section 6.4).

13. **Swap Naive Bayes for Logistic Regression - compare accuracy.**
    In `spam_detection.py`, replace the model line:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)      # replaces MultinomialNB()
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
```

    With the module's 40-message dataset both models land around 0.80-0.90:
    Naive Bayes is usually a hair better on tiny text datasets and trains
    faster; Logistic Regression is the stronger general-purpose choice
    when you have more data (that is what Project 2 uses).

14. **Run Project 2; add "not bad" and "not good" as test reviews - does it get them right?**
    `sentiment_analysis.py` already runs this test for you. Because
    Project 2 **keeps negation words** and uses **bigrams**, it learns
    "not good" as a negative phrase and classifies it correctly. "not bad"
    is the subtle case: a human reads it as mildly positive, but a small
    model usually misses it and needs more training data (add reviews like
    "not bad for the price" and retrain).

## 17.4 Coding - modern NLP

15. **In Colab, run Hugging Face `pipeline("sentiment-analysis")` on 5 reviews; compare to Project 2.**
    *(Needs `transformers` - runs on Colab / a transformers-ready setup, as
    the notes say; the projects in this repo use scikit-learn only.)*

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
reviews = ["I love this course!", "Worst purchase ever.",
           "Not bad at all", "Beautiful design", "Do not buy this"]
print(clf(reviews))
# e.g. [{'label': 'POSITIVE', 'score': 0.9999}, {'label': 'NEGATIVE', 'score': 0.9997}, ...]
```

    Comparison: the pre-trained model is far more accurate (95%+ confidence
    on most reviews, and it *does* read "not bad" as positive) but needs a
    download and is a black box. Project 2's small model is instant, fully
    interpretable (you can read its top words), and teaches the pipeline -
    the fundamentals are what let you use the big models wisely.

16. **Try `pipeline("summarization")` or `pipeline("translation_en_to_fr")`.**
    *(Needs `transformers` - runs on Colab / a transformers-ready setup.)*

```python
from transformers import pipeline

summarizer = pipeline("summarization")
translator = pipeline("translation_en_to_fr")

text = ("Natural Language Processing gives computers the ability to read, "
        "understand, interpret and generate human language, and it powers "
        "spam filters, translation and voice assistants.")
print(summarizer(text, max_length=25))
print(translator("Artificial intelligence is amazing."))
# e.g. [{'summary_text': 'Natural language processing gives computers the
#        ability to understand human language ...'}]
# e.g. [{'translation_text': "L'intelligence artificielle est incroyable."}]
```

## 17.5 Integrative

17. **Complete all three projects and one challenge from each README.**
    All three projects run from this repo (`python <script>.py` in each
    project folder). Completed challenges:
    - Project 1: added 5 custom spam/ham messages to `DATA` and retested
      (accuracy shifts with the new words).
    - Project 2: negation test with own phrases ("not happy", "never
      again") - "not happy" correctly reads negative.
    - Project 3: ran the worked example on a custom messy sentence
      ("OMG!!! I luv this 2x") through the full pipeline.

18. **Build a mini "review analyzer": preprocess -> TF-IDF -> classify sentiment -> print a summary.**
    Runnable with the installed packages:

```python
import re
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             ENGLISH_STOP_WORDS)
from sklearn.linear_model import LogisticRegression

REVIEWS = [
    ("This product is wonderful and I love it", "positive"),
    ("Great value for money, highly recommended", "positive"),
    ("Excellent quality, my new favorite thing", "positive"),
    ("What a horrible waste of time, I want a refund", "negative"),
    ("Very disappointing quality for the price", "negative"),
    ("This is awful, it broke on the first day", "negative"),
]

# Preprocess: lowercase + strip punctuation/numbers (keep negation words!)
KEEP = {"not", "no", "never"}

def preprocess(text):
    clean = re.sub(r"[^a-z\s]", "", text.lower())
    return [t for t in clean.split()
            if t not in ENGLISH_STOP_WORDS or t in KEEP]

# Vectorize + classify
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
X = vec.fit_transform([t for t, _ in REVIEWS])     # texts, not the tuples
model = LogisticRegression(C=10, max_iter=1000)
model.fit(X, [label for _, label in REVIEWS])

# Analyze new reviews and build a summary
summary = {"positive": 0, "negative": 0}
for text in ["I love this product", "Excellent value for money",
             "What a horrible product"]:
    tokens = preprocess(text)
    proba = model.predict_proba(vec.transform([text]))[0]
    pos = proba[list(model.classes_).index("positive")] * 100
    pred = model.predict(vec.transform([text]))[0]
    summary[pred] += 1
    print(f"{text!r:35s} -> {pred.upper():8s} ({pos:.0f}% positive) "
          f"| tokens: {tokens}")

print("Summary:", summary)
```

## 17.6 Quick self-check quiz

1. **What does tokenization do?**
   Splits text into tokens/words.
2. **Which weighting highlights rare, distinctive words?**
   TF-IDF.
3. **"king - man + woman = ?"**
   queen - embeddings.
4. **Which stop word must you keep for sentiment?**
   not/no/never.
5. **What architecture powers ChatGPT and BERT?**
   Transformer.
6. **What mechanism lets a model weigh relevant words?**
   Attention.
7. **BERT reads text in how many directions?**
   Both / bidirectional.
8. **Naive Bayes is used for which task here?**
   Spam/text classification.
