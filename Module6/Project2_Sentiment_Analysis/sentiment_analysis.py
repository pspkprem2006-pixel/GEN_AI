"""
Project 2 - Sentiment Analysis
==============================
Classifies product reviews as POSITIVE or NEGATIVE - with the crucial
negation lesson of Module 6, section 6.3: keep negation words ("not", "no",
"never") and use bigrams, so the model can learn that "not good" is negative
instead of deleting the "not" and reading only "good".

Pipeline: review -> TF-IDF (unigrams + bigrams) -> Logistic Regression.

Uses only scikit-learn and matplotlib (labelled reviews embedded below).
Saves a chart of the words the model treats as positive (green) vs
negative (red).

Run:
    python sentiment_analysis.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA: small embedded labelled dataset - ("review", "positive"/"negative").
# 36 reviews, 18 positive and 18 negative. Negative reviews deliberately use
# "not"-phrases so the model learns the negation lesson.
# ---------------------------------------------------------------------------
DATA = [
    # --- positive (18) ---
    ("This product is wonderful and I love it", "positive"),
    ("Great value for money, highly recommended", "positive"),
    ("Excellent quality, my new favorite thing", "positive"),
    ("Absolutely beautiful design, worth every penny", "positive"),
    ("I loved this so much I bought a second one", "positive"),
    ("Amazing performance, works perfectly every time", "positive"),
    ("The best purchase I have made all year", "positive"),
    ("Fantastic service and friendly staff", "positive"),
    ("Superb craftsmanship, I am very satisfied", "positive"),
    ("I am very happy with this purchase", "positive"),
    ("You will not be disappointed, this product is amazing", "positive"),
    ("Beautiful and practical, the packaging is lovely", "positive"),
    ("Superb sound quality, crystal clear audio", "positive"),
    ("Very comfortable and stylish, exactly as described", "positive"),
    ("I would recommend this to all my friends", "positive"),
    ("The delivery was fast and the box was neat", "positive"),
    ("Very easy to set up, worked straight away", "positive"),
    ("A delightful little gadget, very clever design", "positive"),
    # --- negative (18) ---
    ("What a horrible waste of time, I want a refund", "negative"),
    ("This is awful, it broke on the first day", "negative"),
    ("Very disappointing quality for the price", "negative"),
    ("The product was boring and unhelpful", "negative"),
    ("Bad customer service, they never replied to me", "negative"),
    ("Not good at all, do not buy this", "negative"),
    ("Terrible battery life, not worth the money", "negative"),
    ("Waste of money, I regret buying it", "negative"),
    ("Poor quality, it fell apart immediately", "negative"),
    ("Unhelpful instructions and nothing worked", "negative"),
    ("This is not what I expected, very poor quality", "negative"),
    ("Do not waste your money on this terrible item", "negative"),
    ("The app kept crashing, a complete failure", "negative"),
    ("Cheap and flimsy, it felt like a toy", "negative"),
    ("The worst product I have ever used", "negative"),
    ("This cable is not compatible with my laptop", "negative"),
    ("Not recommended, the support was useless", "negative"),
    ("The shoes are not comfortable at all", "negative"),
]

# New reviews the model has never seen.
TEST_REVIEWS = [
    "This is the best thing I have ever bought",
    "What a horrible waste of time, I want a refund",
]

# The negation test (Practice Exercise 17.3, question 14):
# "not good" should flip to negative; "not bad" is the subtle case.
NEGATION_TEST = [
    ("not good", "negative in reality - 'not' flips 'good'"),
    ("not bad", "mildly positive in reality - the subtle case"),
]


def main():
    reviews = [text for text, _ in DATA]
    labels = [label for _, label in DATA]

    print(f"Dataset: {len(DATA)} reviews "
          f"({labels.count('positive')} positive, "
          f"{labels.count('negative')} negative).")

    # The crucial setup: KEEP negation words (no stop-word removal) and use
    # bigrams, so "not good" is learned as a phrase (Module 6, section 13.1).
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    X = vectorizer.fit_transform(reviews)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=34, stratify=labels)

    # C=10: less regularization = more confident on small data.
    model = LogisticRegression(C=10, max_iter=1000)
    model.fit(X_train, y_train)

    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

    # Which words are the strongest signals? (unigrams only, for readability;
    # skip filler stop words but NEVER hide negation words - they flip meaning)
    features = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    negations = {"not", "no", "never"}
    unigrams = [(f, c) for f, c in zip(features, coefs)
                if " " not in f
                and (f in negations or f not in ENGLISH_STOP_WORDS)]
    pos_words = [w for w, _ in sorted(unigrams, key=lambda t: t[1],
                                      reverse=True)[:6]]
    neg_words = [w for w, _ in sorted(unigrams, key=lambda t: t[1])[:6]]
    print("Most POSITIVE words: " + ", ".join(pos_words) + " ...")
    print("Most NEGATIVE words: " + ", ".join(neg_words) + " ...")

    # Chart: positive (green) vs negative (red) words.
    top_pos = sorted(unigrams, key=lambda t: t[1], reverse=True)[:6]
    top_neg = sorted(unigrams, key=lambda t: t[1])[:6]
    words = [w for w, _ in top_pos] + [w for w, _ in top_neg]
    scores = [c for _, c in top_pos] + [c for _, c in top_neg]
    colors = ["green"] * 6 + ["red"] * 6
    plt.figure(figsize=(9, 5))
    plt.bar(words, scores, color=colors)
    plt.title("Words the logistic regression treats as POSITIVE (green) "
              "vs NEGATIVE (red)")
    plt.ylabel("logistic regression coefficient")
    plt.tight_layout()
    plt.savefig("sentiment_words.png", dpi=150)
    print("Chart saved: sentiment_words.png")

    print()
    print("----- PREDICTIONS ON NEW REVIEWS -----")
    for text in TEST_REVIEWS:
        proba = model.predict_proba(vectorizer.transform([text]))[0]
        pos_idx = list(model.classes_).index("positive")
        p_pos = proba[pos_idx] * 100
        tag = ("[POSITIVE]" if model.classes_[proba.argmax()] == "positive"
               else "[NEGATIVE]")
        print(f"   {tag} ({p_pos:3.0f}% positive)  {text}")

    print()
    print("----- NEGATION TEST (does the model understand 'not'?) -----")
    for text, truth in NEGATION_TEST:
        proba = model.predict_proba(vectorizer.transform([text]))[0]
        pos_idx = list(model.classes_).index("positive")
        p_pos = proba[pos_idx] * 100
        pred = model.classes_[proba.argmax()]
        print(f"   '{text}' -> {pred.upper():8s} "
              f"({p_pos:3.0f}% positive)  [truth: {truth}]")
    print("   Note: because Project 2 KEEPS negation words and uses bigrams,")
    print("   'not good' is learned as a negative phrase. 'not bad' is the")
    print("   classic subtle case - a human reads it as mildly positive, but a")
    print("   small model may need more training data like 'not bad for the")
    print("   price' to learn it.")


if __name__ == "__main__":
    main()
