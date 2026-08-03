"""
Project 1 - Spam Detection
==========================
A spam filter that classifies SMS-style messages as spam or ham.

Pipeline (the whole NLP pipeline of Module 6, section 1.4, end to end):
    raw text -> TF-IDF vectors -> Multinomial Naive Bayes -> predict

Uses only scikit-learn and matplotlib (no external dataset files - the
labelled DATA list below is embedded in this script). Saves a chart of the
words the model treats as SPAM signals.

Run:
    python spam_detection.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA: small embedded labelled dataset - ("message", "spam"/"ham") tuples.
# 40 messages, 20 spam and 20 ham.
# ---------------------------------------------------------------------------
DATA = [
    # --- spam (20) ---
    ("Congratulations! Claim your FREE prize money now", "spam"),
    ("You have won a FREE iPhone, click here to claim it today", "spam"),
    ("Limited time offer: 50% discount on all products", "spam"),
    ("FREE money waiting for you, text WIN to 44567", "spam"),
    ("Your account has been suspended, verify your details now", "spam"),
    ("Cheap watches for sale, click this link for details", "spam"),
    ("Win a luxury holiday this week, enter now", "spam"),
    ("Get a free voucher for every purchase this week", "spam"),
    ("URGENT: your account needs verification, send your details", "spam"),
    ("Offer ends tonight, claim your free gift", "spam"),
    ("You are the lucky winner of our weekly prize draw", "spam"),
    ("Text FREE to 80085 for a chance to win big", "spam"),
    ("Limited stock, buy cheap electronics today", "spam"),
    ("Your credit card details are needed to receive your prize", "spam"),
    ("Hot deal: free shipping on everything this week", "spam"),
    ("Claim your exclusive offer before it expires", "spam"),
    ("You have been selected for a free trial offer", "spam"),
    ("Win big money playing our new game, details inside", "spam"),
    ("Confirm your account details to receive your refund", "spam"),
    ("Best cheap deals in town, click for details", "spam"),
    # --- ham (20) ---
    ("Hey, can we reschedule our call to 3pm tomorrow?", "ham"),
    ("The meeting notes are attached to the email", "ham"),
    ("Did you finish the report for Friday?", "ham"),
    ("Lunch at the usual place at noon?", "ham"),
    ("Please review the attached document when you have time", "ham"),
    ("Happy birthday, hope you have a wonderful day", "ham"),
    ("The project deadline has been moved to Monday", "ham"),
    ("Can you send me the updated budget spreadsheet?", "ham"),
    ("I will be late for the morning standup today", "ham"),
    ("Thanks for your help with the presentation", "ham"),
    ("Remember to bring your laptop to the workshop", "ham"),
    ("The client approved our proposal, great news", "ham"),
    ("Call me when you get a chance, nothing urgent", "ham"),
    ("Are you coming to the team lunch on Friday?", "ham"),
    ("The quarterly numbers look good, nice work", "ham"),
    ("Please confirm your attendance for the training", "ham"),
    ("We need to discuss the new feature requirements", "ham"),
    ("The test results are in the shared folder", "ham"),
    ("Do not forget to submit your timesheet tonight", "ham"),
    ("See you at the office tomorrow morning", "ham"),
]

# New messages we have never trained on - what does the model say?
NEW_MESSAGES = [
    "Congratulations! Claim your FREE prize money now",
    "Hey, can we reschedule our call to 3pm tomorrow?",
    "Cheap watches for sale, text WIN to claim your offer",
    "Remember to bring your laptop to the workshop",
]


def main():
    messages = [text for text, _ in DATA]
    labels = [label for _, label in DATA]

    print(f"Dataset: {len(DATA)} messages "
          f"({labels.count('spam')} spam, {labels.count('ham')} ham).")

    # Text -> numbers: TF-IDF (downweights common words, highlights
    # distinctive ones like "free", "win"). Remove English stop words.
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(messages)

    # Train/test split: fit the vectorizer on TRAINING data only (no leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=1, stratify=labels)

    # The classic text classifier: Multinomial Naive Bayes.
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # Interpretability bonus: which words does the model treat as SPAM
    # signals? feature_log_prob_[1] is log P(word | spam), [0] is ham.
    log_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    features = vectorizer.get_feature_names_out()
    top_spam = sorted(zip(features, log_diff), key=lambda t: t[1],
                      reverse=True)[:10]
    print("Top words the model treats as SPAM signals:")
    print("   " + ", ".join(word for word, _ in top_spam))

    # Chart: top spam-signal words.
    words = [w for w, _ in top_spam][::-1]
    scores = [s for _, s in top_spam][::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(words, scores, color="crimson")
    plt.title("Top words the Naive Bayes model treats as SPAM signals")
    plt.xlabel("log P(word | spam) - log P(word | ham)")
    plt.tight_layout()
    plt.savefig("spam_top_words.png", dpi=150)
    print("Chart saved: spam_top_words.png")

    print()
    print("----- PREDICTIONS ON NEW MESSAGES -----")
    for text, proba in zip(NEW_MESSAGES,
                           model.predict_proba(vectorizer.transform(NEW_MESSAGES))):
        spam_idx = list(model.classes_).index("spam")
        p_spam = proba[spam_idx] * 100
        tag = "[SPAM]" if model.classes_[proba.argmax()] == "spam" else "[ham ]"
        print(f"   {tag} ({p_spam:3.0f}% spam)  {text}")


if __name__ == "__main__":
    main()
