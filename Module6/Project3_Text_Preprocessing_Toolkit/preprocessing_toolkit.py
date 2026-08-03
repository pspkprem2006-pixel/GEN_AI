"""
Project 3 - Text Preprocessing Toolkit
======================================
The foundation under Projects 1 & 2: turn raw text into numbers, step by
step. You SEE text become tokens, get cleaned, and finally turn into the
BoW / TF-IDF numbers that the classifiers feed on (Module 6, section 14).

Every preprocessing step with before/after output:
    [0] Original -> [1] Lowercase -> [2] Clean (no punctuation/numbers)
    -> [3] Tokenize -> [4] Stop words removed -> [5] Stem
    -> [6] Bag-of-Words -> [7] TF-IDF

Runs with ZERO downloads: stop words come from scikit-learn and the
stemmer is a small built-in one (no NLTK data needed). Saves a word
frequency chart.

Run:
    python preprocessing_toolkit.py
"""

import re
import collections

from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             ENGLISH_STOP_WORDS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SENTENCES = [
    "I love Natural Language Processing, it is AMAZING in 2026!",
    "Natural Language Processing turns text into numbers",
    "Running quickly through the cleaning pipeline is fun",
    "I love learning about Natural Language Processing every day",
]


def lowercase(text):
    """[1] Lowercase everything so APPLE/apple/Apple match."""
    return text.lower()


def clean(text):
    """[2] Remove everything that is not a lowercase letter or a space."""
    return re.sub(r"[^a-z\s]", "", text)


def tokenize(text):
    """[3] Split the cleaned text into word tokens."""
    return text.split()


def remove_stop_words(tokens):
    """[4] Drop common words (the, is, a, and...) that carry little meaning.

    (For sentiment analysis you would KEEP not/no/never - see Project 2.)
    """
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def simple_stem(word):
    """[5] A small built-in stemmer (no NLTK data downloads needed).

    Chops off common suffixes so running/ran/runs are treated as related.
    Rough on purpose - like a real stemmer, roots need not be real words.
    """
    if len(word) > 5 and word.endswith("ing"):
        root = word[:-3]
        # collapse doubles ("running" -> "runn" -> "run", "shopping" -> "shop")
        # but NOT in "process" or "calling", where the double letter is real.
        if (len(root) > 3 and root[-2] == root[-1]
                and root[-1] not in "aeiousl"):
            root = root[:-1]
        return root
    if len(word) > 4 and word.endswith("ed"):
        root = word[:-2]
        if (len(root) > 3 and root[-2] == root[-1]
                and root[-1] not in "aeiousl"):
            root = root[:-1]
        return root
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]              # quickly -> quick
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"        # studies -> study
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]              # movies -> movie
    if len(word) > 2 and word.endswith("s"):
        return word[:-1]              # cats -> cat
    return word


def stem(tokens):
    """Stem every token."""
    return [simple_stem(t) for t in tokens]


def preprocess(text):
    """The whole pipeline 1-5 on one piece of text."""
    step1 = lowercase(text)
    step2 = clean(step1)
    step3 = tokenize(step2)
    step4 = remove_stop_words(step3)
    step5 = stem(step4)
    return step1, step2, step3, step4, step5


def main():
    print("=== [0] Original sentences ===")
    for s in SENTENCES:
        print("   ", s)

    print()
    print("=== Preprocessing pipeline, step by step (sentence 1) ===")
    s1 = SENTENCES[0]
    step1, step2, step3, step4, step5 = preprocess(s1)
    print(f"[1] Lowercase:      {step1}")
    print(f"[2] Cleaned:        {step2}")
    print(f"[3] Tokenized:      {step3}")
    print(f"[4] Stop words removed: {step4}")
    print(f"[5] Stemmed:        {step5}")

    print()
    print("=== The full worked example (Module 6, section 2.7) ===")
    raw = "The 2 CATS are Running QUICKLY!! :)"
    print(f"Original:  {raw!r}")
    a, b, c, d, e = preprocess(raw)
    print(f"[1] lower:  {a!r}")
    print(f"[2] clean:  {b!r}")
    print(f"[3] tokens: {c}")
    print(f"[4] no stop:{d}")
    print(f"[5] stem:   {e}")
    print("The messy original becomes clean, meaningful root tokens - ready")
    print("to turn into numbers. This is what feeds Projects 1 and 2.")

    print()
    print("=== [6] Bag-of-Words (counting words, order ignored) ===")
    print("Vectorize the CLEANED text: preprocessed tokens joined per sentence,")
    print("then CountVectorizer counts how often each word appears.")
    cleaned = [" ".join(stem(remove_stop_words(tokenize(clean(lowercase(s))))))
               for s in SENTENCES]
    cv = CountVectorizer()
    bow = cv.fit_transform(cleaned)
    print("Vocabulary:", list(cv.get_feature_names_out()))
    print("Count matrix (one row per sentence, one count per word):")
    print(bow.toarray())

    print()
    print("=== [7] TF-IDF (weighting words by how distinctive they are) ===")
    tv = TfidfVectorizer()
    tfidf = tv.fit_transform(cleaned)
    print("TF-IDF weights for sentence 1:")
    names = tv.get_feature_names_out()
    weights = sorted(zip(names, tfidf.toarray()[0]),
                     key=lambda t: t[1], reverse=True)
    for i in range(0, len(weights), 4):
        line = "   "
        for word, w in weights[i:i + 4]:
            line += f"{word}: {w:.2f}   "
        print(line)
    print("'amaz' appears in only this sentence, so it gets the highest")
    print("weight; 'natural', 'language', 'process' appear in several")
    print("sentences, so they are downweighted. Rare, distinctive words are")
    print("the useful signals for the classifiers in Projects 1 and 2.")

    print()
    print("=== Word frequency chart ===")
    all_tokens = []
    for s in SENTENCES:
        all_tokens += stem(remove_stop_words(tokenize(clean(lowercase(s)))))
    counts = collections.Counter(all_tokens).most_common(10)
    print("Top tokens:", ", ".join(f"{w} ({n})" for w, n in counts))
    words = [w for w, _ in counts][::-1]
    nums = [n for _, n in counts][::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(words, nums, color="steelblue")
    plt.title("Top 10 token frequencies after preprocessing")
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig("word_frequency.png", dpi=150)
    print("Chart saved: word_frequency.png")


if __name__ == "__main__":
    main()
