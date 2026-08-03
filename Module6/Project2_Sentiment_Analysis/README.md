# Project 2 - Sentiment Analysis

Classifies product reviews as **positive** or **negative** - with the
crucial **negation lesson** of Module 6 (section 6.3 / 13): a normal
stop-word list deletes "not", so "not good" would be read as "good".
Project 2 **keeps negation words** ("not", "no", "never") and uses
**bigrams**, so the model can learn "not good" as a negative phrase.

```
review -> TF-IDF (unigrams + bigrams) -> Logistic Regression -> predict
```

## Files
- `sentiment_analysis.py` - the full program
- `sentiment_words.png` - chart of positive (green) vs negative (red) words
  (created when you run it)

## How it works
1. **Data:** a small embedded labelled list of 36 reviews
   (18 positive, 18 negative) as `("text", "positive"/"negative")` tuples.
2. **Vectorize:** `TfidfVectorizer(lowercase=True, ngram_range=(1, 2))` -
   no stop-word removal (negations survive!) and word *pairs* like
   "not good" become features.
3. **Classify:** `LogisticRegression(C=10)` - less regularization = more
   confident predictions on small data.
4. **Negation test:** the program itself runs Practice Exercise 17.3
   question 14 - it feeds "not good" and "not bad" to the trained model.

## How to run
```bash
python sentiment_analysis.py
```

## Expected output
```
Dataset: 36 reviews (18 positive, 18 negative).
Accuracy: 0.889
Most POSITIVE words: purchase, beautiful, design, love, wonderful, great ...
Most NEGATIVE words: not, waste, disappointing, price, boring, unhelpful ...

----- PREDICTIONS ON NEW REVIEWS -----
   [POSITIVE] ( 54% positive)  This is the best thing I have ever bought
   [NEGATIVE] ( 14% positive)  What a horrible waste of time, I want a refund

----- NEGATION TEST (does the model understand 'not'?) -----
   'not good' -> NEGATIVE ( 38% positive)  [truth: negative ...]
   'not bad' -> NEGATIVE ( 26% positive)  [truth: mildly positive ...]
```

Notice **"not" is the strongest negative signal** - proof the negation
handling works. "not good" is correctly classified as negative. "not bad"
is the classic subtle case: humans read it as mildly positive, but a small
model needs more training data (e.g. "not bad for the price") to learn it.

## Challenges
1. Run the negation test with your own phrases ("not happy", "no problem",
   "never again") and see what the model says.
2. Add 5 more reviews to `DATA` (mix of positive and negative) and rerun -
   more data pushes accuracy higher (Module 6, section 6.4).
3. Try `ngram_range=(1, 3)` (add trigrams) - does accuracy improve?
