# Project 1 - Spam Detection

The syllabus activity for Module 6: a spam filter that classifies
SMS-style messages as **spam** or **ham** (not spam). This is the whole NLP
pipeline (Module 6 notes, section 1.4) end to end:

```
raw text -> TF-IDF vectors -> Multinomial Naive Bayes -> predict
```

## Files
- `spam_detection.py` - the full program
- `spam_top_words.png` - chart of the words the model treats as SPAM signals
  (created when you run it)

## How it works
1. **Data:** a small embedded labelled list of 40 messages
   (20 spam, 20 ham) as `("text", "spam"/"ham")` tuples - no external
   dataset files needed.
2. **Vectorize:** `TfidfVectorizer(stop_words="english")` turns each message
   into a TF-IDF vector, downweighting common words and highlighting
   distinctive ones like "free", "win", "claim".
3. **Classify:** `MultinomialNB` - the classic fast, interpretable text
   classifier.
4. **Split:** 75/25 train/test with `stratify` (fit the vectorizer on
   training data only - no leakage).
5. **Interpret:** prints the 10 words with the biggest
   `log P(word|spam) - log P(word|ham)` - the model's "thinking" made
   visible.

## How to run
```bash
python spam_detection.py
```

## Expected output
```
Dataset: 40 messages (20 spam, 20 ham).
Accuracy: 0.900
Top words the model treats as SPAM signals:
   free, details, claim, offer, money, click, cheap, win, week, account

----- PREDICTIONS ON NEW MESSAGES -----
   [SPAM] ( 83% spam)  Congratulations! Claim your FREE prize money now
   [ham ] ( 27% spam)  Hey, can we reschedule our call to 3pm tomorrow?
   [SPAM] ( 82% spam)  Cheap watches for sale, text WIN to claim your offer
   [ham ] ( 30% spam)  Remember to bring your laptop to the workshop
```

The model hits ~90% accuracy on a tiny dataset and shows *which words* it
treats as spam signals - the bonus of interpretable models.

## Challenges
1. Add **5 of your own** spam/ham messages to the `DATA` list (same
   `("text", "label")` format) and rerun - how does accuracy change?
2. Swap `MultinomialNB()` for `LogisticRegression(max_iter=1000)` and
   compare accuracy (Practice Exercise 17.3, question 13).
3. Try `TfidfVectorizer(stop_words=None)` (keep stop words) - what happens
   to the top spam signals and the accuracy?
