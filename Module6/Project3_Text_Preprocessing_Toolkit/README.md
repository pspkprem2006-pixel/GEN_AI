# Project 3 - Text Preprocessing Toolkit

The foundation under Projects 1 & 2: turn raw text into numbers, step by
step. You **see** text become tokens, get cleaned, and finally turn into
the Bag-of-Words / TF-IDF numbers that the classifiers feed on
(Module 6, section 14).

```
[0] Original -> [1] Lowercase -> [2] Clean (no punctuation/numbers)
 -> [3] Tokenize -> [4] Stop words removed -> [5] Stem
 -> [6] Bag-of-Words -> [7] TF-IDF
```

## Files
- `preprocessing_toolkit.py` - the full program
- `word_frequency.png` - top-10 token frequency chart (created when you
  run it)

## How it works
Every preprocessing step runs on real text with before/after output:

1. `lowercase()` - "APPLE"/"apple" become the same word.
2. `clean()` - `re.sub(r"[^a-z\s]", "", text)` strips punctuation/digits.
3. `tokenize()` - split into word tokens.
4. `remove_stop_words()` - drops common filler words (scikit-learn's
   `ENGLISH_STOP_WORDS`, **zero downloads**).
5. `simple_stem()` - a small built-in stemmer (running -> run, quickly ->
   quick). No NLTK data needed - per the notes, Project 3 "uses a
   simplified stemmer so it runs with zero downloads".
6. `CountVectorizer` - the Bag-of-Words count matrix.
7. `TfidfVectorizer` - TF-IDF weights, showing that rare, distinctive
   words get high weight while common ones are downweighted.

It also replays the **full worked example** from the notes (section 2.7):
"The 2 CATS are Running QUICKLY!! :)" -> `['cat', 'run', 'quick']`.

## How to run
```bash
python preprocessing_toolkit.py
```

## Expected output (key lines)
```
[3] Tokenized:      ['i', 'love', 'natural', 'language', 'processing', ...]
[4] Stop words removed: ['love', 'natural', 'language', 'processing', 'amazing']
[5] Stemmed:        ['love', 'natural', 'language', 'process', 'amaz']

--- the full worked example ---
[4] no stop:['cats', 'running', 'quickly']
[5] stem:   ['cat', 'run', 'quick']

[6] Bag-of-Words: Vocabulary: ['amaz', 'clean', 'day', ...] + count matrix
[7] TF-IDF weights for sentence 1:
   amaz: 0.59   love: 0.47   language: 0.38   natural: 0.38
   process: 0.38   ...
```

## Challenges
1. Run the worked example on your own messy text ("OMG!!! I luv this 2x").
2. Add 2 more sentences to `SENTENCES` and watch the vocab/matrix change.
3. Add a lemmatization-style rule to `simple_stem()` (e.g. "better" ->
   "good" needs a dictionary - why is stemming easier?).
