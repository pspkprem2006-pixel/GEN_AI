# Project 3 - Customer Segmentation (Clustering)

Group customers into segments with **no labels** using **K-Means** on two
features: annual income (in k) and spending score. Per Module 4 notes,
section 14.

## Files
- `customer_segmentation.py` - the full program
- `segments_and_elbow.png` - elbow plot + colored clusters with centroids
  (created when you run it)

## What it does
1. **Generates its own synthetic dataset** - 200 customers as 5 blobs of
   40 (high/low income x high/low spend, plus an average group) using
   `random_state=42` (fully reproducible, no CSV needed).
2. **Scales first!** `StandardScaler` - K-Means is distance-based, so
   unscaled features would dominate.
3. **Elbow method** - inertia for k = 1..10 to sanity-check k.
4. **Fits** `KMeans(n_clusters=5, random_state=42, n_init=10)` with no `y`
   - this is unsupervised learning.
5. **Describes** each segment (count + average income/spend) and **names**
   it with a business label: Budget, Young Spenders, Average,
   Savers (win them over), Premium (VIP - target!).
6. **Saves** one PNG: the elbow plot next to the colored clusters with
   their centroids.

## How to run
```bash
python customer_segmentation.py
```
Needs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Expected output
```
----- ELBOW METHOD (inertia for k = 1..10) -----
   The curve flattens sharply after k=5 -> elbow at k=5

----- CUSTOMER SEGMENTS FOUND -----
   Segment 0:  40 customers | avg income   29.6k | avg spend  78.1 | Young Spenders
   Segment 1:  40 customers | avg income   55.4k | avg spend  50.9 | Average
   Segment 2:  40 customers | avg income   78.7k | avg spend  19.7 | Savers (win them over)
   Segment 3:  40 customers | avg income   85.1k | avg spend  82.6 | Premium (VIP - target!)
   Segment 4:  40 customers | avg income   32.4k | avg spend  28.9 | Budget
```
Same 5 clean segments as the notes' sample output (section 14.2) - just
possibly in a different order, because K-Means labels are arbitrary.

## Challenges
1. Re-run with `n_clusters=3` - which two segments merge, and what name
   would you give the merged group?
2. Compute the **silhouette score** for k = 2..10 - does it agree with
   the elbow at k=5?
3. Describe a marketing action for each segment (e.g. Premium gets a VIP
   loyalty program, Savers get a discount on premium products).
