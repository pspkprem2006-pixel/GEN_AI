# Practice Questions with Answers

## 17.1 Concept checks

1. Question: Explain the difference between regression and classification with an example each.
   - Answer: Regression predicts a continuous value, while classification predicts a category.
   - Example of regression: predicting house price.
   - Example of classification: predicting whether an email is spam or not.

2. Question: Why must you never test a model on its training data?
   - Answer: Because it gives an overly optimistic result and does not show how well the model generalizes to new data.

3. Question: Why is accuracy a poor metric for a fraud detector? What would you use instead?
   - Answer: Accuracy is poor because fraud is rare, so a model can get high accuracy by predicting mostly "not fraud".
   - Better metrics: precision, recall, F1-score, ROC-AUC, and PR-AUC.

4. Question: What does an R² of 0.85 mean? What about R² of 0?
   - Answer: An R² of 0.85 means the model explains 85% of the variation in the target variable.
   - An R² of 0 means the model explains none of the variance and performs no better than predicting the mean.

5. Question: Explain precision vs recall using a disease-screening example.
   - Answer: Precision means how many predicted positives are actually positive. Recall means how many actual positives were correctly identified.
   - In disease screening, recall is often more important because missing a sick patient is costly.

6. Question: When would you scale features? Name two models that need it and two that don't.
   - Answer: Scale features when using models that depend on distance or gradient-based optimization, such as Logistic Regression, SVM, KNN, and neural networks.
   - Models that usually do not need scaling: Decision Trees and Random Forests.

7. Question: What is overfitting, and how do you detect and reduce it?
   - Answer: Overfitting means the model learns the training data too specifically and fails on new data.
   - Detect it with a train-test gap or validation curves.
   - Reduce it with more data, simpler models, regularization, or cross-validation.

8. Question: In K-Means, what is a centroid and how do you choose k?
   - Answer: A centroid is the center of a cluster.
   - Choose k using the elbow method, silhouette score, or domain knowledge.

## 17.2 Coding - regression

1. Question: Load a CSV, split into X/y and train/test, and train a LinearRegression. Report R² and MAE.
   - Answer: Use train_test_split, fit LinearRegression, and evaluate with r2_score and mean_absolute_error.

2. Question: Print and interpret the model's coefficients — which feature matters most?
   - Answer: Interpret coefficients by looking at their size and sign. A larger absolute coefficient means a stronger effect on the prediction.

3. Question: Swap in a RandomForestRegressor; does R² improve on the test set?
   - Answer: Compare the test R² of RandomForestRegressor with LinearRegression and choose the model with the higher score.

## 17.3 Coding - classification

1. Question: Train a LogisticRegression on a binary dataset; print the classification report.
   - Answer: Fit LogisticRegression on the training set and print classification_report(y_test, pred).

2. Question: Draw the confusion matrix; identify the false positives and false negatives.
   - Answer: Use confusion_matrix(y_test, pred) and interpret the matrix entries.

3. Question: Use predict_proba to list the 5 samples the model is most confident are positive.
   - Answer: Use model.predict_proba(X_test)[:, 1] and sort the probabilities in descending order.

## 17.4 Coding - clustering

1. Question: Run K-Means with k=3 and k=5 on a 2-feature dataset; plot both.
   - Answer: Fit KMeans for each k and create scatter plots of the clusters.

2. Question: Draw an elbow plot for k = 1…10 and pick the best k.
   - Answer: Plot inertia values for each k and choose the point where the curve bends.

3. Question: Name each cluster you find based on its feature averages.
   - Answer: Compare the mean values of each cluster and assign a meaningful label such as high, low, or medium.

## 17.5 Integrative

1. Question: Complete all three projects, then do one README challenge in each.
   - Answer: Build the projects and document each one clearly in its README with the problem, workflow, and results.

2. Question: Take a real Kaggle dataset (e.g., Titanic survival) and build + evaluate a classifier end to end.
   - Answer: Load the dataset, clean it, split it, train a classifier, and evaluate it with metrics such as accuracy, precision, recall, and F1-score.

## 17.6 Quick self-check quiz

1. Question: Predicting temperature is regression or classification?
   - Answer: Regression

2. Question: What are the 4 lines of the ML rhythm?
   - Answer: choose, fit, predict, evaluate

3. Question: Which metric can be misleading on imbalanced data?
   - Answer: accuracy

4. Question: What does stratify=y do?
   - Answer: It keeps class balance in the split

5. Question: Which algorithm needs no y?
   - Answer: K-Means / clustering

6. Question: High train score, low test score means…?
   - Answer: overfitting

7. Question: What does predict_proba return?
   - Answer: class probabilities

8. Question: What does the Elbow Method choose?
   - Answer: the number of clusters k
