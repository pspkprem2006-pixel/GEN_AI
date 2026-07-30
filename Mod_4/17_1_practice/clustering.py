import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
# Replace 'your_file.csv' with your actual CSV file path
df = pd.read_csv('your_file.csv')

# Use two features for clustering
X = df[['feature_1', 'feature_2']]

# K-Means with k=3 and k=5
for k in [3, 5]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    plt.figure(figsize=(6, 4))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
    plt.title(f'K-Means clustering with k={k}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Elbow plot
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Cluster summaries
best_k = 3  # replace with the value you choose from the elbow plot
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = model.fit_predict(X)

cluster_summary = pd.DataFrame({'cluster': labels})
cluster_summary['feature_1'] = X.iloc[:, 0].values
cluster_summary['feature_2'] = X.iloc[:, 1].values

print(cluster_summary.groupby('cluster').mean())
