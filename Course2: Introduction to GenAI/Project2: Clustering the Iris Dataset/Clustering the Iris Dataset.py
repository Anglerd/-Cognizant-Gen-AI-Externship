from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['class'] = iris.target  # Add the target column for later comparison
print(df.head())

from sklearn.preprocessing import StandardScaler

# Normalize the features
scaler = StandardScaler()
X = df.drop("class", axis=1)  # Exclude the class column
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Use the Elbow Method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# PART 2
print("----------PART 2----------")

print("Scaled Features:")
print(X_scaled[:5])

from sklearn.metrics import silhouette_score

# Fit K-Means with the optimal number of clusters (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Test different random_state values
for seed in [42, 123, 456]:
    kmeans = KMeans(n_clusters=3, random_state=seed)
    kmeans.fit(X_scaled)
    print(f"Random State: {seed}, Inertia: {kmeans.inertia_:.2f}")

from sklearn.decomposition import PCA

# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering (k=3)")
plt.show()

# Part 3
print("----------PART 3----------")

# Add cluster labels to the dataframe
df['cluster'] = kmeans.labels_

# Compare cluster labels with true labels
print(df[['class', 'cluster']].head())

from sklearn.metrics import adjusted_rand_score

# Calculate Adjusted Rand Index
ari = adjusted_rand_score(df['class'], df['cluster'])
print(f"Adjusted Rand Index: {ari:.2f}")

from sklearn.metrics import confusion_matrix

# Generate confusion matrix
conf_matrix = confusion_matrix(df['class'], df['cluster'])
print("Confusion Matrix:")
print(conf_matrix)
