import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
df.head()
print(df.info())
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Gender Distribution')
plt.show()
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()
# Drop ID column (not useful for clustering)
df.drop('CustomerID', axis=1, inplace=True)
# Encode 'Gender'
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# Features to use for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
# Determine optimal number of clusters using Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()
# Apply KMeans with optimal k (assume 5 for this dataset)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# Silhouette Score
score = silhouette_score(X_scaled, df['Cluster'])
print(f" Silhouette Score: {score:.3f}")
# Visualize clusters (2D)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10')
plt.title('Customer Segments based on Income and Spending Score')
plt.show()
# 3D Plot
fig = px.scatter_3d(
    df,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color='Cluster',
    title='3D Cluster Visualization'
)
fig.show()
