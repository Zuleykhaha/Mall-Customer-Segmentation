import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Load the Dataset
df = pd.read_csv('Mall_Customers.csv')

# 2. Dataset Preview
print("--- Dataset Preview ---")
print(df.head())

print("\n--- Missing Value Check ---")
print(df.isnull().sum())

# 3. Feature Selection (Annual Income and Spending Score)
X = df.iloc[:, [3, 4]].values

# 4. Determine Optimal Number of Clusters using the Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 5. Train K-Means Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 6. Add Cluster Labels to the Original Dataset
df['Segment'] = y_kmeans

# 7. Segment Analysis Table
segment_analysis = df.groupby('Segment').agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Age': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Customer Count'}).round(2)

print("\n--- SEGMENT ANALYSIS TABLE ---")
print(segment_analysis)

# 8. Visualization of Customer Segments
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = [
    'Average Customers (Mid Income / Mid Spending)',
    'VIP Customers (High Income / High Spending)',
    'Low Value Customers (Low Income / Low Spending)',
    'Impulsive Customers (Low Income / High Spending)',
    'Cautious Wealthy Customers (High Income / Low Spending)'
]

for i in range(5):
    plt.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=100,
        c=colors[i],
        label=labels[i],
        edgecolors='black',
        alpha=0.7
    )

# Mark cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids',
    marker='X',
    edgecolors='black'
)

plt.title('Mall Customer Segmentation using K-Means', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
plt.tight_layout()
plt.show()

print("\nAnalysis completed successfully.")
