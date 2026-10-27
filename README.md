K-Means Clustering

Machine Learning Internship Task 1 – Prodigy InfoTech

Project Overview

This project uses the K-Means Clustering algorithm to perform customer segmentation based on the Mall Customers dataset.

Dataset

The dataset "Mall_Customers.csv" contains the following features:

- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly

Methodology

1. Loaded and explored the dataset.
2. Performed data preprocessing and feature scaling.
3. Used the Elbow Method to examine the optimal number of clusters.
4. Applied the K-Means Clustering algorithm.
5. Evaluated clustering using the Silhouette Score.
6. Visualized customer segments using 2D and 3D plots.

Results

- Number of clusters used: 5
- Silhouette Score: 0.417
- Customer segments were visualized based on age, annual income, and spending score.

Files

- "kmeans_clustering.py" – Python source code
- "Mall_Customers.csv" – Dataset

How to Run

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn plotly

Run the Python file:

python kmeans_clustering.py

Conclusion

The project demonstrates how K-Means Clustering can group customers with similar characteristics and help understand customer segmentation.
