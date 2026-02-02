# Customer-Segmentation-for-E-Commerce


## Description
This project performs customer segmentation using unsupervised learning.
Before clustering, feature relevance is estimated using XGBoost regression on total spending.
The most important variables are then used to improve clustering quality and interpretability.

## Dataset
The project uses a marketing dataset (https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) containing demographic,
behavioral, and spending information.

## Clustering methods
- K-Means
- Hierarchical clustering
- Gaussian Mixture Models
- Spectral clustering

## Results
For each method, the best configuration is selected and analyzed using:
- Silhouette score
- t-SNE visualizations
- Numerical cluster profiles
