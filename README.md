# App_User_Behavior-Segmentation-using-Unsupervised-Machine-Learning

This project focuses on analyzing and segmenting mobile application users based on their engagement and usage behavior using Unsupervised Machine Learning techniques.
The main objective of this project is to identify different categories of users based on their behavioral patterns and engagement metrics.

## Technologies used:
``` text
~ Python
~ Pandas
~ NumPy
~ Matplotlib
~ Seaborn
~ Scikit-learn
~ Streamlit
```
## Project Structure:
```
App_User_Behavior_Segmentation/
│
├── app/
│   └── user_streamlit.py
│
├── data/
│   ├── app_user_behavior_dataset.csv
│   └── app_user_clean.csv
│
├── notebooks/
│   └── App_user.ipynb
│
├── images/
│   ├── cluster_distribution.png
│   ├── elbow_method.png
│   ├── silhouette_analysis.png
│   └── pca_visualization.png
│
├── README.md
│
└── requirements.txt
```
## Project Features
The project includes :
``` text
~ Exploratory Data Analysis (EDA)
~ Correlation Heatmaps
~ Cluster Distribution Analysis
~ PCA Visualization for dimensionality reduction
~ Interactive Streamlit Dashboard
```
## User Segments Identified
The main objective of the project is to identify different groups of users such as:
``` text
~ High Engagement Users
~ Moderate Users
~ Occasional Users
~ At-Risk Users
```
## Features used for Clustering
The project users behavioral features like:
```
~ Daily active minutes
~ Session per week
~ Average session duration
~ Enagement score
```
These behavioral features were used to perform clustering analysis and user segmentation.

## Machine Learning Techniques used
```text
~ Feature Engineering
~ Feature Scaling using StandardScaler
~ MiniBatchKMeans Clustering
~ Elbow Method
~ Silhouette Anlaysis
~ PCA (Principal Component Analysis)
```

## Optimization Techniques
To improve computational efficiency:
```text
~ MiniBatchKMeans was used instead of traditional KMeans
~ Sampling techniques were applied during silhouette score calculation
This significantly reduced the running time for large datasets.
```

## Streamlit Dashboard
The project includes an interactive Streamlit dashboard for:
```text
* Visualizing user clusters
* analysing engagement behaviour
* performing clustering analysis interactively.
```
## Conclusion

This project demonstrates how Unsupervised Machine Learning techniques can be used to analyze user engagement behavior and identify meaningful user segments for better business decision-making.
