import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

st.set_page_config(layout = 'wide')
st.title('App User Behaviour Segmentation')

@st.cache_data
def Load_data():
    #Load dataset
    df = pd.read_csv('app_user_clean.csv')
    return df
df = Load_data()

def center_plot(fig):
    left,center,right = st.columns([1,2,1])
    with center:
        st.pyplot(fig,use_container_width = True)
st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "EDA",
        "Clustering",
        "PCA Visualization",
    ]
)

#Feature selection
features = [
       'engagement_score',
        'avg_session_duration_min',
        'sessions_per_week',
        'daily_active_minutes'

]
x = df[features]

#Feature Scaling
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(x)

#Final KMeans Model
kmeans = MiniBatchKMeans(n_clusters = 4, random_state = 42, batch_size = 1000)
df['cluster'] = kmeans.fit_predict(X_scaled)
cluster_names = {
    0: 'Moderate Users',
    1: 'Occasional Users',
    2: 'At-Risk Users',
    3: 'High Engagement Users'
}
df['cluster_name'] = df['cluster'].map(cluster_names)

if section == "Overview":
    st.subheader('Dataset Overview')
    st.write(df.head())
    st.write('This dataset contains information about user behaviour '
    'in a mobile applications.''It incluses various features such as user ' \
    'demographics, app user enagement metrics & device information.')
    

elif section == 'EDA':
    st.subheader('Exploratory Data Analysis')

    #Pairplot graph
    pairplot = sns.pairplot(df[['engagement_score',
                                'daily_active_minutes',
                                'sessions_per_week']])
    pairplot.figure.suptitle("Pairplot of Key Features",y = 1.02)
    plt.tight_layout()
    center_plot(pairplot.figure)
    plt.close()

    st.markdown('---')
    #Distribution of daily active in minutes

    fig,ax = plt.subplots(figsize = (7,4))
    sns.histplot(df['daily_active_minutes'],ax = ax,kde = True)
    ax.set_title("Distribution of Daily Active Minutes")
    plt.tight_layout()
    center_plot(fig)
    plt.close()

    st.markdown('---')
    #Correlation heatmap
    fig, ax = plt.subplots(figsize = (7,4))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    center_plot(fig)
    plt.close()

    st.markdown('---')
    #Distribution of device type web
    fig, ax = plt.subplots(figsize=(7,4))
    sns.countplot(
        x='device_type',
        data=df,
        ax=ax
    )
    ax.set_title("Device Type Distribution")
    plt.tight_layout()
    center_plot(fig)
    plt.close()

    st.markdown('---')
    #Distribution of clusters
    fig, ax = plt.subplots(figsize = (7,4))
    sns.countplot(x = 'cluster_name', data = df, palette = 'Set2')
    ax.set_title("Cluster Distribution")
    plt.tight_layout()
    center_plot(fig)
    plt.close()

elif section == "Clustering":
    #Elbow method
    inertia = []
    st.subheader('Elbow Method for optimal k')
    for k in range(2,8):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size = 1000)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(range(2,8), inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    plt.tight_layout()
    center_plot(fig)
    plt.close()

    st.markdown('---')
    #Find optimal k using silhouette score
    st.subheader('Silhouette Analysis for Optimal k')
    sample_size = min(3000, len(X_scaled))
    x_sample = resample(X_scaled, n_samples = sample_size,random_state = 42)
    silhouette_scores = []
    for k in range(2,8):
        kmeans = MiniBatchKMeans(n_clusters = k, random_state = 42, batch_size = 1000)
        labels = kmeans.fit_predict(x_sample)
        score = silhouette_score(x_sample, labels)
        silhouette_scores.append(score)
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    best_score = max(silhouette_scores)

    st.subheader('Optimal Cluster Selection')
    col1, col2 = st.columns(2)
    col1.metric("Best K Value",best_k)
    col2.metric("Silhouette Score", round(best_score,4))

    st.markdown('---')
    st.subheader('Silhouette Scores for Different k Values')
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(
        range(2,8),
        silhouette_scores,
        marker='o'
    )
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")
    plt.tight_layout()
    center_plot(fig)
    plt.close()

    #groupby
    st.subheader("Cluster Summary")
    st.write(
        df.groupby('cluster_name')[['daily_active_minutes',
                        'sessions_per_week',
                        'engagement_score']].mean()
    )

    st.markdown('----')
    st.subheader('Cluster Visualization')
    #scatter plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        x='daily_active_minutes',
        y='engagement_score',
        hue = 'cluster_name',
        data=df,
        palette = 'Set2',
        ax = ax
    )
    ax.set_title('User Clusters')
    plt.tight_layout()
    center_plot(fig)
    plt.close()

elif section == "PCA Visualization":

    st.markdown("""
    PCA reduces high-dimensional data into 2 components (PC1 & PC2)
    to visualize user clusters more effectively.
    """)
    st.subheader('PCA Visualization')
    #PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster_name'] = df['cluster_name']
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
            x='PC1',
            y='PC2',
            hue='cluster_name',
            data=pca_df,
            palette='Set2',
            ax=ax
        )
    ax.set_title('PCA Cluster Visualization')
    plt.tight_layout()
    center_plot(fig)
    plt.close()

    st.subheader("Cluster Interpretation")
    st.markdown("""
    - Cluster 0 → Moderate Users
    - Cluster 1 → Occasional Users
    - Cluster 2 → At-Risk Users
    - Cluster 3 → High Engagement Users
    """)