import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from scipy.stats import pearsonr, spearmanr
import matplotlib.patheffects as path_effects

# Loaimport matplotlib.patheffects as path_effectsd the updated CSV file with new columns
file_path = 'ablation.csv'
data = pd.read_csv(file_path)

# Sort the DataFrame by the 'best_combined_features' column
data = data.sort_values(by='best_combined_features')

# Select relevant features for clustering (AUC and ERROR)
features = data[['best_combined_auc', 'best_combined_value', 'best_combined_metric']]

# Normalize the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []  # Within-cluster sum of squares
max_clusters = 15  # Adjust the max number of clusters to test if needed
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(normalized_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 8))
plt.plot(range(1, max_clusters+1), wcss, marker='o')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(range(1, max_clusters+1), fontsize=12)
plt.yticks(np.arange(0, max(wcss), step=10000), fontsize=12)

# Add the difference between each step in red, with adjusted positions to avoid overlap
for i in range(1, max_clusters):

    # Get the half difference between the current and next WCSS
    difference_y = (wcss[i] - wcss[i-1]) / 2 

    # Set the half difference between the current and next number of clusters
    difference_x = 0.5

    # Get the normalized difference in WCSS and cluster number
    norm_difference_y = difference_y / max(wcss)
    norm_difference_x = difference_x / max_clusters

    # Calculate the slope of the line between the current and next WCSS
    slope = abs(norm_difference_y / norm_difference_x)

    print(difference_y, slope)

    # Set the cutoff for the difference in WCSS
    cutoff = -4500

    # If the difference in WCSS is above cutoff
    if difference_y > cutoff:
        # Set the difference in y to -cutoff
        difference_y = cutoff

    # If the slope is below or equal to 45 degrees
    if slope > 1:
        # If the slope is above 1.5
        if slope > 1.5:
            # Make it 1.5
            slope = 1.5
        # Multiply the difference in x by the slope
        difference_y = difference_y * slope * ((i - max_clusters) / 150)
        difference_x = ((i - max_clusters) / 24) * difference_x

    plt.text(i + difference_x, wcss[i] + difference_y, f'{wcss[i-1] - wcss[i]:.0f}', color='red', fontsize=10, ha='center')
    
plt.grid(True)
plt.show()


# K-Means
###############################################################################

# Choose the optimal number of clusters (e.g., from the elbow plot)
optimal_clusters = 2  # Replace with the actual optimal number from your elbow plot

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

# Add cluster labels to the data
data['cluster'] = clusters

# Calculate the Pearson correlation coefficient
corr_coef, p_value_pearson = pearsonr(data['best_combined_auc'], data['best_combined_value'])

# Calculate the Spearman correlation coefficient
spearman_corr, p_value_spearman = spearmanr(data['best_combined_auc'], data['best_combined_value'])

# Calculate the Spearman correlation coefficient for the entire dataset
spearman_corr, p_value_spearman = spearmanr(data['best_combined_auc'], data['best_combined_value'])

# Plot the clusters
plt.figure(figsize=(14, 8))

# Plot the elements with scatter plot
sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette='viridis', edgecolor='black', s=50)

# Add a regression line for Spearman correlation
sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan')

# Annotate the Spearman correlation coefficient and p-value next to its line
spearman_text = plt.text(0.75, 0.62, f'Spearman Correlation: {spearman_corr:.2f}', color='cyan', fontsize=14, weight='bold')
spearman_pvalue_text = plt.text(0.75, 0.618, f'P-value: {p_value_spearman:.2e}', color='cyan', fontsize=14, weight='bold')

# Add edgecolor to Spearman correlation text
for text in [spearman_text, spearman_pvalue_text]:
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
"""
# Calculate and annotate Spearman correlation for each cluster
for i, cluster_id in enumerate(sorted(data['cluster'].unique())):
    cluster_data = data[data['cluster'] == cluster_id]
    spearman_corr_cluster, p_value_spearman_cluster = spearmanr(cluster_data['best_combined_auc'], cluster_data['best_combined_value'])
    
    # Determine position for text annotation based on cluster
    x_pos = 0.02
    y_pos = 0.95 - (0.05 * i)  # Adjust y position for each cluster to avoid overlap
    
    cluster_text = plt.text(x_pos, y_pos, f'Cluster {cluster_id} Spearman: {spearman_corr_cluster:.2f}', fontsize=12, color='black', transform=plt.gca().transAxes)
    cluster_pvalue_text = plt.text(x_pos, y_pos - 0.025, f'P-value: {p_value_spearman_cluster:.2e}', fontsize=12, color='black', transform=plt.gca().transAxes)
    
    # Add edgecolor to cluster-specific text
    for text in [cluster_text, cluster_pvalue_text]:
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
"""
# Add a title and labels
plt.title('K-Means Clusters of Data Points using AUC and RMSE', fontsize=16)
plt.xlabel('AUC', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.legend(title='Cluster', fontsize=12, title_fontsize='13')
plt.savefig('Kmeans.png')
plt.show()


# DBSCAN
###############################################################################

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.01, min_samples=10)
data['cluster'] = dbscan.fit_predict(data[['best_combined_auc', 'best_combined_value']])

# Calculate the Spearman correlation coefficient for the entire dataset
spearman_corr, p_value_spearman = spearmanr(data['best_combined_auc'], data['best_combined_value'])

# Plot the clusters
plt.figure(figsize=(14, 8))

# Plot the elements with scatter plot
sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette='viridis', edgecolor='black', s=50)

# Add a regression line for Spearman correlation
sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan')

# Annotate the Spearman correlation coefficient and p-value next to its line
spearman_text = plt.text(0.75, 0.62, f'Spearman Correlation: {spearman_corr:.2f}', color='cyan', fontsize=14, weight='bold')
spearman_pvalue_text = plt.text(0.75, 0.618, f'P-value: {p_value_spearman:.2e}', color='cyan', fontsize=14, weight='bold')

# Add edgecolor to Spearman correlation text
for text in [spearman_text, spearman_pvalue_text]:
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# Calculate and annotate Spearman correlation for each cluster
for i, cluster_id in enumerate(sorted(data['cluster'].unique())):
    if cluster_id == -1:
        continue  # Skip noise points
    
    cluster_data = data[data['cluster'] == cluster_id]
    spearman_corr_cluster, p_value_spearman_cluster = spearmanr(cluster_data['best_combined_auc'], cluster_data['best_combined_value'])
    
    # Determine position for text annotation based on cluster
    x_pos = 0.02
    y_pos = 0.95 - (0.05 * i)  # Adjust y position for each cluster to avoid overlap
    
    cluster_text = plt.text(x_pos, y_pos, f'Cluster {cluster_id} Spearman: {spearman_corr_cluster:.2f}', fontsize=12, color='black', transform=plt.gca().transAxes)
    cluster_pvalue_text = plt.text(x_pos, y_pos - 0.025, f'P-value: {p_value_spearman_cluster:.2e}', fontsize=12, color='black', transform=plt.gca().transAxes)
    
    # Add edgecolor to cluster-specific text
    for text in [cluster_text, cluster_pvalue_text]:
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

# Add a title and labels
plt.title('DBSCAN Clusters of Data Points using AUC and RMSE', fontsize=16)
plt.xlabel('AUC', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.legend(title='Cluster', fontsize=12, title_fontsize='13')
plt.savefig('DBSCAN.png')
#plt.show()


# HDBSCAN
###############################################################################

# Perform HDBSCAN clustering
hdbscan = HDBSCAN(min_samples=10, min_cluster_size=10)
data['cluster'] = hdbscan.fit_predict(data[['best_combined_auc', 'best_combined_value']])

# Calculate the Spearman correlation coefficient for the entire dataset
spearman_corr, p_value_spearman = spearmanr(data['best_combined_auc'], data['best_combined_value'])

# Plot the clusters
plt.figure(figsize=(14, 8))

# Plot the elements with scatter plot
sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette='viridis', edgecolor='black', s=50)

# Add a regression line for Spearman correlation
sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan')

# Annotate the Spearman correlation coefficient and p-value next to its line
spearman_text = plt.text(0.75, 0.62, f'Spearman Correlation: {spearman_corr:.2f}', color='cyan', fontsize=14, weight='bold')
spearman_pvalue_text = plt.text(0.75, 0.618, f'P-value: {p_value_spearman:.2e}', color='cyan', fontsize=14, weight='bold')

# Add edgecolor to Spearman correlation text
for text in [spearman_text, spearman_pvalue_text]:
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# Calculate and annotate Spearman correlation for each cluster
for i, cluster_id in enumerate(sorted(data['cluster'].unique())):
    if cluster_id == -1:
        continue  # Skip noise points
    
    cluster_data = data[data['cluster'] == cluster_id]
    spearman_corr_cluster, p_value_spearman_cluster = spearmanr(cluster_data['best_combined_auc'], cluster_data['best_combined_value'])
    
    # Determine position for text annotation based on cluster
    x_pos = 0.02
    y_pos = 0.95 - (0.05 * i)  # Adjust y position for each cluster to avoid overlap
    
    cluster_text = plt.text(x_pos, y_pos, f'Cluster {cluster_id} Spearman: {spearman_corr_cluster:.2f}', fontsize=12, color='black', transform=plt.gca().transAxes)
    cluster_pvalue_text = plt.text(x_pos, y_pos - 0.025, f'P-value: {p_value_spearman_cluster:.2e}', fontsize=12, color='black', transform=plt.gca().transAxes)
    
    # Add edgecolor to cluster-specific text
    for text in [cluster_text, cluster_pvalue_text]:
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

# Add a title and labels
plt.title('HDBSCAN Clusters of Data Points using AUC and RMSE', fontsize=16)
plt.xlabel('AUC', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.legend(title='Cluster', fontsize=12, title_fontsize='13')
plt.savefig('HDBSCAN.png')
#plt.show()

# Hightlight the best combined features, the all 0s and all 1s
###############################################################################

# Add a new column to the data to highlight the best combined features
data['highlight'] = 'none'

# For each row in the data
for index, row in data.iterrows():
    # Get the first 16 characters of the best_combined_features string, which represent the features
    features = row['best_combined_features'][:16]

    # If the best_combined_features string is made up of all 1s
    if features == '1' * len(features):
        # Highlight the row as 'all_1'
        data.at[index, 'highlight'] = 'all_1'
    # If the best_combined_features string is made up of all 0s
    elif features == '0' * len(features):
        # Highlight the row as 'all_0'
        data.at[index, 'highlight'] = 'all_0'
    
# Set the highlight in the best_combined_metric to 'best' for the best combined value (lowest)
data.loc[data['best_combined_metric'] == data['best_combined_metric'].min(), 'highlight'] = 'best'

# Calculate the Spearman correlation coefficient for the entire dataset
spearman_corr, p_value_spearman = spearmanr(data['best_combined_auc'], data['best_combined_value'])

# Plot the data points
plt.figure(figsize=(14, 8))

# Plot the elements with scatter plot, highlighting all 1s and all 0s
sns.scatterplot(x='best_combined_auc', y='best_combined_value', data=data[data['highlight'] == 'none'], color='gray', edgecolor='black', s=50)
sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='highlight', data=data[data['highlight'] != 'none'], palette={'all_1': 'red', 'all_0': 'blue', 'best': 'green'}, edgecolor='black', s=100, marker='D')

# Add a regression line for Spearman correlation
sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan')

# Annotate the Spearman correlation coefficient and p-value next to its line
spearman_text = plt.text(0.75, 0.62, f'Spearman Correlation: {spearman_corr:.2f}', color='cyan', fontsize=14, weight='bold')
spearman_pvalue_text = plt.text(0.75, 0.618, f'P-value: {p_value_spearman:.2e}', color='cyan', fontsize=14, weight='bold')

# Add edgecolor to Spearman correlation text
for text in [spearman_text, spearman_pvalue_text]:
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# Add a title and labels
plt.title('Data Points with AUC and RMSE', fontsize=16)
plt.xlabel('AUC', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.legend(title='Highlight', fontsize=12, title_fontsize='13')
plt.show()

print(data[data['highlight'] != 'none'])

# Display the first few rows of the data with cluster labels
print(data.head())
