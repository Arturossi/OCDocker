# This script is used to determine the separation between ligand and decoy scores and check in which side (negative or positive) the ligands are more concentrated. It also checks the percentage of ligands above the median, the Kolmogorov-Smirnov test, and the percentage of ligands that overlap with the decoy range.

# SPOILER: The ligands are more concentrated on the negative side

# TODO: Complete this code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# pred_df comes from processing just as NN_predict.py but loading DUDEz data

# Reset index to avoid issues with duplicate labels
pred_df.reset_index(inplace=True)

# Descriptive Stats
ligand_stats = pred_df.loc[pred_df['index'] == 'ligand', 'OCScore'].describe()
decoy_stats = pred_df.loc[pred_df['index'] == 'decoy', 'OCScore'].describe()

# Print Descriptive Stats
print("Ligand Stats:\n", ligand_stats)
print("\nDecoy Stats:\n", decoy_stats)

# Mean Difference
ligand_mean = pred_df.loc[pred_df['index'] == 'ligand', 'OCScore'].mean()
decoy_mean = pred_df.loc[pred_df['index'] == 'decoy', 'OCScore'].mean()

print(f"Ligand Mean: {ligand_mean}, Decoy Mean: {decoy_mean}")

# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=pred_df, x='OCScore', hue='index', shade=True)
plt.xlabel('OCScore')
plt.ylabel('Density')
plt.title('KDE Plot of OCScore for Ligands and Decoys')
plt.show()

# Percentage of Ligands Above the Median
median_score = pred_df['OCScore'].median()
ligand_above_median = pred_df.loc[(pred_df['index'] == 'ligand') & (pred_df['OCScore'] > median_score)].shape[0]
ligand_below_median = pred_df.loc[(pred_df['index'] == 'ligand') & (pred_df['OCScore'] <= median_score)].shape[0]
total_ligands = pred_df.loc[pred_df['index'] == 'ligand'].shape[0]

print(f"Ligands Above Median: {ligand_above_median}/{total_ligands} ({ligand_above_median / total_ligands * 100:.2f}%)")
print(f"Ligands Below Median: {ligand_below_median}/{total_ligands} ({ligand_below_median / total_ligands * 100:.2f}%)")

# Kolmogorov-Smirnov Test (KS Test)
ligand_scores = pred_df.loc[pred_df['index'] == 'ligand', 'OCScore']
decoy_scores = pred_df.loc[pred_df['index'] == 'decoy', 'OCScore']
ks_stat, p_value = ks_2samp(ligand_scores, decoy_scores)

print(f"KS Statistic: {ks_stat}, p-value: {p_value}")

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=pred_df['index'], y=pred_df['OCScore'])
plt.title('Box Plot of OCScore for Ligands and Decoys')
plt.show()

# Count Ligand Overlap with Decoy Range
decoy_min = pred_df.loc[pred_df['index'] == 'decoy', 'OCScore'].min()
decoy_max = pred_df.loc[pred_df['index'] == 'decoy', 'OCScore'].max()
overlapping_ligands = pred_df.loc[(pred_df['index'] == 'ligand') & (pred_df['OCScore'] >= decoy_min) & (pred_df['OCScore'] <= decoy_max)].shape[0]
total_ligands = pred_df.loc[pred_df['index'] == 'ligand'].shape[0]

print(f"Overlapping Ligands: {overlapping_ligands}/{total_ligands} ({overlapping_ligands / total_ligands * 100:.2f}%)")
