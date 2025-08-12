import seaborn as sns
import matplotlib.pyplot as plt

# Correlação entre valores SHAP
corr = shap_deep_df.corr()

# Visualizar como heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("SHAP value correlations")
plt.show()