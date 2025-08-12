import numpy as np

# Calcular variância SHAP por feature
shap_variances = shap_deep_df.var()
top_features = shap_variances.sort_values(ascending=False).head(30).index  # ajuste N aqui

# Subset do DataFrame
filtered_df = shap_deep_df[top_features]

import seaborn as sns
import matplotlib.pyplot as plt

# Correlação entre valores SHAP
corr = filtered_df.corr()

# Visualizar como heatmap
plt.figure(figsize=(12, 10))
#sns.heatmap(corr, cmap='coolwarm', center=0)
sns.clustermap(corr, cmap='coolwarm', center=0, figsize=(10, 10))
plt.title("SHAP value correlations")
plt.show()

top_corr = shap_deep_df.corrwith(shap_deep_df['countM']).sort_values(ascending=False)

top_corr_no_nan = top_corr.dropna()

print(top_corr_no_nan.head(11)[1:])
print(top_corr_no_nan.tail(10))