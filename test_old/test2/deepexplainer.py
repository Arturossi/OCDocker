import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. Preparar X_eval com as features corretas
X_eval = X_val.copy()

# 2. Converter para tensor
X_eval_tensor = torch.tensor(X_eval.to_numpy(), dtype=torch.float32).to("cuda")

# 3. Rodar SHAP sobre todas as amostras
shap_values_deep = deep_explainer.shap_values(X_eval_tensor)

# 4. Converter para DataFrame
shap_deep_df = pd.DataFrame(
    np.squeeze(shap_values_deep),  # Remove eixo extra da saída
    columns=X_eval.columns
)

import matplotlib.pyplot as plt

# Calcula importância relativa (%)
mean_abs_shap = np.abs(shap_values_deep[:, :, 0]).mean(axis=0)
relative_importance = (mean_abs_shap / mean_abs_shap.sum()) * 100

# Ordena
sorted_idx = np.argsort(relative_importance)[::-1]

# Plota
plt.figure(figsize=(10, 6))
plt.barh(
    y=np.array(X_train.columns)[sorted_idx][:20], 
    width=relative_importance[sorted_idx][:20]
)
plt.xlabel('Importância Relativa (%)')
plt.title('Importância dos Descritores (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300)
plt.close()

# 5. Gerar rótulos binários
y_eval = np.array([0 if t == "decoy" else 1 for t in df_dudez["type"]])

# 6. Calcular o score SHAP final
expected_value = deep_explainer.expected_value[0]
shap_score_full = shap_deep_df.sum(axis=1) + expected_value

# 7. Calcular AUC
auc = roc_auc_score(y_eval, shap_score_full)
print(f"AUC com SHAP sobre df_dudez: {auc:.4f}")