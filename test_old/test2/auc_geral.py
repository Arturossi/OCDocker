import pandas as pd

# Suponha que df seja seu DataFrame
# Seleciona colunas cujo nome começa com SMINA, VINA, PLANTS ou ODDT
sfs = data['X_val'].filter(regex=r'^(SMINA|VINA|PLANTS|ODDT)').copy()
sfs["OCScore"] = neural.prediction
sfs["class"] = data['y_val']

# Selecionar apenas as colunas desejadas
scores_df = sfs.filter(regex=r'^(SMINA|VINA|PLANTS|ODDT)')

# Calcular média por linha
sfs['mean_score'] = scores_df.mean(axis=1)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.cm as cm
import numpy as np

# Classe real
y_true = sfs['class']

sfs["ligand"] = df_dudez['ligand']
sfs["receptor"] = df_dudez['receptor']
sfs["name"] = df_dudez['name']

# Scores preditivos (sem as colunas indesejadas)
score_cols = [c for c in sfs.columns if c not in ("class", "receptor", "ligand", "name")]

# Calcular ROC e AUC
roc_data = []
for col in score_cols:
    fpr, tpr, _ = roc_curve(y_true, sfs[col])
    roc_auc = auc(fpr, tpr)
    roc_data.append((col, fpr, tpr, roc_auc))

# Ordenar por AUC (opcional, melhora a leitura)
roc_data.sort(key=lambda x: x[3], reverse=True)

# Gerar paleta de cores com base em `tab20`, `tab20b`, etc.
num_curves = len(roc_data)
color_map = plt.get_cmap('tab20')  # ou tab20b, tab20c, etc.
colors = [color_map(i % 20) for i in range(num_curves)]

# Plot
plt.figure(figsize=(12, 8))
for i, (col, fpr, tpr, roc_auc) in enumerate(roc_data):
    plt.plot(fpr, tpr, label=f"{col} (AUC = {roc_auc:.2f})", color=colors[i])

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Scoring Functions')
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('auc.png', dpi=300)
#plt.show()

# Enrichment Factor

import pandas as pd
from typing import List, Dict, Tuple, Union

def compute_enrichment_rank(
        df: pd.DataFrame,
        score_col: str,
        active_col: str,
        cutoffs: Union[float, List[float]] = [0.01, 0.05, 0.10]
    ) -> Dict[str, float]:
    """
    Calcula o Enrichment Factor (EF) para os percentis especificados.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com pelo menos duas colunas: scores e indicador de ativo.
    score_col : str
        Nome da coluna de score (ordenação decrescente = melhores compostos).
    active_col : str
        Nome da coluna booleana ou 0/1 que indica se o composto é ativo.
    cutoffs : float ou list de float
        Frações do ranking a serem avaliadas (e.g. 0.01 = top 1%).

    Retorna
    -------
    Dict[str, float]
        Dicionário mapeando 'EF_1%', 'EF_5%', etc. aos valores calculados.
    """

    if isinstance(cutoffs, float):
        cutoffs = [cutoffs]

    N = len(df)
    n_actives = df[active_col].sum()
    if n_actives == 0:
        raise ValueError("Não há compostos ativos no DataFrame.")

    # ordena decrescentemente pelo score
    df_sorted = df.sort_values(by=score_col, ascending=False)

    results = {}
    for x in cutoffs:
        k = max(int(round(x * N)), 1)
        n_top = df_sorted.iloc[:k][active_col].sum()
        ef = (n_top / k) / (n_actives / N)
        pct_label = int(x * 100)
        results[f"EF_{pct_label}%"] = ef

    return results

def compute_enrichment_auc(
    df: pd.DataFrame,
    score_col: str,
    active_col: str,
    fprs: Union[float, List[float]] = (0.01, 0.05, 0.10),
    flip_if_needed: bool = True,
    interpolation: str = "linear"
    ) -> Dict[str, float]:
    """
    Calcula o Enrichment Factor pela definição na curva ROC (EF_AUC),
    isto é, EF_AUC(ε) = TPR(FPR=ε) / ε, para uma ou mais tolerâncias ε.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com pelo menos as colunas de score e rótulo (ativo).
    score_col : str
        Nome da coluna de score (maior = melhor).
    active_col : str
        Nome da coluna booleana ou 0/1 indicando se o composto é ativo.
    fprs : float ou list de float
        Lista de valores de FPR (entre 0 e 1; ex.: 0.01 = 1%).
    flip_if_needed : bool
        Se True, inverte o sinal do score quando o ROC AUC pooled < 0.5.
    interpolation : {"linear","step"}
        Modo de obter TPR no FPR desejado:
        - "linear": interpola linearmente na curva ROC (np.interp)
        - "step": usa o último ponto com FPR <= ε (degrau à esquerda)

    Retorna
    -------
    Dict[str, float]
        Dicionário mapeando "EF_AUC_1%", "EF_AUC_5%", ... para os valores.
        (Adicionalmente inclui a chave "_flipped" com True/False.)
    """
    # Segurança básica
    if isinstance(fprs, (float, int)):
        fprs = [float(fprs)]
    fprs = [float(eps) for eps in fprs]
    if any((eps <= 0 or eps >= 1) for eps in fprs):
        raise ValueError("Todos os valores de FPR (ε) devem estar em (0,1).")

    # Seleção e limpeza
    sub = df[[score_col, active_col]].dropna()
    if sub.empty:
        raise ValueError("DataFrame sem dados válidos após remoção de NaNs.")
    y = sub[active_col].astype(int).to_numpy()
    s = sub[score_col].astype(float).to_numpy()

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("É necessário haver ao menos um ativo e um inativo.")

    # Orientação do score (opcional)
    flipped = False
    if flip_if_needed:
        auc = roc_auc_score(y, s)
        if auc < 0.5:
            s = -s
            flipped = True

    # Curva ROC
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)

    # Interpolação
    def tpr_at(eps: float) -> float:
        if interpolation == "linear":
            # np.interp assume fpr crescente; extrapola com bordas se necessário
            return float(np.interp(eps, fpr, tpr))
        elif interpolation == "step":
            idx = np.searchsorted(fpr, eps, side="right") - 1
            idx = max(0, min(idx, len(tpr) - 1))
            return float(tpr[idx])
        else:
            raise ValueError("interpolation deve ser 'linear' ou 'step'.")

    # EF_AUC(ε) = TPR(FPR=ε) / ε
    out: Dict[str, float] = {}
    for eps in fprs:
        t = tpr_at(eps)
        out[f"EF_AUC_{int(round(eps*100))}%"] = t / eps

    out["_flipped"] = flipped
    return out

# monta um DataFrame de EF
table_rank = (
    pd.DataFrame([
        {"score_col_rank": col, **compute_enrichment_rank(sfs, col, "class", [0.01,0.05,0.10])}
        for col in score_cols
    ])
    .set_index("score_col_rank")
)

table_auc = (
    pd.DataFrame([
        {"score_col_auc": col, **compute_enrichment_auc(sfs, col, "class", fprs=[0.01] + np.arange(0.05, 0.31, 0.05).tolist())}
        for col in score_cols
    ])
    .set_index("score_col_auc")
)

# (opcional) remover a coluna de orientação
table_auc = table_auc.drop(columns="_flipped")

# (opcional) alinhar índice ao da tabela de rank e juntar tudo
table_auc.index.name = "score_col_rank"

def bootstrap_ci_percentile(values, B=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.array(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    point = np.nanmedian(x)
    if len(x) == 1:
        return point, point, point
    boots = []
    n = len(x)
    for _ in range(B):
        samp = rng.choice(x, size=n, replace=True)
        boots.append(np.nanmedian(samp))
    lo = np.percentile(boots, 2.5)
    hi = np.percentile(boots, 97.5)
    return point, lo, hi

# --- 1) IC95% da mediana por receptor (para vários modelos e FPRs) ---
def ef_auc_by_target_with_ci(
    df: pd.DataFrame,
    target_col: str,
    active_col: str,
    score_cols: List[str],
    fprs=(0.01, 0.05),
    B=2000,
    seed=42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # calcula EF_AUC por receptor para cada modelo
    rows = []
    for m in score_cols:
        for tgt, g in df.groupby(target_col):
            res = compute_enrichment_auc(g, score_col=m, active_col=active_col, fprs=list(fprs))
            row = {"model": m, "target": tgt}
            for eps in fprs:
                row[f"EF_AUC_{int(round(eps*100))}%"] = res[f"EF_AUC_{int(round(eps*100))}%"]
            rows.append(row)
    per_target = pd.DataFrame(rows)

    # agrega mediana entre receptores + IC95% via bootstrap por receptor
    out = []
    for m, g in per_target.groupby("model"):
        for eps in fprs:
            vals = g[f"EF_AUC_{int(round(eps*100))}%"].values
            point, lo, hi = bootstrap_ci_percentile(vals, B=B, seed=seed)
            out.append({
                "model": m,
                "metric": f"EF_AUC_{int(round(eps*100))}%",
                "median_across_targets": point,
                "CI95_lo": lo,
                "CI95_hi": hi,
                "n_targets": len(vals)
            })
    summary = pd.DataFrame(out).sort_values(["metric","median_across_targets"], ascending=[True, False])
    return per_target, summary

# --- 2) IC95% pooled (estratificado por classe) para EF_AUC ---
def ef_auc_pooled_with_ci(
    df: pd.DataFrame,
    active_col: str,
    score_cols: List[str],
    fprs=(0.01, 0.05),
    B=2000,
    seed=42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = df[active_col].astype(int).to_numpy()
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    rows = []
    for m in score_cols:
        # valor observado (pooled “de verdade”)
        base = compute_enrichment_auc(df[[m, active_col]].rename(columns={m:"score"}).assign(__y=y),
                                      score_col="score", active_col="__y", fprs=list(fprs))
        obs = {f"EF_AUC_{int(round(e*100))}%": base[f"EF_AUC_{int(round(e*100))}%"] for e in fprs}

        # bootstrap estratificado por classe
        boots = {f"EF_AUC_{int(round(e*100))}%": [] for e in fprs}
        for _ in range(B):
            samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            samp_idx = np.concatenate([samp_pos, samp_neg])
            g = df.iloc[samp_idx]
            res = compute_enrichment_auc(g, score_col=m, active_col=active_col, fprs=list(fprs))
            for e in fprs:
                boots[f"EF_AUC_{int(round(e*100))}%"].append(res[f"EF_AUC_{int(round(e*100))}%"])

        for e in fprs:
            vals = np.array(boots[f"EF_AUC_{int(round(e*100))}%"], dtype=float)
            lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            rows.append({
                "model": m,
                "metric": f"EF_AUC_{int(round(e*100))}%",
                "pooled_value": obs[f"EF_AUC_{int(round(e*100))}%"],
                "CI95_lo": float(lo),
                "CI95_hi": float(hi),
                "B": B
            })
    return pd.DataFrame(rows).sort_values(["metric","pooled_value"], ascending=[True, False])

# parâmetros
target_col = "receptor"
active_col = "class"

# 1) por receptor: EF_AUC@1% e @5% (mediana + IC95%)
per_target, summary_targets = ef_auc_by_target_with_ci(
    sfs, target_col, active_col, score_cols, fprs=(0.01, 0.05, 0.1, 0.2, 0.3), B=2000, seed=42
)

# 2) pooled: EF_AUC@1% e @5% (valor + IC95% estratificado)
summary_pooled = ef_auc_pooled_with_ci(
    sfs, active_col, score_cols, fprs=(0.01, 0.05, 0.1, 0.2, 0.3), B=2000, seed=42
)

# ====== 3) PR-AUC por receptor rápido (para alimentar a figura por receptor) ======
# Caso você já tenha pr_per_target, pule este bloco.
from sklearn.metrics import average_precision_score

rows = []
for m in score_cols:
    # Usamos per_target para pegar a lista de receptores presentes
    for tgt in sorted(per_target["target"].unique()):
        g = sfs[sfs[target_col] == tgt]
        if g[active_col].nunique() < 2:
            continue
        y = g[active_col].astype(int).to_numpy()
        if m not in g.columns:
            continue
        s = g[m].astype(float).to_numpy()
        pr = average_precision_score(y, s)
        rows.append({"model": m, "target": tgt, "PR_AUC": pr})
pr_per_target = pd.DataFrame(rows)

# ============================================================
# 2) Heatmap PR_AUC por receptor + coluna "Pooled"
#    (aceita pooled_results com/sem coluna 'metric')
# ============================================================
def plot_pr_auc_with_pooled(
    pr_per_target: pd.DataFrame,
    pooled_results: pd.DataFrame,
    *,
    cmap: str = "viridis",
    pooled_colname: str = "Pooled",
    metric_hint: str = "PR"   # substring para localizar a métrica certa quando existir coluna 'metric'
):
    """
    pr_per_target: DataFrame ["model","target","PR_AUC"]
    pooled_results: DataFrame com:
        - OU ["model","pooled_value"]
        - OU ["model","metric","pooled_value"] (ex.: summary com várias métricas)
    """
    # matriz modelos × targets
    pr_matrix = pr_per_target.pivot(index="model", columns="target", values="PR_AUC")

    # extrai série pooled por modelo
    pooled_df = pooled_results.copy()
    if "metric" in pooled_df.columns:
        mask = pooled_df["metric"].astype(str).str.contains(metric_hint, case=False, na=False)
        pooled_df = pooled_df.loc[mask, ["model", "pooled_value"]].drop_duplicates("model", keep="last")
    else:
        pooled_df = pooled_df.loc[:, ["model", "pooled_value"]].drop_duplicates("model", keep="last")

    pooled_series = pooled_df.set_index("model")["pooled_value"]

    # alinha e adiciona coluna Pooled ao fim
    pr_matrix = pr_matrix.copy()
    pr_matrix[pooled_colname] = pooled_series.reindex(pr_matrix.index)

    # ordena modelos pelo pooled (desc)
    pr_matrix = pr_matrix.sort_values(by=pooled_colname, ascending=False)

    # plota
    fig, ax = plt.subplots(figsize=(12, max(6, len(pr_matrix) * 0.42)))
    sns.heatmap(
        pr_matrix, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        cbar_kws={'label': 'PR_AUC'}, linewidths=0.5, linecolor='0.8'
    )
    ax.set_title("PR_AUC por receptor + coluna Pooled")
    ax.set_xlabel("Target")
    ax.set_ylabel("Modelo")

    # destaque na coluna pooled
    nrows, ncols = pr_matrix.shape
    ax.add_patch(Rectangle((ncols - 1, 0), 1, nrows, fill=False, lw=2.0, ec="black"))

    plt.tight_layout()
    return fig, ax

fig, ax = plot_pr_auc_with_pooled(
    pr_per_target=pr_per_target,          # sua tabela por receptor
    pooled_results=summary_pr_pooled,     # ATENÇÃO: pooled de PR_AUC
    cmap="viridis",
    pooled_colname="Pooled"
)
fig.savefig("heatmap_PR_AUC_pooled.png", dpi=300, bbox_inches="tight", pad_inches=0.02)