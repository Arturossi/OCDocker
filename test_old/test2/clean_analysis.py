# screening_analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from matplotlib.colors import Normalize

# =========================
# 0) UTIL
# =========================
def bootstrap_ci_percentile(values, B=2000, alpha=0.05, seed=42, stat="median"):
    """IC por bootstrap (percentil) para vetor 1D."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    point = float(np.nanmedian(x) if stat == "median" else np.nanmean(x))
    if x.size == 1:
        return point, point, point
    boots = []
    n = x.size
    for _ in range(B):
        samp = rng.choice(x, size=n, replace=True)
        boots.append(np.nanmedian(samp) if stat == "median" else np.nanmean(samp))
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return point, float(lo), float(hi)

def _fmt_ci(m, lo, hi, dec=2):
    return f"{m:.{dec}f} [{lo:.{dec}f}–{hi:.{dec}f}]"

def build_summary_table(summary_targets, summary_pooled, models, eps=(1,5,10,20,30), include_pr_auc=False, pr_summary_targets=None, pr_summary_pooled=None):
    # EF_ROC mediana por receptor (1% e 5%)
    eps = tuple(int(e) for e in eps)
    st = summary_targets[summary_targets["metric"].isin([f"EF_ROC_{e}%" for e in eps])].copy()
    med = (st[st["model"].isin(models)]
           .assign(val=lambda d: d.apply(lambda r: _fmt_ci(r["median_across_targets"], r["CI95_lo"], r["CI95_hi"]), axis=1))
           .pivot(index="model", columns="metric", values="val")
           .reindex(models))

    # EF_ROC pooled (1% e 5%)
    sp = summary_pooled[summary_pooled["metric"].isin([f"EF_ROC_{e}%" for e in eps])].copy()
    poo = (sp[sp["model"].isin(models)]
           .assign(val=lambda d: d.apply(lambda r: _fmt_ci(r["pooled_value"], r["CI95_lo"], r["CI95_hi"]), axis=1))
           .pivot(index="model", columns="metric", values="val")
           .reindex(models))

    # Set the dict to rename for median and pooled columns (use dict comprehension)
    rename_dict_med = {f"EF_ROC_{e}%": f"Mediana EF-ROC {e}%" for e in eps}
    rename_dict_poo = {f"EF_ROC_{e}%": f"Pooled EF-ROC {e}%" for e in eps}

    out = med.rename(columns=rename_dict_med).join(
          poo.rename(columns=rename_dict_poo))

    if include_pr_auc and pr_summary_targets is not None and pr_summary_pooled is not None:
        # mesma lógica para PR-AUC (mediana e pooled)
        stp = pr_summary_targets[pr_summary_targets["metric"].eq("PR_AUC") & pr_summary_targets["model"].isin(models)].copy()
        med_pr = (stp.assign(val=lambda d: d.apply(lambda r: _fmt_ci(r["median_across_targets"], r["CI95_lo"], r["CI95_hi"]), axis=1))
                       .pivot(index="model", columns="metric", values="val")).reindex(models)
        spp = pr_summary_pooled[pr_summary_pooled["metric"].eq("PR_AUC") & pr_summary_pooled["model"].isin(models)].copy()
        poo_pr = (spp.assign(val=lambda d: d.apply(lambda r: _fmt_ci(r["pooled_value"], r["CI95_lo"], r["CI95_hi"]), axis=1))
                       .pivot(index="model", columns="metric", values="val")).reindex(models)
        out = (out.join(med_pr.rename(columns={"PR_AUC":"Mediana PR-AUC"}))
                  .join(poo_pr.rename(columns={"PR_AUC":"Pooled PR-AUC"})))
    return out

# =========================
# 1) MÉTRICAS EF
# =========================
def compute_enrichment_auc(
    df: pd.DataFrame,
    score_col: str,
    active_col: str,
    fprs: Union[float, List[float]] = (0.01, 0.05),
    flip_if_needed: bool = True,
    interpolation: str = "linear",
    ) -> Dict[str, float]:
    """EF_ROC(ε) = TPR(FPR=ε)/ε via curva ROC."""
    if isinstance(fprs, (float, int)):
        fprs = [float(fprs)]
    fprs = [float(e) for e in fprs]
    if any(e <= 0 or e >= 1 for e in fprs):
        raise ValueError("FPRs devem estar em (0,1).")

    sub = df[[score_col, active_col]].dropna()
    y = sub[active_col].astype(int).to_numpy()
    s = sub[score_col].astype(float).to_numpy()
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("Precisa de pelo menos um ativo e um inativo.")

    flipped = False
    if flip_if_needed:
        auc = roc_auc_score(y, s)
        if auc < 0.5:
            s = -s
            flipped = True

    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)

    def tpr_at(eps: float) -> float:
        if interpolation == "linear":
            return float(np.interp(eps, fpr, tpr))
        idx = np.searchsorted(fpr, eps, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        return float(tpr[idx])

    out = {f"EF_ROC_{int(round(e*100))}%": tpr_at(e) / e for e in fprs}
    out["_flipped"] = flipped
    return out

def EF_ROC_by_target_with_ci(
    df: pd.DataFrame,
    target_col: str,
    active_col: str,
    score_cols: List[str],
    fprs=(0.01, 0.05, 0.10, 0.20, 0.30),
    B=2000,
    seed=42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """EF_ROC por receptor + IC95% da mediana entre targets (por modelo)."""
    rows = []
    for m in score_cols:
        for tgt, g in df.groupby(target_col):
            res = compute_enrichment_auc(g, score_col=m, active_col=active_col, fprs=list(fprs))
            row = {"model": m, "target": tgt}
            for eps in fprs:
                row[f"EF_ROC_{int(round(eps*100))}%"] = res[f"EF_ROC_{int(round(eps*100))}%"]
            rows.append(row)
    per_target = pd.DataFrame(rows)

    out = []
    for m, g in per_target.groupby("model"):
        for eps in fprs:
            vals = g[f"EF_ROC_{int(round(eps*100))}%"].values
            point, lo, hi = bootstrap_ci_percentile(vals, B=B, seed=seed, stat="median")
            out.append({
                "model": m,
                "metric": f"EF_ROC_{int(round(eps*100))}%",
                "median_across_targets": point,
                "CI95_lo": lo,
                "CI95_hi": hi,
                "n_targets": len(vals),
            })
    summary = (pd.DataFrame(out)
               .sort_values(["metric", "median_across_targets"], ascending=[True, False])
               .reset_index(drop=True))
    return per_target, summary

def EF_ROC_pooled_with_ci(
    df: pd.DataFrame,
    active_col: str,
    score_cols: List[str],
    fprs=(0.01, 0.05, 0.10, 0.20, 0.30),
    B=2000,
    seed=42,
    ) -> pd.DataFrame:
    """EF_ROC pooled (dataset inteiro) + IC95% por bootstrap estratificado por classe."""
    rng = np.random.default_rng(seed)
    y = df[active_col].astype(int).to_numpy()
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    rows = []
    for m in score_cols:
        base = compute_enrichment_auc(df[[m, active_col]].rename(columns={m: "score"}).assign(__y=y),
                                      score_col="score", active_col="__y", fprs=list(fprs))
        obs = {f"EF_ROC_{int(round(e*100))}%": base[f"EF_ROC_{int(round(e*100))}%"] for e in fprs}

        boots = {f"EF_ROC_{int(round(e*100))}%": [] for e in fprs}
        for _ in range(B):
            samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            samp_idx = np.concatenate([samp_pos, samp_neg])
            g = df.iloc[samp_idx]
            res = compute_enrichment_auc(g, score_col=m, active_col=active_col, fprs=list(fprs))
            for e in fprs:
                boots[f"EF_ROC_{int(round(e*100))}%"].append(res[f"EF_ROC_{int(round(e*100))}%"])

        for e in fprs:
            vals = np.asarray(boots[f"EF_ROC_{int(round(e*100))}%"], float)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({
                "model": m,
                "metric": f"EF_ROC_{int(round(e*100))}%",
                "pooled_value": obs[f"EF_ROC_{int(round(e*100))}%"],
                "CI95_lo": float(lo),
                "CI95_hi": float(hi),
                "B": B,
            })
    return (pd.DataFrame(rows)
            .sort_values(["metric", "pooled_value"], ascending=[True, False])
            .reset_index(drop=True))

# =========================
# 2) MÉTRICAS PR-AUC
# =========================
def pr_auc_by_target_with_ci(
    df: pd.DataFrame,
    target_col: str,
    active_col: str,
    score_cols: List[str],
    B: int = 2000,
    seed: int = 42,
    agg: str = "median",  # "median" ou "mean"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """PR_AUC por receptor + IC95% da mediana/ média entre targets (por modelo)."""
    rows = []
    for m in score_cols:
        for tgt, g in df.groupby(target_col):
            if g[active_col].nunique() < 2:
                continue
            y = g[active_col].astype(int).to_numpy()
            if m not in g.columns:
                continue
            s = g[m].astype(float).to_numpy()
            pr = float(average_precision_score(y, s))
            rows.append({"model": m, "target": tgt, "PR_AUC": pr})
    per_target = pd.DataFrame(rows)

    out = []
    for m, g in per_target.groupby("model"):
        vals = g["PR_AUC"].to_numpy(float)
        point, lo, hi = bootstrap_ci_percentile(vals, B=B, seed=seed, stat=("mean" if agg == "mean" else "median"))
        out.append({
            "model": m,
            "metric": "PR_AUC",
            "median_across_targets": point,
            "CI95_lo": lo,
            "CI95_hi": hi,
            "n_targets": len(vals),
        })
    summary = (pd.DataFrame(out)
               .sort_values(["metric", "median_across_targets"], ascending=[True, False])
               .reset_index(drop=True))
    return per_target, summary

def pr_auc_pooled_with_ci(
    df: pd.DataFrame,
    active_col: str,
    score_cols: List[str],
    B: int = 2000,
    seed: int = 42,
    ) -> pd.DataFrame:
    """PR_AUC pooled (dataset inteiro) + IC95% via bootstrap estratificado por classe."""
    rng = np.random.default_rng(seed)
    y = df[active_col].astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("y tem uma única classe no dataset inteiro.")
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    rows = []
    for m in score_cols:
        if m not in df.columns:
            continue
        s_all = df[m].astype(float).to_numpy()
        obs = float(average_precision_score(y, s_all))
        boots = []
        for _ in range(B):
            samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            samp_idx = np.concatenate([samp_pos, samp_neg])
            y_b = y[samp_idx]
            s_b = s_all[samp_idx]
            boots.append(float(average_precision_score(y_b, s_b)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "model": m,
            "metric": "PR_AUC",
            "pooled_value": obs,
            "CI95_lo": float(lo),
            "CI95_hi": float(hi),
            "B": B,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })
    return (pd.DataFrame(rows)
            .sort_values(["metric", "pooled_value"], ascending=[True, False])
            .reset_index(drop=True))

# =========================
# 3) VISUAL — Heatmap PR_AUC + coluna Pooled
# =========================
def plot_pr_auc_with_pooled(
    pr_per_target: pd.DataFrame,      # ["model","target","PR_AUC"]
    pooled_results: pd.DataFrame,     # ["model","pooled_value"] OU ["model","metric","pooled_value"]
    *,
    cmap: str = "viridis",
    pooled_colname: str = "Pooled",
    metric_hint: str = "PR"           # usado se existir coluna 'metric'
    ):
    pr_matrix = pr_per_target.pivot(index="model", columns="target", values="PR_AUC")

    pooled_df = pooled_results.copy()
    if "metric" in pooled_df.columns:
        mask = pooled_df["metric"].astype(str).str.contains(metric_hint, case=False, na=False)
        pooled_df = pooled_df.loc[mask, ["model", "pooled_value"]].drop_duplicates("model", keep="last")
    else:
        pooled_df = pooled_df.loc[:, ["model", "pooled_value"]].drop_duplicates("model", keep="last")
    pooled_series = pooled_df.set_index("model")["pooled_value"]

    pr_matrix = pr_matrix.copy()
    pr_matrix[pooled_colname] = pooled_series.reindex(pr_matrix.index)
    pr_matrix = pr_matrix.sort_values(by=pooled_colname, ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pr_matrix) * 0.42)))
    sns.heatmap(
        pr_matrix, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        cbar_kws={'label': 'PR_AUC'}, linewidths=0.5, linecolor='0.8'
    )
    ax.set_title("PR_AUC por receptor + Pooled")
    ax.set_xlabel("Target")
    ax.set_ylabel("Modelo")

    nrows, ncols = pr_matrix.shape
    ax.add_patch(Rectangle((ncols - 1, 0), 1, nrows, fill=False, lw=2.0, ec="black"))
    plt.tight_layout()
    return fig, ax

# =========================
# 4) ORQUESTRADOR (1 chamada)
# =========================
def run_full_analysis(
    sfs: pd.DataFrame,
    target_col: str,
    active_col: str,
    score_cols: List[str],
    *,
    fprs=(0.01, 0.05, 0.10, 0.20, 0.30),
    B=2000,
    seed=42
    ):
    # EF
    ef_per_target, ef_summary_targets = EF_ROC_by_target_with_ci(
        sfs, target_col, active_col, score_cols, fprs=fprs, B=B, seed=seed
    )
    ef_summary_pooled = EF_ROC_pooled_with_ci(
        sfs, active_col, score_cols, fprs=fprs, B=B, seed=seed
    )

    # PR
    pr_per_target, pr_summary_targets = pr_auc_by_target_with_ci(
        sfs, target_col, active_col, score_cols, B=B, seed=seed, agg="median"
    )
    pr_summary_pooled = pr_auc_pooled_with_ci(
        sfs, active_col, score_cols, B=B, seed=seed
    )

    return {
        "ef_per_target": ef_per_target,
        "ef_summary_targets": ef_summary_targets,
        "ef_summary_pooled": ef_summary_pooled,
        "pr_per_target": pr_per_target,
        "pr_summary_targets": pr_summary_targets,
        "pr_summary_pooled": pr_summary_pooled
    }

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

# Inputs
target_col = "receptor"
active_col = "class"

# Scores preditivos (sem as colunas indesejadas)
score_cols = [c for c in sfs.columns if c not in ("class", "receptor", "ligand", "name")]

# ---- helper para formatar valor + IC (usado na tabela) ----
def _fmt_ci(val, lo, hi, digits=2):
    if pd.isna(val) or pd.isna(lo) or pd.isna(hi):
        return "-"
    return f"{val:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

out = run_full_analysis(
    sfs, target_col, active_col, score_cols,
    fprs=(0.01,0.05,0.10,0.20,0.30), B=2000, seed=42,
)

# ---- extrai os artefatos do orquestrador ----
ef_per_target       = out["ef_per_target"]
ef_summary_targets  = out["ef_summary_targets"]
ef_summary_pooled   = out["ef_summary_pooled"]
pr_per_target       = out["pr_per_target"]
pr_summary_targets  = out["pr_summary_targets"]
pr_summary_pooled   = out["pr_summary_pooled"]

# ---- figura: heatmap PR_AUC por receptor + coluna Pooled ----
fig_pr, ax_pr = plot_pr_auc_with_pooled(
    pr_per_target=pr_per_target,           # ["model","target","PR_AUC"]
    pooled_results=pr_summary_pooled,      # ["model","metric","pooled_value",...]
    cmap="viridis",
    pooled_colname="Pooled",               # nome da última coluna
    metric_hint="PR"                       # identifica a métrica no DF pooled
)
fig_pr.savefig("Heatmap_PR_AUC_pooled.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# ---- tabela-resumo (EF mediana por target + EF pooled; opcional PR mediana/pooled) ----
summary_table = build_summary_table(
    summary_targets=ef_summary_targets,
    summary_pooled=ef_summary_pooled,
    models=score_cols,
    include_pr_auc=True,                   # agora adiciona PR também
    pr_summary_targets=pr_summary_targets,
    pr_summary_pooled=pr_summary_pooled,
)
print(summary_table)

# ---- plot: EF_ROC pooled com IC95% (vários ε) ----
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_efauc_forest(
    summary: pd.DataFrame,       # summary_pooled OU summary_targets
    models=None,
    eps=(1, 5),
    sort_by=5,
    label_map: dict = None,
    row_height=0.4,
    marker_size=4,
    capsize=2,
    legend_position="inside",    # "inside" ou "outside"
    title=None,                  # se None, ajusta pelo kind
    dodge="auto",                # "auto" ou float
    auto_style=True,
    kind="pooled",               # "pooled" ou "median"
):
    # quais colunas usar
    if kind == "pooled":
        val_col, lo_col, hi_col = "pooled_value", "CI95_lo", "CI95_hi"
        if title is None: title = r"EF$_{ROC}$ pooled with CI$_{95\%}$"
    elif kind == "median":
        val_col, lo_col, hi_col = "median_across_targets", "CI95_lo", "CI95_hi"
        if title is None: title = r"EF$_{ROC}$ (median across targets) with CI$_{95\%}$"
    else:
        raise ValueError("kind deve ser 'pooled' ou 'median'.")

    # sanitiza eps e filtra
    eps = tuple(int(e) for e in eps)
    df = summary[summary["metric"].isin([f"EF_ROC_{e}%" for e in eps])].copy()
    if models is None:
        models = sorted(df["model"].unique())

    # ordenação pela métrica sort_by
    key = f"EF_ROC_{int(sort_by)}%"
    order = (df[df["metric"].eq(key)]
             .set_index("model")[[val_col]]
             .reindex(models)
             .dropna()
             .sort_values(val_col)
             .index.tolist())
    if not order:
        raise ValueError(f"No model found with metric {key}.")

    # dimensões
    n = len(order)
    fig_h = max(3.0, row_height * n)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # rótulos
    if label_map is None: label_map = {}
    ylabels = [label_map.get(m, m) for m in order]
    y = np.arange(n)

    # espaçamento vertical entre séries
    k = max(1, len(eps))
    if dodge == "auto":
        band = min(1.0, 0.55 + 0.08*min(k, 8))
        dy = band / k
    else:
        dy = float(dodge) / k

    ms, cs, alpha = marker_size, capsize, 0.9
    markers = ['o','s','^','D','v','P','X','*','h','>','<']
    all_lo, all_hi = [], []

    # faixas alternadas
    for idx in range(n):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, facecolor="0.95", zorder=0)

    # plot
    for i, e in enumerate(eps):
        metric = f"EF_ROC_{e}%"
        dfe = df[df["metric"].eq(metric)].set_index("model").reindex(order)
        x  = dfe[val_col].to_numpy(float)
        lo = dfe[lo_col].to_numpy(float)
        hi = dfe[hi_col].to_numpy(float)
        xerr = np.vstack((x - lo, hi - x))

        y_offset = (i - (k - 1) / 2) * dy
        m = markers[i % len(markers)] if auto_style else 'o'
        ax.errorbar(x, y + y_offset, xerr=xerr,
                    fmt=m, ms=ms, capsize=cs, elinewidth=1,
                    linestyle='none', alpha=alpha, label=f"ε={e}%")
        if np.isfinite(lo).any(): all_lo.append(np.nanmin(lo))
        if np.isfinite(hi).any(): all_hi.append(np.nanmax(hi))

    # estilo
    ax.axvline(1.0, linestyle='--', linewidth=1, color='0.6', alpha=0.8)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(ylabels)
    ax.set_xlabel(r"EF$_{ROC}$"); ax.set_title(title)
    xmin = float(np.nanmin(all_lo)) if all_lo else 0.0
    xmax = float(np.nanmax(all_hi)) if all_hi else 1.0
    span = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(max(0, xmin - 0.03*span), xmax + 0.05*span)
    ax.set_ylim(-0.5, n - 0.5); ax.margins(y=0.01)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # legenda
    if legend_position == "inside":
        ax.legend(title="Operating point", loc="lower right", frameon=False, ncol=min(k,4))
        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(left=0.28, right=0.98, top=0.93, bottom=0.06)
    else:
        ax.legend(title="Operating point", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(pad=0.2); fig.subplots_adjust(right=0.82)

    return fig, ax

def plot_efroc_rank_heatmap_rotated_all(
    summary,
    eps=(1,5,10,20,30),
    kind="pooled",
    cmap="viridis_r",
    sort_by_eps=5,                 # ordena pelos ranks no ε escolhido
    label_map=None,
    highlight_models=("OCScore",), # pode ser str, tupla/lista ou None
    highlight_color="red",
    highlight_lw=2,
    highlight_alpha=0.85,
    ):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize

    val_col = "pooled_value" if kind == "pooled" else "median_across_targets"
    eps = [int(e) for e in eps]

    # --------- montar rank por ε (linhas=ε, colunas=modelos) ---------
    ranks = {}
    for e in eps:
        dfe = (summary[summary["metric"].eq(f"EF_ROC_{e}%")]
               [["model", val_col]].dropna()
               .sort_values(val_col, ascending=False)
               .reset_index(drop=True))
        ranks[e] = {m: i+1 for i, m in enumerate(dfe["model"])}

    all_models = sorted(set().union(*[set(r.keys()) for r in ranks.values()]))
    dfR = pd.DataFrame(index=eps, columns=all_models, dtype=float)
    for e in eps:
        for m, r in ranks[e].items():
            dfR.loc[e, m] = r

    # --------- ordenar colunas pelo ε escolhido ---------
    if sort_by_eps in dfR.index:
        base_order = dfR.loc[sort_by_eps].sort_values().dropna().index
        dfR = dfR.reindex(columns=base_order)

    # rótulos (opcionalmente mapear nomes)
    if label_map is None: label_map = {}
    xlabels = [label_map.get(m, m) for m in dfR.columns]
    ylabels = [f"ε={e}%" for e in dfR.index]

    # --------- dimensões e fontes automáticas ---------
    n_models, n_eps = dfR.shape[1], dfR.shape[0]
    fs_xtick = max(7, min(12, 12 - 0.06*(n_models-10)))
    fs_cell  = max(7, min(12, 12 - 0.05*(n_models-12)))
    w = max(10, 0.55*n_models + 3.0)
    h = max(3.6, 0.8*n_eps + 1.6)

    fig, ax = plt.subplots(figsize=(w, h))

    # --------- HEATMAP com arestas explícitas (alinhado) ---------
    Z = dfR.values.astype(float)
    x = np.arange(n_models + 1) - 0.5
    y = np.arange(n_eps + 1) - 0.5
    vmin, vmax = np.nanmin(Z), np.nanmax(Z)
    pc = ax.pcolormesh(x, y, Z, cmap=cmap, shading="flat", vmin=vmin, vmax=vmax)

    # limites e ticks exatamente no centro das células
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(n_eps - 0.5, -0.5)  # origem no topo
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_eps))

    ax.set_xticklabels(
        xlabels, rotation=40, ha="right", rotation_mode="anchor", fontsize=fs_xtick
    )
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.tick_params(axis="x", pad=6)

    # --------- números nas células (cor adaptativa) ---------
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    for i in range(n_eps):
        for j in range(n_models):
            if np.isfinite(Z[i, j]):
                L = np.dot(cmap_obj(norm(Z[i, j]))[:3], [0.2126, 0.7152, 0.0722])
                ax.text(j, i, f"{int(Z[i, j])}", ha="center", va="center",
                        fontsize=fs_cell, color=("black" if L > 0.6 else "white"))

    # --------- destaque: borda contínua na(s) coluna(s) ---------
    if highlight_models:
        if isinstance(highlight_models, (str,)):
            highlight_models = [highlight_models]
        for hm in highlight_models:
            if hm in dfR.columns:
                j = list(dfR.columns).index(hm)
                rect = patches.Rectangle(
                    (j - 0.5, -0.5), 1, n_eps,  # coluna inteira
                    linewidth=highlight_lw, edgecolor=highlight_color,
                    facecolor="none", alpha=highlight_alpha, zorder=3
                )
                ax.add_patch(rect)

    # --------- título e colorbar ---------
    ax.set_title("Ranking por ponto de operação", fontsize=16, pad=12)
    cbar = fig.colorbar(pc, ax=ax, shrink=0.75)
    cbar.set_label("posição no ranking", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)  # espaço extra p/ labels do X
    return fig, ax

# Import necessário p/ ticks de 1 em 1 (se ainda não tiver)
from matplotlib.ticker import MultipleLocator

FPRS = (0.01, 0.05, 0.10, 0.20, 0.30)

# ---- Forest plot: EF_ROC pooled com IC95% (vários ε) ----
# Usa as mesmas FPRs do pipeline, mas em %
EPS = tuple(int(round(e*100)) for e in FPRS)      # ex.: (1, 5, 10, 20, 30)
SORT_BY = 5 if 5 in EPS else EPS[0]               # ordena pela coluna de 5% se existir

# (opcional) nomes bonitos p/ modelos na figura
label_map = {
    "mean_score": "Média Dockings",
    # adicione outros: "SMINA_X": "SMINA X", ...
}

fig_ef, ax_ef = plot_efauc_forest(
    summary=ef_summary_pooled,
    models=score_cols,
    eps=EPS,
    sort_by=SORT_BY,
    label_map=label_map,
    row_height=1,#0.60,
    marker_size=5,
    capsize=2,
    legend_position="outside",
    title=r"EF$_{ROC}$ pooled com IC$_{95\%}$",
    dodge="auto",
    auto_style=True
)
fig_ef.savefig("Fig_EF_ROC_pooled.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

fig_med, ax_med = plot_efauc_forest(
    summary=ef_summary_targets,        # ou ef_summary_targets, se for o seu nome
    models=score_cols,
    eps=EPS,
    sort_by=SORT_BY,
    label_map=label_map,
    row_height=1,#0.60,
    marker_size=5,
    capsize=2,
    legend_position="outside",
    title=r"EF$_{ROC}$ (mediana entre receptores) com CI$_{95\%}$",
    dodge="auto",
    auto_style=True,
    kind="median"                   # <- chave da troca
)
fig_med.savefig("Fig_EF_ROC_median.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def _composite_order(rank_by_eps, sort_by, weights=None):
    eps_list = sorted(rank_by_eps.keys())
    if weights is None:
        weights = {e: 1.0 for e in eps_list}
    # junta ranks por modelo
    models = set().union(*[set(df["model"]) for df in rank_by_eps.values()])
    rows = []
    for m in models:
        r = {"model": m}
        for e in eps_list:
            s = rank_by_eps[e].set_index("model")
            r[f"r{e}"] = float(s.loc[m, "mean_rank"]) if m in s.index else np.nan
        rows.append(r)
    R = pd.DataFrame(rows).set_index("model")
    # média ponderada ignorando NaN
    w = np.array([weights[e] for e in eps_list], float)
    V = R[[f"r{e}" for e in eps_list]].to_numpy(float)
    mask = ~np.isnan(V)
    num = np.nansum(V * w, axis=1)
    den = np.nansum(mask * w, axis=1)
    comp = num / np.where(den == 0, np.nan, den)
    R["comp"] = comp
    # desempate: ε de referência, depois demais ε em ordem
    tie_cols = [f"r{sort_by}"] + [f"r{e}" for e in eps_list if e != sort_by]
    order = (R.sort_values(by=["comp"] + tie_cols, ascending=True)
               .index.tolist())
    return order

def avg_rank_with_ci(per_target: pd.DataFrame, eps=5, models=None, B=2000, seed=42):
    col = f"EF_ROC_{int(eps)}%"
    P = per_target.pivot(index="target", columns="model", values=col)
    if models is not None:
        P = P[[m for m in models if m in P.columns]]
    R = P.rank(axis=1, ascending=False, method="average")
    mean_rank = R.mean(0)
    med_rank  = R.median(0)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(R))
    boots = []
    for _ in range(B):
        S = R.iloc[rng.choice(idx, size=len(idx), replace=True)]
        boots.append(S.mean(0))
    Bdf = pd.DataFrame(boots)
    out = pd.DataFrame({
        "model": R.columns,
        "mean_rank": mean_rank.values,
        "CI95_lo": Bdf.quantile(0.025).values,
        "CI95_hi": Bdf.quantile(0.975).values,
        "median_rank": med_rank.values,
        "n_targets": len(R)
    }).sort_values("mean_rank")
    return out

def plot_rank_forest_multi(rank_by_eps: dict, sort_by=5, sort_mode="ref",
                           label_map=None, row_height=0.4, marker_size=4, capsize=2,
                           legend_position="outside",
                           title=r"EF$_{ROC}$ — rank médio por alvo"):
    eps_list = sorted(rank_by_eps.keys())
    if sort_by not in eps_list:
        sort_by = eps_list[0]

    # ordem dos modelos
    if sort_mode == "ref":
        order = (rank_by_eps[sort_by]
                 .sort_values("mean_rank", ascending=False)["model"].tolist())
    elif sort_mode == "composite":
        order = _composite_order(rank_by_eps, sort_by=sort_by)
    else:
        raise ValueError("sort_mode deve ser 'ref' ou 'composite'.")

    # figura
    n = len(order)
    fig_h = max(3.0, row_height * n)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    if label_map is None: label_map = {}
    ylabels = [label_map.get(m, m) for m in order]
    y = np.arange(n)

    # deslocamento
    k = len(eps_list)
    band = min(1.0, 0.55 + 0.08*min(k, 8))
    dy = band / max(1, k)

    markers = ['o','s','^','D','v','P','X','>','<','h','*']

    # fundo listrado
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i-0.5, i+0.5, color='0.95', zorder=0)

    # plota todos ε
    for i, e in enumerate(eps_list):
        df = (rank_by_eps[e].set_index("model").reindex(order).reset_index())
        x = df["mean_rank"].to_numpy(float)
        lo = df["CI95_lo"].to_numpy(float)
        hi = df["CI95_hi"].to_numpy(float)
        xerr = np.vstack((x-lo, hi-x))
        ax.errorbar(x, y + (i-(k-1)/2)*dy,
                    xerr=xerr, fmt=markers[i%len(markers)],
                    ms=marker_size, capsize=2, elinewidth=1,
                    linestyle='none', label=f"ε={e}%")

    # estilo
    ax.grid(True, axis='x', ls=':', lw=0.8, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(ylabels)
    ax.set_xlabel("rank médio entre alvos"); ax.set_title(title)
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylim(-0.5, n-0.5); ax.margins(y=0.01)

    # legenda
    if legend_position == "inside":
        ax.legend(title="Operating point", loc="lower right", frameon=False, ncol=min(k,4))
        fig.tight_layout(pad=0.2); fig.subplots_adjust(left=0.28, right=0.98, top=0.93, bottom=0.06)
    else:
        ax.legend(title="Operating point", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(pad=0.2); fig.subplots_adjust(right=0.82)

    return fig, ax


rank_by_eps = {e: avg_rank_with_ci(ef_per_target, eps=e, models=score_cols)
               for e in EPS}

# 2) plota tudo em UMA imagem (ordenando por ε=5%)
fig, ax = plot_rank_forest_multi(rank_by_eps, sort_by=5, row_height=1,#0.6,
                                 title=r"EF$_{ROC}$ rank médio por alvo (CI95%)")
fig.savefig("Forest_AvgRank_multi.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

def _orientation_map_global(df, active_col, score_cols):
    """Decide orientação por modelo usando AUC pooled (evita flip alvo a alvo)."""
    y_all = df[active_col].astype(int).to_numpy()
    orient = {}
    for m in score_cols:
        if m in df.columns:
            s = df[m].astype(float).to_numpy()
            try:
                auc = roc_auc_score(y_all, s)
                orient[m] = -1 if auc < 0.5 else +1
            except ValueError:
                orient[m] = +1
    return orient

def roc_auc_per_target(df, target_col, active_col, score_cols, flip_by_global=True):
    orient = _orientation_map_global(df, active_col, score_cols) if flip_by_global else {m:+1 for m in score_cols}
    rows = []
    for tgt, g in df.groupby(target_col):
        y = g[active_col].astype(int).to_numpy()
        if np.unique(y).size < 2:
            continue
        for m in score_cols:
            if m not in g.columns: 
                continue
            s = g[m].astype(float).to_numpy() * orient.get(m, +1)
            try:
                auc = float(roc_auc_score(y, s))
            except ValueError:
                continue
            rows.append({"model": m, "target": tgt, "ROC_AUC": auc})
    return pd.DataFrame(rows)

def roc_auc_pooled(df, active_col, score_cols, flip_by_global=True):
    orient = _orientation_map_global(df, active_col, score_cols) if flip_by_global else {m:+1 for m in score_cols}
    y = df[active_col].astype(int).to_numpy()
    out = []
    for m in score_cols:
        if m not in df.columns: 
            continue
        s = df[m].astype(float).to_numpy() * orient.get(m, +1)
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            continue
        out.append({"model": m, "Pooled": auc})
    return pd.DataFrame(out)

# ==== 2) Matriz (modelos x alvos+Pooled) para heatmap ====
def make_roc_heatmap_matrix(roc_per_tgt, roc_pooled, score_cols=None):
    M = roc_per_tgt.pivot(index="model", columns="target", values="ROC_AUC")
    M = M.reindex(index=[m for m in (score_cols or M.index) if m in M.index])
    M["Pooled"] = M.index.map(dict(roc_pooled.set_index("model")["Pooled"]))
    # ordenar por pooled (desc)
    M = M.sort_values("Pooled", ascending=False)
    return M

# ==== 3) Plot do heatmap (matplotlib puro) ====
def plot_auc_heatmap(M, title="ROC_AUC por receptor + Pooled", fmt="{:.2f}",
                     pooled_colname="Pooled", cmap_name="viridis"):
    """
    M: DataFrame (modelos x [alvos... , Pooled]) já ordenado.
    Visual tipo seaborn:
      - sem outline preto do gráfico (spines off)
      - rótulos sem rotação
      - grade leve entre células
      - coluna 'Pooled' destacada
    """
    nrows, ncols = M.shape
    fig_w = max(12, 0.55*(ncols+3))
    fig_h = max(6, 0.42*max(1, nrows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    # bordas das células -> ticks centralizados
    x_edges = np.arange(-0.5, ncols + 0.5, 1.0)
    y_edges = np.arange(-0.5, nrows + 0.5, 1.0)

    # colormap e escala
    norm = Normalize(vmin=0.5, vmax=1.0)
    cmap = plt.get_cmap(cmap_name)
    Z = M.values.astype(float)

    # heatmap com grade leve (sem contorno preto externo)
    pc = ax.pcolormesh(
        x_edges, y_edges, Z,
        cmap=cmap, norm=norm, shading="flat",
        edgecolors="0.85", linewidth=0.5
    )

    # ticks/rótulos (sem rotação)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(M.columns)   # sem rotation
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(M.index)

    # limites: primeira linha no topo
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    # remover outline do gráfico (spines)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # destacar 'Pooled'
    if pooled_colname in M.columns:
        j = M.columns.get_loc(pooled_colname)
        ax.add_patch(plt.Rectangle((j-0.5, -0.5), 1, nrows, fill=False, ec="black", lw=2.0))

    # anotações com cor legível
    for i in range(nrows):
        for j in range(ncols):
            v = Z[i, j]
            if np.isfinite(v):
                rgba = cmap(norm(v))
                lum = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
                txt_color = "black" if lum > 0.6 else "white"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8, color=txt_color)

    # colorbar (sem outline preto também)
    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label("ROC_AUC")
    cbar.outline.set_visible(False) # type: ignore

    ax.set_title(title)
    ax.set_xlabel("Target")
    ax.set_ylabel("Modelo")
    return fig, ax

# ===== Execução =====
roc_tgt = roc_auc_per_target(sfs, target_col, active_col, score_cols, flip_by_global=True)
roc_pool = roc_auc_pooled(sfs, active_col, score_cols, flip_by_global=True)
M = make_roc_heatmap_matrix(roc_tgt, roc_pool, score_cols=score_cols)

fig, ax = plot_auc_heatmap(M, title="ROC_AUC por receptor + Pooled")
fig.savefig("Heatmap_ROC_AUC_por_receptor.png", dpi=300, bbox_inches="tight")
