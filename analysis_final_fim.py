from matplotlib.ticker import MultipleLocator

def _fmt_ci(m, lo, hi, dec=2):
    return f"{m:.{dec}f} [{lo:.{dec}f}–{hi:.{dec}f}]"

def build_summary_table(summary_targets, summary_pooled, models, include_pr_auc=False, pr_summary_targets=None, pr_summary_pooled=None):
    # EF_AUC mediana por receptor (1% e 5%)
    st = summary_targets[summary_targets["metric"].isin(["EF_AUC_1%","EF_AUC_5%"])].copy()
    med = (st[st["model"].isin(models)]
           .assign(val=lambda d: d.apply(lambda r: _fmt_ci(r["median_across_targets"], r["CI95_lo"], r["CI95_hi"]), axis=1))
           .pivot(index="model", columns="metric", values="val")
           .reindex(models))

    # EF_AUC pooled (1% e 5%)
    sp = summary_pooled[summary_pooled["metric"].isin(["EF_AUC_1%","EF_AUC_5%"])].copy()
    poo = (sp[sp["model"].isin(models)]
           .assign(val=lambda d: d.apply(lambda r: _fmt_ci(r["pooled_value"], r["CI95_lo"], r["CI95_hi"]), axis=1))
           .pivot(index="model", columns="metric", values="val")
           .reindex(models))

    out = med.rename(columns={"EF_AUC_1%":"Mediana EF_AUC@1%","EF_AUC_5%":"Mediana EF_AUC@5%"}).join(
          poo.rename(columns={"EF_AUC_1%":"Pooled EF_AUC@1%","EF_AUC_5%":"Pooled EF_AUC@5%"}))

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

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_pooled_efauc(
    summary_pooled: pd.DataFrame,
    models=None,
    eps=(1, 5),
    sort_by=5,
    label_map: dict = None,
    row_height=0.4,          # decrease if you want more compact rows
    marker_size=4,            # fixed marker size
    capsize=2,
    legend_position="inside", # "inside" (compact) or "outside"
    title=r"EF$_{AUC}$ pooled with CI$_{95\%}$",
    dodge="auto",             # "auto" or float: controls vertical spacing between series per row
    auto_style=True           # alternate markers/transparency automatically across series
    ):
    # sanitize eps
    eps = tuple(int(e) for e in eps)

    # filter metrics present in eps
    df = summary_pooled[summary_pooled["metric"].isin([f"EF_AUC_{e}%" for e in eps])].copy()
    if models is None:
        models = sorted(df["model"].unique())

    # ordering by the chosen key
    key = f"EF_AUC_{int(sort_by)}%"
    order = (df[df["metric"].eq(key)]
             .set_index("model")[["pooled_value"]]
             .reindex(models)
             .dropna()
             .sort_values("pooled_value")
             .index.tolist())
    if not order:
        raise ValueError(f"No model found with metric {key}.")

    # figure dimensions
    n = len(order)
    fig_h = max(3.0, row_height * n)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # y labels
    if label_map is None:
        label_map = {}
    ylabels = [label_map.get(m, m) for m in order]

    # base y positions (one row per model)
    y = np.arange(n)

    # ---- spacing & styling for many series ----
    k = max(1, len(eps))
    if dodge == "auto":
        # increased total band for more vertical separation
        band = min(1.0, 0.55 + 0.08*min(k, 8))
        dy = band / k
    else:
        dy = float(dodge) / k

    # fixed marker size
    ms = marker_size
    cs = capsize
    alpha = 0.9

    # marker shapes
    markers = ['o','s','^','D','v','P','X','*','h','>','<']

    all_lo, all_hi = [], []

    # ---- background stripes for readability ----
    for idx in range(n):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, facecolor="0.95", zorder=0)

    # ---- plot loop ----
    for i, e in enumerate(eps):
        metric = f"EF_AUC_{e}%"
        dfe = (df[df["metric"].eq(metric)]
               .set_index("model")
               .reindex(order))

        x  = dfe["pooled_value"].to_numpy(float)
        lo = dfe["CI95_lo"].to_numpy(float)
        hi = dfe["CI95_hi"].to_numpy(float)
        xerr = np.vstack((x - lo, hi - x))

        y_offset = (i - (k - 1) / 2) * dy
        m = markers[i % len(markers)] if auto_style else 'o'
        z = 3 + i

        ax.errorbar(
            x, y + y_offset, xerr=xerr,
            fmt=m, ms=ms, capsize=cs, elinewidth=1,
            linestyle='none', alpha=alpha, zorder=z,
            label=f"ε={e}%"
        )

        if np.isfinite(lo).any():
            all_lo.append(np.nanmin(lo))
        if np.isfinite(hi).any():
            all_hi.append(np.nanmax(hi))

    # reference line and grid
    ax.axvline(1.0, linestyle='--', linewidth=1, color='0.6', alpha=0.8)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.8, alpha=0.6)

    # y axis labels
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)

    # labels/title
    ax.set_xlabel(r"EF$_{AUC}$")
    ax.set_title(title)

    # tight x/y limits
    xmin = float(np.nanmin(all_lo)) if len(all_lo) else 0.0
    xmax = float(np.nanmax(all_hi)) if len(all_hi) else 1.0
    span = xmax - xmin if (xmax - xmin) > 0 else 1.0
    ax.set_xlim(max(0, xmin - 0.03*span), xmax + 0.05*span)
    ax.set_ylim(-0.5, n - 0.5)
    ax.margins(y=0.01)

    # major ticks every 1
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # legend
    if legend_position == "inside":
        ax.legend(title="Operating point", loc="lower right", frameon=False, ncol=min(k, 4))
        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(left=0.28, right=0.98, top=0.93, bottom=0.06)
    else:
        ax.legend(title="Operating point", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(right=0.82)

    return fig, ax

def plot_by_target_DEPRECATED(pr_per_target, ef_per_target, models, eps=(1,5)):
    # pr_per_target: colunas ["model","target","PR_AUC"]
    # ef_per_target: colunas ["model","target","EF_AUC_1%","EF_AUC_5%"]
    targets = sorted(ef_per_target["target"].unique())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    # Painel superior: PR-AUC por target (apenas modelos-chave, marcadores diferentes)
    for m in models:
        d = pr_per_target[pr_per_target["model"].eq(m)].set_index("target").reindex(targets)
        ax1.plot(range(len(targets)), d["PR_AUC"].values, marker='o', linestyle='-', label=m)
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("Por receptor: PR-AUC")
    ax1.legend(ncol=min(3,len(models)))

    # Painel inferior: EF_AUC@1% e 5% por target (modelo final vs baseline)
    mfinal = models[0]      # ex.: "OCScore"
    mbaseline = models[1]   # ex.: "mean_score"
    for eps in eps:
        col = f"EF_AUC_{eps}%"
        for m, marker in [(mfinal, 'o'), (mbaseline, '^')]:
            d = ef_per_target[ef_per_target["model"].eq(m)].set_index("target").reindex(targets)
            ax2.plot(range(len(targets)), d[col].values, marker=marker, linestyle='-', label=f"{m} @ {eps}%")
    ax2.set_ylabel("EF_AUC")
    ax2.set_title("Por receptor: EF_AUC @ 1% e 5%")
    ax2.set_xticks(range(len(targets)))
    ax2.set_xticklabels(targets, rotation=30, ha='right')

    plt.tight_layout()
    return fig, (ax1, ax2)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def table_full_by_target(ef_per_target, models):
    cols = ["target","model","EF_AUC_1%","EF_AUC_5%","EF_AUC_10%","EF_AUC_20%","EF_AUC_30%"]
    t = ef_per_target[ef_per_target["model"].isin(models)][cols].copy()
    return t.sort_values(["target","model"])

summary_table = build_summary_table(
    summary_targets=summary_targets,
    summary_pooled=summary_pooled,
    models=score_cols,
    include_pr_auc=False  # mude para True se tiver PR-AUC resumido
)
print(summary_table)

fig1, _ = plot_pooled_efauc(summary_pooled, models=score_cols, eps=(1,5,10,20,30))
fig1.savefig("Fig_Pooled_EF_AUC_1_5.png", dpi=300, bbox_inches="tight")








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pooled_efauc_sorted_compact(
    summary_pooled: pd.DataFrame,
    models=None,
    eps=(1, 5),
    sort_by=5,
    label_map: dict = None,
    row_height=0.38,          # ↓ diminua se ainda quiser mais compacto
    marker_size=4,
    capsize=2,
    legend_position="inside", # "inside" (compacto) ou "outside"
    title="EF_AUC pooled com IC95% (ordenado)"
):
    eps = tuple(int(e) for e in eps)
    df = summary_pooled[summary_pooled["metric"].isin([f"EF_AUC_{e}%" for e in eps])].copy()
    if models is None:
        models = sorted(df["model"].unique())

    # ordenação
    key = f"EF_AUC_{int(sort_by)}%"
    order = (df[df["metric"].eq(key)]
             .set_index("model")[["pooled_value"]]
             .reindex(models)
             .dropna()
             .sort_values("pooled_value")
             .index.tolist())
    if not order:
        raise ValueError(f"Nenhum modelo com métrica {key} encontrada.")

    # dimensões
    n = len(order)
    fig_h = max(3.0, row_height * n)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # rótulos
    if label_map is None: label_map = {}
    ylabels = [label_map.get(m, m) for m in order]

    # plot
    y = np.arange(n)
    dy = min(0.24, 0.6 / max(1, len(eps)))  # separação entre séries
    all_lo, all_hi = [], []

    for i, e in enumerate(eps):
        metric = f"EF_AUC_{e}%"
        dfe = (df[df["metric"].eq(metric)]
               .set_index("model")
               .reindex(order))
        x  = dfe["pooled_value"].to_numpy(float)
        lo = dfe["CI95_lo"].to_numpy(float)
        hi = dfe["CI95_hi"].to_numpy(float)
        xerr = np.vstack((x - lo, hi - x))

        ax.errorbar(x, y + (i - (len(eps)-1)/2)*dy,
                    xerr=xerr, fmt='o', ms=marker_size, capsize=capsize,
                    elinewidth=1, label=f"ε={e}%")
        all_lo.append(np.nanmin(lo)); all_hi.append(np.nanmax(hi))

    # eixo/estilo compactos
    ax.axvline(1.0, linestyle='--', linewidth=1, color='0.6', alpha=0.8)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("EF_AUC")
    ax.set_title(title)

    # limites justos e quase zero de margem vertical
    xmin = float(np.nanmin(all_lo)); xmax = float(np.nanmax(all_hi))
    span = xmax - xmin if (xmax - xmin) > 0 else 1.0
    ax.set_xlim(max(0, xmin - 0.03*span), xmax + 0.05*span)
    ax.set_ylim(-0.5, n - 0.5)   # corta o branco de cima/baixo
    ax.margins(y=0.01)

    # legenda
    if legend_position == "inside":
        ax.legend(title="Ponto de operação", loc="lower right", frameon=False, ncol=len(eps))
        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(left=0.28, right=0.98, top=0.93, bottom=0.06)
    else:
        ax.legend(title="Ponto de operação", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(right=0.82)

    return fig, ax

fig, ax = plot_pooled_efauc_sorted_compact(summary_pooled, eps=(1,5), sort_by=5, legend_position="inside")
fig.savefig("Fig_Pooled_EF_AUC_compact.png", dpi=300, bbox_inches="tight", pad_inches=0.02)


# ====== 4) Figura por receptor: PR-AUC (topo) + EF_AUC@1% e 5% (base) ======
# Filtrar per_target para conter só os 'score_cols' escolhidos
ef_per_target = per_target[per_target["model"].isin(score_cols)].copy()

fig2, _ = plot_by_target(
    pr_per_target=pr_per_target,
    ef_per_target=ef_per_target,
    models=score_cols,
    eps=(1,5)
)
fig2.savefig("Fig_ByTarget_PR_EF_AUC_1_5.png", dpi=300, bbox_inches="tight")

# ====== 5) Tabela completa por receptor (apêndice) ======
table_by_target = table_full_by_target(ef_per_target, models=score_cols)
table_by_target.to_csv("ApX_EF_AUC_full_by_target.csv", index=False)

print("Arquivos gerados:")
print(" - Resumo_EF_AUC_1_5.csv")
print(" - Fig_Pooled_EF_AUC_1_5.png")
print(" - Fig_ByTarget_PR_EF_AUC_1_5.png")
print(" - ApX_EF_AUC_full_by_target.csv")
