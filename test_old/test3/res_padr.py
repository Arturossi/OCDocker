from typing import Iterable, Optional

# Requirements: import numpy as np, import matplotlib.pyplot as plt, import pandas as pd, import matplotlib

def _strength_from_nbs_norm(nbs_norm: float, thresholds=(0.10, 0.20, 0.35)) -> str:
    """Bucketize |NBS_norm| in [0,1] into none/weak/moderate/strong/very strong."""
    a = abs(nbs_norm)
    if a == 0:
        return 'none'
    if a < thresholds[0]:
        return 'weak'
    if a < thresholds[1]:
        return 'moderate'
    if a < thresholds[2]:
        return 'strong'
    return 'very strong'


def plot_impact_arrows_inline_labels(
    impact_df: pd.DataFrame,
    title: str,
    outpath: str | None = None,
    tau: float = 0.05,                 # tau na ESCALA ORIGINAL do NBS ([-2,2])
    thresholds=(0.10, 0.20, 0.35),     # para força a partir do |NBS_norm|
    xpad: float = 0.025,               # distância horizontal do texto ao marcador
    height_per_feature: float = 0.42,  # altura por feature
    max_height: float = 28.0,          # altura máxima do gráfico
    font_size: int = 10
):
    """
    Arrow plot com TODOS os nomes ao lado do marcador.
    - Normaliza NBS -> NBS_norm em [-1,1] APENAS para visualização.
    - Cores: direção (positivo/negativo/neutro) derivada de NBS e tau.
    - Símbolos: força (derivada de |NBS_norm|).

    Espera colunas em impact_df: ['Feature','NBS'].
    Se 'Direction' não existir, é inferida de NBS e tau.
    """

    df = impact_df.copy()

    # Normalização para o eixo visual
    df['NBS_norm'] = (df['NBS'] / 2.0).clip(-1.0, 1.0)
    tau_norm = tau / 2.0

    # Direção (se não estiver na tabela)
    if 'Direction' not in df.columns:
        df['Direction'] = np.where(df['NBS'] > +tau, 'positive',
                            np.where(df['NBS'] < -tau, 'negative', 'neutral'))

    # Força a partir do |NBS_norm|
    df['Strength'] = df['NBS_norm'].apply(lambda v: _strength_from_nbs_norm(v, thresholds))

    # Mapeamentos visuais
    color_map  = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#7f7f7f'}
    marker_map = {'none': 'o', 'weak': 'o', 'moderate': 's', 'strong': 'D', 'very strong': 'P'}
    size_map   = {'none': 40,  'weak': 60,  'moderate': 90, 'strong': 120, 'very strong': 150}
    alpha_map  = {'none': 0.35,'weak': 0.5,'moderate': 0.7,'strong': 0.9,'very strong': 0.95}

    df['Color']  = df['Direction'].map(color_map).fillna('#7f7f7f')
    df['Marker'] = df['Strength'].map(marker_map).fillna('o')
    df['Size']   = df['Strength'].map(size_map).fillna(60)
    df['Alpha']  = df['Strength'].map(alpha_map).fillna(0.6)

    # Ordena por NBS_norm para ficar natural (negativos à esquerda)
    df = df.sort_values('NBS_norm').reset_index(drop=True)
    y = np.arange(len(df))

    # Altura dinâmica para caber todos os nomes
    fig_h = min(max(6.0, height_per_feature * len(df)), max_height)
    plt.figure(figsize=(12, fig_h))

    # Hastes + marcadores
    for i, r in df.iterrows():
        plt.plot([0, r['NBS_norm']], [i, i], color=r['Color'], linewidth=2, alpha=r['Alpha'])
        plt.scatter(r['NBS_norm'], i, s=r['Size'], c=r['Color'],
                    marker=r['Marker'], edgecolor='k', linewidth=0.4, zorder=3)

    # Rótulos AO LADO de cada ponto (todos)
    for i, r in df.iterrows():
        # desloca para direita/esquerda conforme o sinal
        x = r['NBS_norm'] + (xpad if r['NBS_norm'] >= 0 else -xpad)
        ha = 'left' if r['NBS_norm'] >= 0 else 'right'
        plt.text(x, i, str(r['Feature']), va='center', ha=ha, fontsize=font_size)

    # Eixo Y sem ticks (nomes já estão ao lado)
    plt.yticks([], [])

    # Guias
    plt.axvline(0,          color='k', linestyle='--', linewidth=1)
    plt.axvline(+tau_norm,  color='k', linestyle=':',  linewidth=1)
    plt.axvline(-tau_norm,  color='k', linestyle=':',  linewidth=1)

    plt.xlabel("Net Benefit Score (normalized: −1 worse ← 0 → better +1)")
    plt.xlim(-1.05, 1.05)
    plt.title(title)

    # Legendas compactas
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    leg_dir = [mpatches.Patch(color=color, label=lbl) for lbl, color in
               [('positivo', color_map['positive']), ('negativo', color_map['negative']), ('neutro', color_map['neutral'])]]
    leg_str = [
        mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=8, label='none/weak'),
        mlines.Line2D([], [], color='k', marker='s', linestyle='None', markersize=8, label='moderate'),
        mlines.Line2D([], [], color='k', marker='D', linestyle='None', markersize=8, label='strong'),
        mlines.Line2D([], [], color='k', marker='P', linestyle='None', markersize=8, label='very strong'),
    ]
    lg1 = plt.legend(handles=leg_dir, title='direção', loc='upper left', frameon=False, fontsize=9)
    plt.gca().add_artist(lg1)
    plt.legend(handles=leg_str, title='força (símbolos)', loc='upper right', frameon=False, fontsize=9)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()


def get_neutral_features(impact_df: pd.DataFrame, tau: float = 0.05) -> list[str]:
    """Lista de features neutras (|NBS| < tau na escala original, ou Direction == 'neutral')."""
    if 'Direction' in impact_df.columns:
        return (impact_df.loc[impact_df['Direction'] == 'neutral', 'Feature']
                .astype(str).sort_values().tolist())
    return (impact_df.loc[impact_df['NBS'].abs() < tau, 'Feature']
            .astype(str).sort_values().tolist())


def _beneficial_categories(metric: str, categories: Iterable[str], custom: Optional[Iterable[str]] = None) -> set[str]:
    """
    Decide which categories are 'beneficial' for the given metric.
    Default rules:
      - AUC: categories containing 'high' (high, very high)
      - RMSE: categories containing 'low' (low, very low)
    Fallback: top half of ordered categories.
    """
    cats = [str(c) for c in categories]
    if custom is not None:
        return set(map(str, custom))

    m = metric.strip().upper()
    if m == 'AUC':
        good = {c for c in cats if 'high' in c.lower()}
    elif m == 'RMSE':
        good = {c for c in cats if 'low' in c.lower()}
    else:
        # fallback: top half by order
        k = len(cats)
        good = set(cats[k//2:])
    # safety
    return good if good else set(cats[-max(1, len(cats)//2):])

def _proportion_delta(contingency: pd.DataFrame, presence_level: int | str = 1) -> pd.Series:
    """
    Δp(c) = p(c | feature=1) - p(c | feature=0) for each category c.
    """
    cont = contingency.copy()
    # normalize rows
    props = cont.div(cont.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # choose keys
    if presence_level in cont.index:
        k1 = presence_level
    elif str(presence_level) in cont.index:
        k1 = str(presence_level)
    else:
        k1 = cont.index[-1]

    if 0 in cont.index:
        k0 = 0
    elif '0' in cont.index:
        k0 = '0'
    else:
        k0 = cont.index[0]

    return (props.loc[k1] - props.loc[k0]).astype(float)

def _net_benefit(delta: pd.Series, beneficial: set[str]) -> float:
    """
    Net Benefit Score in [-1, 1]:
      NBS = sum_{c in good} Δp(c) - sum_{c in bad} Δp(c)
    """
    idx = delta.index.astype(str)
    good = [c for c in idx if c in beneficial]
    bad  = [c for c in idx if c not in beneficial]
    return float(delta[good].sum() - delta[bad].sum())

def _strength_from_v(v: float) -> str:
    if pd.isna(v):
        return 'unknown'
    if v < 0.10:
        return 'none'
    if v < 0.20:
        return 'weak'
    if v < 0.30:
        return 'moderate'
    if v < 0.50:
        return 'strong'
    return 'very strong'

def build_impact_overview(chi_df: pd.DataFrame,
                          contingency_dict: dict[str, pd.DataFrame],
                          metric: str,
                          presence_level: int | str = 1,
                          beneficial_custom: Optional[Iterable[str]] = None,
                          tau: float = 0.05) -> pd.DataFrame:
    """
    Produz uma tabela clara: direção (positive/negative/neutral) e força.
    Colunas-chave:
      Feature, NBS, Direction, Strength, Chi2, p-value, CramersV,
      FavoredCategory, HurtCategory
    """
    # pega o conjunto de categorias a partir de qualquer tabela (assume todas iguais)
    any_cont = next(iter(contingency_dict.values()))
    categories = any_cont.columns.astype(str).tolist()
    beneficial = _beneficial_categories(metric, categories, custom=beneficial_custom)

    rows = []
    for _, r in chi_df.iterrows():
        feat = r['Feature']
        cont = contingency_dict.get(feat)
        if cont is None or cont.empty:
            rows.append({'Feature': feat, 'NBS': np.nan, 'Direction': 'neutral',
                         'Strength': 'unknown', 'Chi2': r['Chi2 Statistic'],
                         'p-value': r['p-value'], 'CramersV': r["Cramér's V"],
                         'FavoredCategory': None, 'HurtCategory': None})
            continue

        delta = _proportion_delta(cont, presence_level=presence_level)
        nbs = _net_benefit(delta, beneficial)

        # direção com tolerância tau
        if abs(nbs) < tau:
            direction = 'neutral'
        else:
            direction = 'positive' if nbs > 0 else 'negative'

        # categorias que mais ajudam/atrapalham (pelo delta)
        favored = delta.idxmax()
        hurt    = delta.idxmin()

        strength = _strength_from_v(r["Cramér's V"])

        rows.append({
            'Feature': feat,
            'NBS': nbs,
            'Direction': direction,
            'Strength': strength,
            'Chi2': r['Chi2 Statistic'],
            'p-value': r['p-value'],
            'CramersV': r["Cramér's V"],
            'FavoredCategory': favored,
            'HurtCategory': hurt
        })

    out = pd.DataFrame(rows)
    # utilidade: magnitude padronizada para ordenar
    out['|NBS|'] = out['NBS'].abs()
    out['NegLog10P'] = -np.log10(np.clip(out['p-value'].astype(float), 1e-300, 1.0))
    return out.sort_values(['Direction','|NBS|','NegLog10P'], ascending=[True, False, False])

# AUC
auc_impact = build_impact_overview(chi_square_results_auc, cont_auc, metric='AUC',
                                   presence_level=1, beneficial_custom=['very high','high'], tau=0.05)

plot_impact_arrows_inline_labels(
    auc_impact,
    title="Impacto por feature — AUC",
    outpath="AUC_impact_inline.png",
    tau=0.05,                 # mesmo tau do overview (escala original)
    xpad=0.025,               # ajuste se quiser o texto mais perto/longe do ponto
    height_per_feature=0.45,  # aumente se os nomes ficarem apertados
    max_height=30.0,
    font_size=10
)

neutros_auc = get_neutral_features(auc_impact, tau=0.05)
print("Neutros AUC:", neutros_auc)

# RMSE
rmse_impact = build_impact_overview(chi_square_results_rmse, cont_rmse, metric='RMSE',
                                    presence_level=1, beneficial_custom=['very low','low'], tau=0.05)

plot_impact_arrows_inline_labels(
    rmse_impact,
    title="Impacto por feature — RMSE",
    outpath="RMSE_impact_inline.png",
    tau=0.05
)

neutros_rmse = get_neutral_features(rmse_impact, tau=0.05)
print("Neutros RMSE:", neutros_rmse)
