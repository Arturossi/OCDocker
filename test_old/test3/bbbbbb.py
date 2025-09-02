from typing import Optional, Iterable

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

def plot_impact_arrows(impact_df: pd.DataFrame,
                       title: str,
                       outpath: Optional[str] = None,
                       tau: float = 0.05,
                       topn: Optional[int] = None):
    """
    Panorama único: uma seta/marker por feature no eixo [-1,1] (NBS).
    Cor indica direção, forma/grossura indica força. Linha tracejada em 0 e ±tau.
    """
    df = impact_df.copy()
    if topn is not None:
        df = df.nlargest(topn, columns='|NBS|')

    # mapeia cor por direção
    color_map = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#7f7f7f'}
    df['Color'] = df['Direction'].map(color_map).fillna('#7f7f7f')

    # marcador por força
    marker_map = {'none': 'o', 'weak': 'o', 'moderate': 's', 'strong': 'D', 'very strong': 'P', 'unknown': 'o'}
    size_map =   {'none': 40,  'weak': 60,  'moderate': 80, 'strong': 110,'very strong': 140, 'unknown': 60}
    df['Marker'] = df['Strength'].map(marker_map).fillna('o')
    df['Size']   = df['Strength'].map(size_map).fillna(60)

    df = df.sort_values('NBS')
    plt.figure(figsize=(10, max(6, 0.35*len(df))))
    y = np.arange(len(df))
    # “haste” do zero até o ponto para reforçar direção
    for i, (_, row) in enumerate(df.iterrows()):
        plt.plot([0, row['NBS']], [i, i], linestyle='-', linewidth=2, color=row['Color'], alpha=0.6)
    # pontos
    for i, (_, row) in enumerate(df.iterrows()):
        plt.scatter(row['NBS'], i, s=row['Size'], c=row['Color'], marker=row['Marker'], edgecolor='k', linewidth=0.4)

    # anotações à direita: força e cat favorecida
    for i, (_, row) in enumerate(df.iterrows()):
        txt = f"{row['Strength']} • fav: {row['FavoredCategory']} • p={row['p-value']:.1e}"
        plt.text(1.02, i, txt, va='center', ha='left', transform=plt.gca().get_yaxis_transform())

    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(+tau, color='k', linestyle=':', linewidth=1)
    plt.axvline(-tau, color='k', linestyle=':', linewidth=1)
    plt.yticks(y, df['Feature'])
    plt.xlim(-1.05, 1.05)
    plt.xlabel("Net Benefit Score (−1 pior ← 0 → melhor +1)")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()

# AUC
auc_impact = build_impact_overview(chi_square_results_auc, cont_auc, metric='AUC', presence_level=1, tau=0.05)
auc_impact.to_csv("AUC_impact_overview.csv", index=False)
plot_impact_arrows(auc_impact, title="Impacto por feature — AUC", outpath="AUC_impact_arrows.png", tau=0.05, topn=None)

# RMSE
rmse_impact = build_impact_overview(chi_square_results_rmse, cont_rmse, metric='RMSE', presence_level=1, tau=0.05)
rmse_impact.to_csv("RMSE_impact_overview.csv", index=False)
plot_impact_arrows(rmse_impact, title="Impacto por feature — RMSE", outpath="RMSE_impact_arrows.png", tau=0.05, topn=None)

