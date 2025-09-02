def prop_delta_2xk(contingency: pd.DataFrame) -> pd.DataFrame:
    """
    For a 2xK contingency table, return per-category proportion deltas:
    delta = prop(feature==1) - prop(feature==0).
    """
    if contingency.shape[0] != 2:
        raise ValueError("contingency must be 2xK.")
    props = contingency.div(contingency.sum(axis=1), axis=0)
    # try to identify row order; assume index contains 0 and 1
    if 1 in contingency.index:
        delta = props.loc[1] - props.loc[0]
    elif '1' in contingency.index:
        delta = props.loc['1'] - props.loc['0']
    else:
        # fallback: use last minus first row
        delta = props.iloc[1] - props.iloc[0]
    return delta.to_frame("prop_delta").reset_index(names="MetricCategory")

def plot_prop_delta(contingency: pd.DataFrame, title: str = "Proportion delta (1 - 0)", outpath: str | None = None):
    """
    Diverging bar chart of proportion deltas across metric categories.
    """
    df = prop_delta_2xk(contingency)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="prop_delta", y="MetricCategory", orient="h",
                palette=df["prop_delta"].map(lambda v: "tab:red" if v>0 else "tab:blue"))
    plt.axvline(0, ls="--", c="k", lw=1)
    for _, r in df.iterrows():
        plt.text(r["prop_delta"] + (0.01 if r["prop_delta"]>=0 else -0.01),
                 r["MetricCategory"], f"{r['prop_delta']:.2f}",
                 va="center", ha="left" if r["prop_delta"]>=0 else "right")
    plt.title(title)
    plt.xlabel("Proportion delta (feature=1 minus feature=0)")
    plt.ylabel("")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()

def plot_residuals_lollipop(residuals_df: pd.DataFrame,
                            feature_name: str,
                            presence_level: int | str = 1,
                            title_suffix: str = "Standardized residuals (feature=1)",
                            outpath: str | None = None):
    """
    Lollipop plot of standardized residuals for the 'presence' row only.
    Draw reference lines at ±2 and ±3.
    """
    # pick the row for presence
    row = None
    for key in (presence_level, str(presence_level)):
        if key in residuals_df.index:
            row = residuals_df.loc[key]
            break
    if row is None:
        row = residuals_df.iloc[-1]  # fallback: last row

    s = row.sort_index()
    x = s.values
    cats = s.index.tolist()

    plt.figure(figsize=(7, 4))
    y = np.arange(len(cats))
    plt.hlines(y, 0, x, lw=2)
    plt.plot(x, y, "o")
    for xi, yi, cat in zip(x, y, cats):
        plt.text(xi + (0.1 if xi>=0 else -0.1), yi, f"{xi:.2f}",
                 va="center", ha="left" if xi>=0 else "right")
    for thr, ls in [(2, "--"), (3, ":")]:
        plt.axvline(+thr, ls=ls, c="red", lw=1)
        plt.axvline(-thr, ls=ls, c="red", lw=1)
    plt.yticks(y, cats)
    plt.xlabel("Standardized residual")
    plt.title(f"{feature_name} — {title_suffix}")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()

def plot_chi2_contrib(contingency: pd.DataFrame,
                      feature_name: str,
                      presence_level: int | str = 1,
                      title: str | None = None,
                      outpath: str | None = None):
    """
    Bar plot of per-category chi-square contributions for the presence row only.
    Contribution = (O-E)^2 / E ; normalized to percentage.
    """
    # totals and expected
    total = contingency.values.sum()
    row_sum = contingency.sum(axis=1).values[:, None]
    col_sum = contingency.sum(axis=0).values[None, :]
    expected = (row_sum @ col_sum) / total
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    # select presence row
    row_key = presence_level if presence_level in contingency.index else \
              (str(presence_level) if str(presence_level) in contingency.index else contingency.index[-1])

    obs = contingency.loc[row_key].astype(float)
    exp = expected_df.loc[row_key].astype(float)
    contrib = ((obs - exp) ** 2) / exp
    share = 100 * contrib / contrib.sum()

    df = share.reset_index()
    df.columns = ["MetricCategory", "Chi2SharePct"]

    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x="Chi2SharePct", y="MetricCategory", orient="h", color="steelblue")
    for _, r in df.iterrows():
        plt.text(r["Chi2SharePct"] + 0.5, r["MetricCategory"], f"{r['Chi2SharePct']:.1f}%", va="center")
    plt.xlabel("Share of Chi-square (%)")
    plt.ylabel("")
    plt.title(title or f"{feature_name} — per-category χ² contribution (feature=1)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()

def feature_report_2xk(feature: str,
                       contingency: pd.DataFrame,
                       residuals_df: pd.DataFrame,
                       p_value: float | None = None,
                       outpath: str = "feature_report.png"):
    """
    Compose a 2x2 figure for a single feature:
      [0,0] proportion delta; [0,1] residual lollipop;
      [1,0] chi2 contribution; [1,1] legend/text box.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # proportion delta
    df = prop_delta_2xk(contingency)
    sns.barplot(data=df, x="prop_delta", y="MetricCategory", orient="h",
                palette=df["prop_delta"].map(lambda v: "tab:red" if v>0 else "tab:blue"), ax=axes[0,0])
    axes[0,0].axvline(0, ls="--", c="k", lw=1)
    axes[0,0].set_title("Proportion delta (1 - 0)")
    axes[0,0].set_xlabel("Δ proportion")
    axes[0,0].set_ylabel("")

    # lollipop residuals (feature=1)
    s = None
    for key in (1, '1'):
        if key in residuals_df.index:
            s = residuals_df.loc[key]
            break
    if s is None:
        s = residuals_df.iloc[-1]
    xv = s.values
    yv = np.arange(len(s))
    axes[0,1].hlines(yv, 0, xv, lw=2)
    axes[0,1].plot(xv, yv, "o")
    for xi, yi, cat in zip(xv, yv, s.index.tolist()):
        axes[0,1].text(xi + (0.1 if xi>=0 else -0.1), yi, f"{xi:.2f}",
                       va="center", ha="left" if xi>=0 else "right")
    for thr, ls in [(2, "--"), (3, ":")]:
        axes[0,1].axvline(+thr, ls=ls, c="red", lw=1)
        axes[0,1].axvline(-thr, ls=ls, c="red", lw=1)
    axes[0,1].set_yticks(yv)
    axes[0,1].set_yticklabels(s.index.tolist())
    axes[0,1].set_xlabel("Standardized residual")
    axes[0,1].set_title("Residuals (feature=1)")

    # chi2 contribution
    total = contingency.values.sum()
    row_sum = contingency.sum(axis=1).values[:, None]
    col_sum = contingency.sum(axis=0).values[None, :]
    expected = (row_sum @ col_sum) / total
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    row_key = 1 if 1 in contingency.index else ('1' if '1' in contingency.index else contingency.index[-1])
    obs = contingency.loc[row_key].astype(float)
    exp = expected_df.loc[row_key].astype(float)
    contrib = ((obs - exp) ** 2) / exp
    share = 100 * contrib / contrib.sum()
    sns.barplot(x=share.values, y=share.index, color="steelblue", orient="h", ax=axes[1,0])
    for yi, (cat, val) in enumerate(share.items()):
        axes[1,0].text(val + 0.5, yi, f"{val:.1f}%", va="center")
    axes[1,0].set_title("Per-category χ² contribution (feature=1)")
    axes[1,0].set_xlabel("Share (%)")
    axes[1,0].set_ylabel("")

    # text panel
    axes[1,1].axis('off')
    txt = "How to read:\n" \
          "• Right bars in Δ proportion ⇒ category over-represented when feature=1.\n" \
          "• Residuals > +2 or < -2 are notable.\n" \
          "• χ² share pinpoints which bins drive the association."
    if p_value is not None:
        txt += f"\n\np-value: {p_value:.2e}"
    axes[1,1].text(0, 1, txt, va="top")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def residuals_matrix_from_dict(residuals_dict: dict[str, pd.DataFrame],
                               presence_level: int | str = 1) -> pd.DataFrame:
    """
    Build a matrix (features x categories) with residuals for presence row only.
    """
    rows = {}
    for feat, resdf in residuals_dict.items():
        row = None
        for key in (presence_level, str(presence_level)):
            if key in resdf.index:
                row = resdf.loc[key]
                break
        if row is None:
            row = resdf.iloc[-1]
        rows[feat] = row
    return pd.DataFrame(rows).T  # features x categories

def plot_residuals_matrix(residuals_dict: dict[str, pd.DataFrame],
                          presence_level: int | str = 1,
                          order_by: str = "maxabs",
                          outpath: str = "residuals_matrix.png"):
    """
    Heatmap of features (rows) vs metric categories (columns), values = residuals (feature=1).
    order_by: 'maxabs' (default) or 'chi2' (requires separate chi2 dict).
    """
    mat = residuals_matrix_from_dict(residuals_dict, presence_level=presence_level)
    # order rows by max absolute residual
    idx = mat.abs().max(axis=1).sort_values(ascending=False).index if order_by=="maxabs" else mat.index
    mat = mat.loc[idx]
    plt.figure(figsize=(max(6, mat.shape[1]*1.6), max(6, mat.shape[0]*0.3)))
    ax = sns.heatmap(mat, annot=False, cmap="coolwarm", center=0, cbar_kws={'label': 'Std residual'})
    # optional significance markers (|res|>=2)
    sig = (mat.abs() >= 2)
    ys, xs = np.where(sig.values)
    for y, x in zip(ys, xs):
        ax.text(x+0.5, y+0.5, "•", ha="center", va="center", color="k", fontsize=9)
    plt.title("Residuals (feature=1) across features")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
