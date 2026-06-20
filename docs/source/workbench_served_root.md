# Workbench served root layout

The OCScore Workbench dashboard is started with a **served root** — the OCScore output directory you point the server at (for example `.../OCScore/output`).

## Raw modeling inputs (`raw_prepare/`)

The **Design** tab discovers descriptor columns from CSV headers under:

```
{served_root}/raw_prepare/raw_pdbbind.csv
{served_root}/raw_prepare/raw_dudez.csv
```

When both files exist, the UI auto-loads features on tab open. You do not need to paste paths manually unless your layout differs.

Discovery is **header-only** (`columns_only: true` in the API response): the Workbench reads column names from the first row and never loads the full tables. This keeps large (~400+ column) inputs fast.

**Case sensitivity:** the feature browser filter is **case-insensitive** (display only). Wildcards and policy patterns (`include_patterns`, `exclude_patterns`) are **case-sensitive**, matching Python `fnmatch.fnmatchcase` and CSV column names exactly (e.g. `ligand_*`, not `Ligand_*`).

Metadata columns (`database`, `target`, receptor/ligand identifiers, etc.) and target columns are stripped automatically before building the candidate feature list shown in the browser.

### API: `POST /api/ablation-design/features`

Request body (all optional when defaults exist):

- `raw_input_dir` — directory containing the two CSVs
- `pdbbind_input` / `dudez_input` — explicit file paths
- `merged_input` — single merged CSV alternative
- `feature_source` — `auto`, `pdbbind`, `dudez`, or `union`

The response includes `candidate_features`, grouped columns, `metadata_columns`, `target_columns`, and `columns_only: true`.

## Custom ablation policies

Designed policies can be written into the workspace (not the bundled package) via **Write to workspace** in the Design tab, which calls `POST /api/ablation-design/write`. Shipped policies under `OCScore/Protocols/Ablations/` remain read-only; workspace copies live under paths such as `{layout_root}/Ablations/my_policy.yml`.

## Related endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/ablation-design` | Templates, discovered inputs, existing ablation names |
| `POST /api/ablation-design/preview` | Apply draft rules to candidate features |
| `POST /api/ablation-design/plan` | Training command + preflight |
| `POST /api/ablation-design/write` | Save policy YAML into the served workspace |
