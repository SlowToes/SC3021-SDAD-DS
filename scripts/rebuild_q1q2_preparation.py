from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """
# SC3021 Q1/Q2 - Data Preparation Notebook (Posts + Reels)

This notebook focuses **only** on data preparation.  
It does **not** perform hypothesis testing or modeling yet.

## Goal
Prepare high-quality datasets from:
- `instagram/Instagram - Posts.csv`
- `instagram/Instagram - Reels.csv`

for downstream analysis of post characteristics and virality.
"""
    )
)

cells.append(
    md(
        """
## Preparation Strategy

We follow a rigorous data science preparation pipeline:

1. **Load raw data** from source files.
2. **Profile** data quality (schema, missingness, duplicates, invalid values).
3. **Structure** fields into consistent formats (e.g., datetime, hashtag list).
4. **Clean** invalid values without introducing bias (avoid replacing unknowns with zeros).
5. **Enrich** features useful for later analysis (`has_hashtags`, `hashtag_count`).
6. **Validate** post-clean quality with explicit checks.
7. **Export** prepared datasets for reproducible downstream work.

## Important Design Decision: Keep Posts and Reels Separate

Posts and Reels have different metric coverage (`views` and `followers` are Reel-centric).  
To avoid invalid assumptions and hidden bias, we prepare them separately and only create an optional
"common-schema" combined table for compatible columns.
"""
    )
)

cells.append(
    code(
        """
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)
"""
    )
)

cells.append(
    code(
        """
# Paths
ROOT = Path(".")
POSTS_PATH = ROOT / "instagram" / "Instagram - Posts.csv"
REELS_PATH = ROOT / "instagram" / "Instagram - Reels.csv"
OUTPUT_DIR = ROOT / "instagram" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Posts path exists:", POSTS_PATH.exists(), POSTS_PATH)
print("Reels path exists:", REELS_PATH.exists(), REELS_PATH)
print("Output directory:", OUTPUT_DIR.resolve())
"""
    )
)

cells.append(
    md(
        """
## 1) Load Raw Data

We first load both datasets exactly as provided, preserving raw values for profiling.
"""
    )
)

cells.append(
    code(
        """
df_posts_raw = pd.read_csv(POSTS_PATH)
df_reels_raw = pd.read_csv(REELS_PATH)

print("Posts shape:", df_posts_raw.shape)
print("Reels shape:", df_reels_raw.shape)
"""
    )
)

cells.append(
    md(
        """
## 2) Data Profiling Utilities

The following helper functions provide a reproducible profile:
- schema + dtypes
- missingness (%)
- duplicate indicators
- numeric quality checks (negative/zero counts)
"""
    )
)

cells.append(
    code(
        """
def profile_dataframe(df: pd.DataFrame, name: str, id_col: str = "post_id") -> dict:
    \"\"\"Return a structured quality profile for a dataframe.\"\"\"
    profile = {}
    profile["name"] = name
    profile["shape"] = df.shape
    profile["columns"] = list(df.columns)
    profile["dtypes"] = df.dtypes.astype(str).to_dict()

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    profile["missing"] = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct}
    ).sort_values("missing_pct", ascending=False)

    if id_col in df.columns:
        profile["duplicate_id_count"] = int(df[id_col].duplicated().sum())
    else:
        profile["duplicate_id_count"] = None

    if "url" in df.columns:
        profile["duplicate_url_count"] = int(df["url"].duplicated().sum())
    else:
        profile["duplicate_url_count"] = None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    neg_counts = {}
    zero_counts = {}
    for col in numeric_cols:
        series = df[col]
        neg_counts[col] = int((series < 0).sum())
        zero_counts[col] = int((series == 0).sum())

    profile["negative_counts"] = neg_counts
    profile["zero_counts"] = zero_counts
    return profile


def display_profile(profile: dict, top_missing: int = 15):
    \"\"\"Pretty-print key profile results.\"\"\"
    print("=" * 80)
    print(f"Dataset: {profile['name']}")
    print(f"Shape: {profile['shape']}")
    print(f"Duplicate post_id: {profile['duplicate_id_count']}")
    print(f"Duplicate url: {profile['duplicate_url_count']}")
    print("-" * 80)
    print("Top missing columns:")
    print(profile["missing"].head(top_missing))
    print("-" * 80)
    print("Negative counts (numeric columns):")
    print(profile["negative_counts"])
    print("-" * 80)
    print("Zero counts (numeric columns):")
    print(profile["zero_counts"])
"""
    )
)

cells.append(
    code(
        """
posts_profile_raw = profile_dataframe(df_posts_raw, "Posts (Raw)")
reels_profile_raw = profile_dataframe(df_reels_raw, "Reels (Raw)")

display_profile(posts_profile_raw)
display_profile(reels_profile_raw)
"""
    )
)

cells.append(
    md(
        """
## 3) Structuring + Cleaning Utilities

Key principles:
- Convert numeric fields with `errors='coerce'`.
- Treat impossible negatives as invalid (`NaN`), not zero.
- Keep true missingness as missing (to avoid biasing medians/rates).
- Parse hashtag strings robustly into Python lists.
"""
    )
)

cells.append(
    code(
        """
def parse_hashtags(value):
    \"\"\"Convert hashtag field into a clean list of hashtag strings.\"\"\"
    if pd.isna(value):
        return []

    if isinstance(value, list):
        raw_list = value
    elif isinstance(value, str):
        text = value.strip()
        if text == "" or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                raw_list = parsed
            else:
                # Fallback: split a plain comma string if parsing is not list-like
                raw_list = [x.strip() for x in text.split(",") if x.strip()]
        except (ValueError, SyntaxError):
            raw_list = [x.strip() for x in text.split(",") if x.strip()]
    else:
        return []

    cleaned = []
    for tag in raw_list:
        if tag is None:
            continue
        tag = str(tag).strip()
        if tag == "":
            continue
        cleaned.append(tag)
    return cleaned


def clean_nonnegative_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    \"\"\"Coerce selected columns to numeric and set negative values to NaN.\"\"\"
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[out[col] < 0, col] = np.nan
    return out
"""
    )
)

cells.append(
    md(
        """
## 4) Prepare Posts Dataset (Separate Flow)
"""
    )
)

cells.append(
    code(
        """
posts = df_posts_raw.copy()
posts["source_type"] = "post"

# Datetime structuring
if "date_posted" in posts.columns:
    posts["date_posted"] = pd.to_datetime(posts["date_posted"], errors="coerce", utc=True)

# Deduplicate conservatively
if "post_id" in posts.columns:
    posts = posts.drop_duplicates(subset=["post_id"], keep="first")
elif "url" in posts.columns:
    posts = posts.drop_duplicates(subset=["url"], keep="first")

# Parse hashtags and engineer hashtag features
posts["hashtags_list"] = posts["hashtags"].apply(parse_hashtags) if "hashtags" in posts.columns else [[] for _ in range(len(posts))]
posts["has_hashtags"] = posts["hashtags_list"].apply(lambda x: len(x) > 0)
posts["hashtag_count"] = posts["hashtags_list"].apply(len)

# Clean core metrics (non-negative constraints)
posts = clean_nonnegative_numeric(posts, ["likes", "num_comments", "followers", "video_view_count", "video_play_count", "posts_count"])

# Track missingness flags for core metrics
for metric in ["likes", "num_comments"]:
    if metric in posts.columns:
        posts[f"{metric}_is_missing"] = posts[metric].isna()

print("Prepared posts shape:", posts.shape)
posts[["source_type", "post_id", "date_posted", "likes", "num_comments", "has_hashtags", "hashtag_count"]].head()
"""
    )
)

cells.append(
    md(
        """
## 5) Prepare Reels Dataset (Separate Flow)
"""
    )
)

cells.append(
    code(
        """
reels = df_reels_raw.copy()
reels["source_type"] = "reel"

# Datetime structuring
if "date_posted" in reels.columns:
    reels["date_posted"] = pd.to_datetime(reels["date_posted"], errors="coerce", utc=True)

# Deduplicate conservatively
if "post_id" in reels.columns:
    reels = reels.drop_duplicates(subset=["post_id"], keep="first")
elif "url" in reels.columns:
    reels = reels.drop_duplicates(subset=["url"], keep="first")

# Parse hashtags and engineer hashtag features
reels["hashtags_list"] = reels["hashtags"].apply(parse_hashtags) if "hashtags" in reels.columns else [[] for _ in range(len(reels))]
reels["has_hashtags"] = reels["hashtags_list"].apply(lambda x: len(x) > 0)
reels["hashtag_count"] = reels["hashtags_list"].apply(len)

# Clean core metrics (non-negative constraints)
reels = clean_nonnegative_numeric(reels, ["likes", "num_comments", "views", "followers", "video_play_count", "following", "posts_count", "length"])

# Track missingness flags for core metrics
for metric in ["likes", "num_comments", "views", "followers"]:
    if metric in reels.columns:
        reels[f"{metric}_is_missing"] = reels[metric].isna()

print("Prepared reels shape:", reels.shape)
reels[["source_type", "post_id", "date_posted", "likes", "num_comments", "views", "followers", "has_hashtags", "hashtag_count"]].head()
"""
    )
)

cells.append(
    md(
        """
## 6) Post-Clean Validation

We rerun quality checks after preparation to verify:
- duplicate handling worked
- impossible negative values were removed
- core engineered fields exist and look reasonable
"""
    )
)

cells.append(
    code(
        """
posts_profile_clean = profile_dataframe(posts, "Posts (Prepared)")
reels_profile_clean = profile_dataframe(reels, "Reels (Prepared)")

display_profile(posts_profile_clean)
display_profile(reels_profile_clean)
"""
    )
)

cells.append(
    code(
        """
print("Posts hashtag feature sanity check:")
print(posts["has_hashtags"].value_counts(dropna=False))
print(posts["hashtag_count"].describe())

print("\\nReels hashtag feature sanity check:")
print(reels["has_hashtags"].value_counts(dropna=False))
print(reels["hashtag_count"].describe())
"""
    )
)

cells.append(
    md(
        """
## 7) Optional Common-Schema Table (For Compatible Analyses Only)

A combined table can be useful for analyses based on shared columns (e.g., hashtags vs likes/comments),
but **should not** be used for metrics requiring Reel-only fields (`views`, `followers`) without careful handling.
"""
    )
)

cells.append(
    code(
        """
common_columns = [
    "source_type",
    "post_id",
    "url",
    "user_posted",
    "date_posted",
    "hashtags_list",
    "has_hashtags",
    "hashtag_count",
    "likes",
    "num_comments",
    "views",
    "followers",
    "is_verified",
    "is_paid_partnership",
]

def select_existing_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    out = df[existing].copy()
    for m in missing:
        out[m] = np.nan
    return out[cols]

posts_common = select_existing_columns(posts, common_columns)
reels_common = select_existing_columns(reels, common_columns)
df_common = pd.concat([posts_common, reels_common], ignore_index=True)

print("Common-schema combined shape:", df_common.shape)
df_common.head()
"""
    )
)

cells.append(
    md(
        """
## 8) Export Prepared Outputs

We export:
- `posts_prepared.csv`
- `reels_prepared.csv`
- `combined_common_schema_prepared.csv` (optional shared schema)
"""
    )
)

cells.append(
    code(
        """
posts_out = OUTPUT_DIR / "posts_prepared.csv"
reels_out = OUTPUT_DIR / "reels_prepared.csv"
common_out = OUTPUT_DIR / "combined_common_schema_prepared.csv"

posts.to_csv(posts_out, index=False)
reels.to_csv(reels_out, index=False)
df_common.to_csv(common_out, index=False)

print("Saved:", posts_out)
print("Saved:", reels_out)
print("Saved:", common_out)
"""
    )
)

cells.append(
    md(
        """
## 9) Data Preparation Summary (No Analysis Yet)

Completed:
- rigorous profiling for raw and cleaned data
- robust structuring/parsing of hashtags
- non-negative validation for engagement metrics
- explicit separation of Posts and Reels preparation pipelines
- reproducible exports for next-stage analysis/modeling

Not included in this notebook:
- statistical testing
- EDA conclusions
- predictive modeling

Those should be done in a separate analysis notebook using these prepared outputs.
"""
    )
)

nb["cells"] = cells

output_path = Path("q1q2.ipynb")
with output_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook rebuilt at: {output_path.resolve()}")
