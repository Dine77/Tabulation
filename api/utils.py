from unittest import result
from .models import ActivityLog
from datetime import datetime
import pandas as pd
import pyreadstat
import math
import re
import numpy as np
import os
import pandas as pd
from django.conf import settings
from .models import Project
import math
from scipy.stats import norm
import numpy as np

def log_activity(user, action, project_name="", file_name="", details=""):
    ActivityLog(
        user=user,
        action=action,
        project_name=project_name,
        file_name=file_name,
        details=details,
        timestamp=datetime.utcnow()
    ).save()

Z_CRITICAL = {
    80: 1.28,
    85: 1.44,
    90: 1.64,
    95: 1.96,
    99: 2.58,
}

def ztest_proportions(count1, base1, count2, base2, alpha=0.1):
    print("Col bases:", base1, base2)
    if base1 < 30 or base2 < 30:
        return False

    p1, p2 = count1 / base1, count2 / base2
    p = (count1 + count2) / (base1 + base2)  # pooled

    se = math.sqrt(p * (1 - p) * (1/base1 + 1/base2))
    if se == 0:
        return False

    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    # Debug log
    print(f"Z={z:.2f}, p={p_value:.3f}, alpha={alpha}")
    return p_value < alpha
# ---------- Significance Test ----------
def run_sig_test(p_count, p_base, c_count, c_base, z_threshold=1.96):
    """Return sig flag 'UP' or 'DOWN' based on Decipher Z-test."""
    if p_base <= 0 or c_base <= 0:
        return None
    
    p_score = p_count / p_base
    c_score = c_count / c_base
    
    if p_score == 0 and c_score == 0:
        return None
    
    pq = (p_count + c_count) / (p_base + c_base)
    q = 1 - pq
    
    se = np.sqrt(pq * q * ((1/p_base) + (1/c_base)))
    if se == 0:
        return None
    
    z = (c_score - p_score) / se
    
    if z >= z_threshold:
        return "UP"    # significantly higher
    elif z <= -z_threshold:
        return "DOWN"  # significantly lower
    return None

# ---------- Apply Sig Test to Table ----------
def apply_sig_tests_to_table(cols, rows, alpha=0.95):
    """
    Apply sig test to all pairs of columns except Total (col A).
    Each cell in rows will get a 'sig' key with UP/DOWN/None.
    """
    z_threshold = norm.ppf(alpha)  # e.g. 1.64 for 90%, 1.96 for 95%

    for i in range(1, len(cols)):   # 🔹 start from 1 → skip Total
        base_i = cols[i][1].shape[0]

        for j in range(i+1, len(cols)):
            base_j = cols[j][1].shape[0]

            for r in rows:
                ci = r["cells"][i]
                cj = r["cells"][j]

                sig = run_sig_test(ci["count"], base_i, cj["count"], base_j, z_threshold)
                if sig == "UP":
                    ci["sig"] = chr(65 + j)   # mark against col letter
                elif sig == "DOWN":
                    cj["sig"] = chr(65 + i)


# --- Common Helper for Merging "Others" ---
def merge_others(rows, base, is_cells=False):
    """
    Merge items (rows or cells) with labels like 'Other', 'Others', 'Any other', 'Specify', 'Misc'.
    rows: list of dicts [{label, count, pct, cells?}]
    base: base count (for recalculating %)
    is_cells: if True → merging across row.cells (SRG case)
    """
    pattern = re.compile(r"(other|specify|misc)", re.IGNORECASE)
    merged = None
    cleaned = []

    for row in rows:
        label = row.get("label", "")
        if pattern.search(label):
            if not merged:
                merged = {
                    "label": "Others",
                    "count": 0,
                    "pct": "0%",
                }
                if is_cells:
                    merged["cells"] = [{"count": 0, "pct": "0%"} for _ in row.get("cells", [])]

            if is_cells:
                for idx, cell in enumerate(row.get("cells", [])):
                    merged["cells"][idx]["count"] += cell.get("count", 0)
            else:
                merged["count"] += row.get("count", 0)
        else:
            cleaned.append(row)

    # recompute pct
    if merged:
        if is_cells:
            for cell in merged["cells"]:
                pct = (cell["count"] / base * 100) if base > 0 else 0
                cell["pct"] = f"{pct:.1f}%"
        else:
            pct = (merged["count"] / base * 100) if base > 0 else 0
            merged["pct"] = f"{pct:.1f}%"
        cleaned.append(merged)

    return cleaned


def crosstab_nps(df, var_name):
    results = {}

    total_base = df[var_name].notna().sum()
    freq = df[var_name].value_counts(dropna=True)
    freq_pct = df[var_name].value_counts(normalize=True, dropna=True) * 100

    # Promoters (9-10)
    promoters = sum(int(freq.get(c, 0)) for c in [9, 10])
    promoters_pct = round(promoters / total_base * 100, 1) if total_base > 0 else 0

    # Passives (7-8)
    passives = sum(int(freq.get(c, 0)) for c in [7, 8])
    passives_pct = round(passives / total_base * 100, 1) if total_base > 0 else 0

    # Detractors (0-6)
    detractors = sum(int(freq.get(c, 0)) for c in range(0, 7))
    detractors_pct = round(detractors / total_base * 100, 1) if total_base > 0 else 0

    # NPS Score
    nps_score = round(promoters_pct - detractors_pct, 1)

    # Extra stats
    data_series = df[var_name].dropna()
    mean_val = round(float(np.mean(data_series)), 2) if not data_series.empty else None
    median_val = round(float(np.median(data_series)), 2) if not data_series.empty else None
    std_val = round(float(np.std(data_series, ddof=0)), 2) if not data_series.empty else None

    results["NPS"] = {
        "score": nps_score,
        "base": int(total_base),
        "promoter": {"count": int(promoters), "pct": f"{promoters_pct}%"},
        "passive": {"count": int(passives), "pct": f"{passives_pct}%"},
        "detractor": {"count": int(detractors), "pct": f"{detractors_pct}%"},
        "mean": mean_val,
        "median": median_val,
        "stddev": std_val,
    }

    return results


def crosstab_single_choice(df, var_name, value_labels, topbreaks=None):
    results = {}

    total_base = df[var_name].notna().sum()
    freq = df[var_name].value_counts(dropna=True)
    freq_pct = df[var_name].value_counts(normalize=True, dropna=True) * 100

    merged = {}

    for code, label in value_labels.items():
        if code in freq:
            code_str = str(int(code)) if isinstance(code, (int, float)) and code == int(code) else str(code)
            group_key = label.strip().lower()
            if group_key not in merged:
                merged[group_key] = {
                    "label": f"[{code_str}] {label}",
                    "count": 0,
                    "pct": 0.0,
                }
            merged[group_key]["count"] += int(freq[code])
            merged[group_key]["pct"] = f"{round(float(freq_pct[code]), 1)}%"


    rows = list(merged.values())
    rows = merge_others(rows, total_base)   # ✅ merge here

    results["Total"] = {"base": int(total_base), "rows": rows}
    return results

def get_topbreak_letters(n: int):
    """Return A, B, C … AA, AB if needed"""
    letters = []
    for i in range(n):
        q, r = divmod(i, 26)
        if q == 0:
            letters.append(chr(65 + r))
        else:
            letters.append(chr(65 + q - 1) + chr(65 + r))
    return letters


def build_topbreak_columns(df, topbreak: str, meta):
    """
    Returns:
      cols = [("Total", df), ("Finance", df1), ("Tech", df2)...]
      columns = [{"label": "Total", "letter": ""}, {"label": "Finance", "letter": "A"}, ...]
    """
    cols = [("Total", df)]

    if topbreak in df.columns:
        for tb_value, tb_df in df.groupby(topbreak):
            # Map code → label if available
            label = str(tb_value)
            if topbreak in meta.variable_value_labels:
                label = meta.variable_value_labels[topbreak].get(tb_value, tb_value)
            cols.append((label, tb_df))

    # assign A/B/C… letters (skip Total)
    letters = get_topbreak_letters(len(cols) - 1)
    columns = [{"label": "Total", "letter": ""}]
    for i, (col_name, _) in enumerate(cols[1:]):
        columns.append({"label": col_name, "letter": letters[i]})

    return cols, columns
    
def build_topbreak(df, meta_df, topbreak_title, meta):
    """
    Build topbreak columns (Total + splits) based on Table_Title or Var_Grp.
    Handles SC/NR normally, expands MR/SRG into multiple columns.
    """
    if not topbreak_title:
        return [("Total", df)], [{"label": "Total", "letter": "A"}]

    # get rows for this title
    rows = meta_df[meta_df["Table_Title"] == topbreak_title]

    # If nothing matches Table_Title, maybe it’s a group title
    if rows.empty and "::" in topbreak_title:
        grp = topbreak_title.split("::")[0]
        rows = meta_df[meta_df["Var_Grp"] == grp]

    if rows.empty:
        return [("Total", df)], [{"label": "Total", "letter": "A"}]

    qtype = rows["Question_Type"].str.upper().iloc[0]

    cols = [("Total", df)]

    # --- Case 1: SC / SR / NR ---
    if qtype in ["SC", "SR", "NR"]:
        var_name = str(rows.iloc[0]["Var_Name"]).strip()
        if var_name not in df.columns:
            return cols, [{"label": "Total", "letter": "A"}]

        for tb_value, tb_df in df.groupby(var_name):
            label = str(tb_value)
            if var_name in meta.variable_value_labels:
                label = meta.variable_value_labels[var_name].get(tb_value, tb_value)
            cols.append((label, tb_df))

        # generate letters for ALL columns including Total
        letters = get_topbreak_letters(len(cols))
        columns = [{"label": col_name, "letter": letters[i]} for i, (col_name, _) in enumerate(cols)]

        return cols, columns

    # --- Case 2: MR / SRG ---
    elif qtype in ["MR", "SRG"]:
        group_rows = meta_df[meta_df["Var_Grp"] == rows.iloc[0]["Var_Grp"]]
        cols = [("Total", df)]
        letters = get_topbreak_letters(len(group_rows) + 1)  # +1 for Total

        columns = [{"label": "Total", "letter": letters[0]}]

        for i, (_, row) in enumerate(group_rows.iterrows()):
            var = str(row["Var_Name"]).strip()
            label = str(row.get("Option_Text", row.get("Var_Label", var)))
            if var not in df.columns:
                continue

            sub_df = df[df[var] == 1]  # respondents who selected this option
            cols.append((label, sub_df))
            columns.append({"label": label, "letter": letters[i + 1]})

        return cols, columns

    return cols, [{"label": "Total", "letter": "A"}]


def build_topbreak_summaries(df, var_group, group_rows, meta, children_results):
    base_table = crosstab_single_response_grid(df, var_group, group_rows, meta)

    merged = {}
    summary_keys = [
        "Top Summary", "Top2 Summary",
        "Bottom Summary", "Bottom2 Summary",
        "Middle Summary", "Mean Summary",
    ]

    for key in summary_keys:
        if not base_table.get(key):
            continue

        # Columns = Total + each topbreak
        combined = {
            "base": base_table[key]["base"],
            "columns": ["Total"] + [c["topbreak_label"].strip() for c in children_results],
            "rows": []
        }

        # Each row = Row1, Row2, Row3 … not just one row
        row_labels = base_table[key]["columns"]  # <- "Row 1", "Row 2", ...
        for ridx, row_label in enumerate(row_labels):
            new_row = {"label": row_label, "cells": []}

            # total cell
            total_cell = base_table[key]["rows"][0]["cells"][ridx]
            new_row["cells"].append(total_cell)

            # each topbreak
            for child in children_results:
                child_summary = child.get(key)
                if child_summary:
                    child_cell = child_summary["rows"][0]["cells"][ridx]
                    new_row["cells"].append(child_cell)
                else:
                    new_row["cells"].append({"count": 0, "pct": "0%"})

            combined["rows"].append(new_row)

        merged[key] = combined

    return merged

# ---------- Add Average Row to Summary ----------
def add_average_or_total_row(summary, key_name):
    """
    - For Top/Bottom summaries → add an 'Average' row (average of pcts + average of counts).
    - For Mean Summary → add a 'Total' row (overall mean across all respondents).
    """
    if not summary or "rows" not in summary:
        return summary

    cols = summary.get("columns", [])
    rows = summary.get("rows", [])
    if not rows:
        return summary

    if key_name == "Mean Summary":
        # ✅ Add Total row for Mean Summary
        total_cells = []
        for ci in range(len(cols)):
            val = rows[0]["cells"][ci].get("count")
            total_cells.append({"count": val, "pct": None})
        summary["rows"].insert(0, {"label": "Total", "cells": total_cells})

    else:
        # ✅ Average row for other summaries
        avg_cells = []
        for ci in range(len(cols)):
            col_pcts, col_counts = [], []
            for r in rows:
                if ci < len(r["cells"]):
                    cell = r["cells"][ci]
                    if "pct" in cell and cell["pct"]:
                        try:
                            col_pcts.append(float(cell["pct"].replace("%", "")))
                        except:
                            pass
                    if "count" in cell and cell["count"] not in (None, ""):
                        col_counts.append(cell["count"])

            avg_pct = f"{(sum(col_pcts)/len(col_pcts)):.1f}%" if col_pcts else "0%"
            avg_count = round(sum(col_counts)/len(col_counts)) if col_counts else None
            avg_cells.append({"count": avg_count, "pct": avg_pct})

        # Insert at top always
        summary["rows"].insert(0, {"label": "Average", "cells": avg_cells})

    return summary


# ---------- Single Choice topbreak----------
def crosstab_single_choice_with_topbreak(
    df, var_name, value_labels, topbreak_title, meta, meta_df, sig_level=95
):
    cols, columns = build_topbreak(df, meta_df, topbreak_title, meta)
    rows = []

    # Column letters (A, B, C…)
    col_letters = [c.get("letter") if isinstance(c, dict) else chr(65+i) for i, c in enumerate(columns)]

    thresholds = {80: 1.28, 90: 1.64, 95: 1.96, 99: 2.58}
    z_threshold = thresholds.get(sig_level, 1.96)

    for code, label in value_labels.items():
        row_data = {"label": f"[{code}] {label}", "cells": []}
        col_stats = []

        for col_name, sub_df in cols:
            base = sub_df[var_name].notna().sum()
            count = (sub_df[var_name] == code).sum()
            pct = f"{round((count / base * 100) if base > 0 else 0, 1)}%" if base > 0 else "0%"

            col_stats.append((count, base))
            row_data["cells"].append({"count": int(count), "pct": pct})

        # 🔹 Run sig test pairwise (skip Total = A at index 0)
        for i in range(1, len(row_data["cells"])):        # start from B
            for j in range(i + 1, len(row_data["cells"])):
                p_count, p_base = col_stats[i]
                c_count, c_base = col_stats[j]
                sig = run_sig_test(p_count, p_base, c_count, c_base, z_threshold)

                if sig == "UP":
                    row_data["cells"][j].setdefault("sig", []).append(col_letters[i])
                elif sig == "DOWN":
                    row_data["cells"][i].setdefault("sig", []).append(col_letters[j])

        # Convert sig list → string "B C"
        for c in row_data["cells"]:
            if "sig" in c:
                c["sig"] = " ".join(c["sig"])

        rows.append(row_data)

    return {"columns": columns, "rows": rows}

# ---------- Multi Response ----------
def crosstab_multi_response_with_topbreak(
    df, var_group, group_rows, meta, topbreak_title, meta_df, sig_level=95
):
    cols, columns = build_topbreak(df, meta_df, topbreak_title, meta)
    rows = []

    # Letters for columns (A = Total, but we’ll skip it in sig tests)
    col_letters = [c.get("letter") if isinstance(c, dict) else chr(65+i) for i, c in enumerate(columns)]

    thresholds = {80: 1.28, 90: 1.64, 95: 1.96, 99: 2.58}
    z_threshold = thresholds.get(sig_level, 1.96)

    for _, row in group_rows.iterrows():
        var = row["Var_Name"]
        label_col = "Option_Text" if "Option_Text" in row.index else "Var_Label"
        label = row[label_col]
        row_data = {"label": label, "cells": []}
        col_stats = []

        # Build cells (include Total in display, but skip in sig test later)
        for col_name, sub_df in cols:
            if var not in sub_df:
                base, count = 0, 0
            else:
                base = sub_df[var].notna().sum()
                count = sub_df[var].sum()
            pct = f"{round((count / base * 100) if base > 0 else 0, 1)}%" if base > 0 else "0%"
            col_stats.append((count, base))
            row_data["cells"].append({"count": int(count), "pct": pct})

        # 🔹 Sig tests (skip index 0 → Total)
        for i in range(1, len(row_data["cells"])):
            for j in range(i + 1, len(row_data["cells"])):
                p_count, p_base = col_stats[i]
                c_count, c_base = col_stats[j]
                sig = run_sig_test(p_count, p_base, c_count, c_base, z_threshold)

                if sig == "UP":
                    row_data["cells"][j].setdefault("sig", []).append(col_letters[i])
                elif sig == "DOWN":
                    row_data["cells"][i].setdefault("sig", []).append(col_letters[j])

        # Convert sig list → "B C" etc
        for c in row_data["cells"]:
            if "sig" in c:
                c["sig"] = "".join(c["sig"])

        rows.append(row_data)

    return {"columns": columns, "rows": rows}




# ---------- Numeric ----------
def crosstab_numeric_with_topbreak(df, var_name, topbreak_title, meta, meta_df):
    cols, columns = build_topbreak(df, meta_df, topbreak_title, meta)
    rows = []

    stats = ["Mean", "Median", "StdDev"]
    for stat in stats:
        row_data = {"label": stat, "cells": []}
        for col_name, sub_df in cols:
            if stat == "Mean":
                val = round(sub_df[var_name].mean(), 2) if not sub_df[var_name].empty else 0
            elif stat == "Median":
                val = round(sub_df[var_name].median(), 2) if not sub_df[var_name].empty else 0
            else:
                val = round(sub_df[var_name].std(), 2) if not sub_df[var_name].empty else 0
            row_data["cells"].append({"count": val, "pct": ""})
        rows.append(row_data)

    return {"columns": columns, "rows": rows}



# ---------- Open Ended ----------
def crosstab_open_ended_with_topbreak(df, var_name, topbreak_title, meta, meta_df):
    cols, columns = build_topbreak(df, meta_df, topbreak_title, meta)
    rows = []

    for _, row in df[[var_name]].dropna().iterrows():
        row_data = {"label": str(row[var_name]), "cells": []}
        for col_name, sub_df in cols:
            row_data["cells"].append({"count": "", "pct": ""})
        rows.append(row_data)

    return {"columns": columns, "rows": rows}

# utils.py
def get_var_name_from_title(meta_df, title):
    # try exact Table_Title match
    rows = meta_df[meta_df["Table_Title"] == title]
    if not rows.empty:
        return rows.iloc[0].get("Var_Name")
    # if you have stored titles as "grp::title", try split
    if "::" in title:
        grp, t = title.split("::", 1)
        rows = meta_df[(meta_df["Var_Grp"] == grp) & (meta_df["Table_Title"] == t)]
        if not rows.empty:
            return rows.iloc[0].get("Var_Name")
    return None


# ---------- Single Response Grid ----------
def crosstab_single_response_grid_with_topbreak(df, var_group, group_rows, meta, topbreak_title, meta_df):
    """
    Return a dict with:
      - "matrix": list of per-topbreak sub-tables
      - merged summaries (Top Summary, Top2 Summary, etc.) across Total + topbreaks
    """

    # fallback: if no valid topbreak given → behave like normal grid
    if not topbreak_title or topbreak_title not in meta_df["Table_Title"].astype(str).values:
        return crosstab_single_response_grid(df, var_group, group_rows, meta)

    results = []

    # find the meta rows for the topbreak Table_Title
    top_rows = meta_df[meta_df["Table_Title"] == topbreak_title]
    if top_rows.empty:
        return crosstab_single_response_grid(df, var_group, group_rows, meta)

    tb_qtype = top_rows["Question_Type"].str.upper().iloc[0]

    # --- Case 1: SC / SR / NR topbreak
    if tb_qtype in ["SC", "SR", "NR"]:
        tb_var = str(top_rows.iloc[0]["Var_Name"]).strip()
        if tb_var not in df.columns:
            return crosstab_single_response_grid(df, var_group, group_rows, meta)

        split_values = [v for v in df[tb_var].dropna().unique()]
        split_labels = [
            meta.variable_value_labels.get(tb_var, {}).get(v, str(v)) for v in split_values
        ]

        for sv, slabel in zip(split_values, split_labels):
            tb_df = df[df[tb_var] == sv]
            sub_table = crosstab_single_response_grid(tb_df, var_group, group_rows, meta)
            results.append({
                "topbreak_label": slabel,
                **sub_table
            })

    # --- Case 2: MR / SRG topbreak
    elif tb_qtype in ["MR", "SRG"]:
        for _, r in top_rows.iterrows():
            child_var = str(r["Var_Name"]).strip()
            child_label = r.get("Option_Text", r.get("Var_Label", child_var))
            if child_var not in df.columns:
                continue

            tb_df = df[df[child_var] == 1]
            sub_table = crosstab_single_response_grid(tb_df, var_group, group_rows, meta)
            results.append({
                "topbreak_label": child_label,
                **sub_table
            })

    else:
        # fallback
        return crosstab_single_response_grid(df, var_group, group_rows, meta)

    # ✅ Build merged summaries ONLY for topbreak mode
    merged_summaries = build_topbreak_summaries(df, var_group, group_rows, meta, results)

    return {
        "matrix": results,          # per-topbreak crosstabs
        **merged_summaries          # merged summaries across Total + all topbreaks
    }









# def crosstab_single_choice(df, var_name, value_labels, topbreaks=None, add_type=""):
    results = {}

    total_base = df[var_name].notna().sum()
    freq = df[var_name].value_counts(dropna=True)
    freq_pct = df[var_name].value_counts(normalize=True, dropna=True) * 100

    rows = []

    if str(add_type).lower() == "nps":
        print("Add Type:", add_type)
        # --- Distribution 0–10 ---
        ordered_codes = sorted(freq.index.tolist())  # ensure only observed values 0..10
        for code in ordered_codes:
            code_str = str(int(code)) if isinstance(code, (int, float)) and code == int(code) else str(code)
            count = int(freq.get(code, 0))
            pct = round(freq_pct.get(code, 0.0), 1)
            label = f"[{code_str}]"  # no label, just the number
            rows.append({"label": label, "count": count, "pct": f"{pct}%"})

        # --- Promoters, Detractors ---
        promoters = sum(int(freq.get(c, 0)) for c in [9, 10])
        detractors = sum(int(freq.get(c, 0)) for c in range(0, 7))

        promoters_pct = (promoters / total_base * 100) if total_base > 0 else 0
        detractors_pct = (detractors / total_base * 100) if total_base > 0 else 0
        nps_score = round(promoters_pct - detractors_pct, 1)

        # --- Add NPS row ---
        rows.append({
            "label": "NPS Score",
            "count": nps_score,
            "pct": ""
        })

    else:
        # 🔹 Normal SC behaviour
        merged = {}
        for code, label in value_labels.items():
            if code in freq:
                code_str = str(int(code)) if isinstance(code, (int, float)) and code == int(code) else str(code)
                group_key = str(label).strip().lower()

                if group_key not in merged:
                    merged[group_key] = {
                        "label": f"[{code_str}] {label}",
                        "count": 0,
                        "pct": 0.0,
                    }

                merged[group_key]["count"] += int(freq[code])
                merged[group_key]["pct"] += round(float(freq_pct[code]), 1)

        rows = list(merged.values())

    results["Total"] = {"base": int(total_base), "rows": rows}
    return results


def safe_number(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return int(x) if isinstance(x, (int, float)) and x == int(x) else x


def crosstab_multi_response(df, var_group, meta_rows, meta):
    results = {}

    vars_in_group = meta_rows["Var_Name"].tolist()
    base = int(df[vars_in_group].notna().any(axis=1).sum())

    merged = {}

    for _, r in meta_rows.iterrows():
        var_name = r["Var_Name"]
        raw_label = r.get("Var_Label")
        if pd.isna(raw_label) or str(raw_label).strip() == "" or str(raw_label).lower() == "nan":
            var_label = var_name
        else:
            var_label = str(raw_label).strip()

        if var_name not in df.columns:
            continue

        count = int(df[var_name].apply(lambda x: 1 if pd.notna(x) and x != 0 else 0).sum())
        pct = round((count / base) * 100, 1) if base > 0 else 0.0

        key = var_label.lower()
        if key not in merged:
            merged[key] = {"label": var_label, "count": 0, "pct": 0.0}
        merged[key]["count"] += count
        merged[key]["pct"] = f"{pct}%"

    rows = []
    seen = set()
    for _, r in meta_rows.iterrows():
        label = str(r.get("Var_Label", r.get("Var_Name", ""))).strip()
        key = label.lower()
        if key in merged and key not in seen:
            rows.append(merged[key])
            seen.add(key)

    rows = merge_others(rows, base)   # ✅ merge here

    results["Total"] = {"base": base, "rows": rows}
    return results


def crosstab_numeric(df, var_name):
    series = df[var_name].dropna()
    base = len(series)
    mean_val = round(series.mean(), 2) if base > 0 else 0
    median_val = round(series.median(), 2) if base > 0 else 0
    std_val = round(series.std(), 2) if base > 1 else 0

    rows = [
        {"label": "Mean", "count": mean_val, "pct": ""},
        {"label": "Median", "count": median_val, "pct": ""},
        {"label": "Standard Deviation", "count": std_val, "pct": ""}
    ]

    rows = merge_others(rows, base)   # ✅ safe, though usually won’t match

    return {"Total": {"base": base, "rows": rows}}


def crosstab_open_ended(df, var_name, top_n=20):
    series = df[var_name].dropna().astype(str).str.strip()
    base = len(series)

    counts = series.value_counts().reset_index()
    counts.columns = ["response", "count"]

    rows = []
    for _, row in counts.head(top_n).iterrows():
        rows.append({
            "label": row["response"],
            "count": int(row["count"]),
            "pct": f"{round((row['count'] / base) * 100, 1)} %"
        })


    rows = merge_others(rows, base)   # ✅ merge here

    return {"Total": {"base": base, "rows": rows}}


def format_code_label(code, value_labels):
    if isinstance(code, float) and code.is_integer():
        code = int(code)
    return f"[{code}] {value_labels.get(code, str(code))}"


def crosstab_single_response_grid(df, var_group, meta_rows, meta):
    attributes = []
    attr_data = {}
    total_base = 0

    add_type = str(meta_rows.iloc[0].get("Add_Question_Type", "")).strip().lower()

    for _, r in meta_rows.iterrows():
        var_name = r["Var_Name"]
        attr_label = r.get("Var_Label", var_name)
        attributes.append((var_name, attr_label))

        if var_name not in df.columns:
            continue

        base = df[var_name].notna().sum()
        total_base += base
        counts = df[var_name].value_counts().to_dict()

        attr_data[var_name] = {
            "label": attr_label,
            "base": base,
            "counts": counts,
            "series": df[var_name].dropna()
        }

    first_var = meta_rows.iloc[0]["Var_Name"]
    value_labels = meta.variable_value_labels.get(first_var, {})

    # Auto-detect type if missing
    if not add_type:
        if all(str(k).isdigit() for k in value_labels.keys()):
            add_type = "rating"
        else:
            add_type = "categorical"

    # --- scale detection ---
    scale_codes = sorted(value_labels.keys())
    scale_length = len(scale_codes)
    ordered_codes = scale_codes

    rows = []

    # --- Total row ---
    total_row = {"label": "Total", "cells": []}
    for var_name, _ in attributes:
        base = attr_data[var_name]["base"]
        total_row["cells"].append({"count": base, "pct": "100%"})
    rows.append(total_row)

    if add_type == "rating":
        bottom_n = 2 if scale_length <= 5 else 3
        bottom_codes = scale_codes[:bottom_n]
        bottom_row = {"label": f"Bottom {bottom_n}", "cells": []}
        for var_name, _ in attributes:
            counts = attr_data[var_name]["counts"]
            base = attr_data[var_name]["base"]
            count = sum(int(counts.get(c, 0)) for c in bottom_codes)
            pct = f"{round((count/base)*100,1)}%" if base > 0 else "0%"
            bottom_row["cells"].append({"count": count, "pct": pct})
        rows.append(bottom_row)

    # --- Scale rows ---
    for code in ordered_codes:
        row_cells = []
        for var_name, _ in attributes:
            counts = attr_data[var_name]["counts"]
            base = attr_data[var_name]["base"]
            count = int(counts.get(code, 0))
            pct = f"{round((count/base)*100,1)}%" if base > 0 else "0%"
            row_cells.append({"count": count, "pct": pct})
        label = format_code_label(code, value_labels)
        rows.append({"label": label, "cells": row_cells})

    if add_type == "rating":
        top_n = 2 if scale_length <= 5 else 3
        top_codes = scale_codes[-top_n:]
        top_row = {"label": f"Top {top_n}", "cells": []}
        for var_name, _ in attributes:
            counts = attr_data[var_name]["counts"]
            base = attr_data[var_name]["base"]
            count = sum(int(counts.get(c, 0)) for c in top_codes)
            pct = f"{round((count/base)*100,1)}%" if base > 0 else "0%"
            top_row["cells"].append({"count": count, "pct": pct})
        rows.append(top_row)

        # --- Mean, Median, StdDev ---
        mean_row, median_row, std_row = (
            {"label": "Mean", "cells": []},
            {"label": "Median", "cells": []},
            {"label": "StdDev", "cells": []},
        )
        for var_name, _ in attributes:
            series = attr_data[var_name]["series"]
            if len(series) > 0:
                mean_row["cells"].append({"count": round(series.mean(), 2)})
                median_row["cells"].append({"count": round(series.median(), 2)})
                std_row["cells"].append({"count": round(series.std(), 2)})
            else:
                mean_row["cells"].append({"count": 0})
                median_row["cells"].append({"count": 0})
                std_row["cells"].append({"count": 0})
        rows.extend([mean_row, median_row, std_row])

    # ✅ merge "Others"
    rows = merge_others(rows, total_base, is_cells=True)

    result = {
        "Matrix": {
            "base": total_base,
            "columns": [label for _, label in attributes],
            "rows": rows,
            "type": add_type,
        }
    }

    # --- add summaries only for rating ---
    if add_type == "rating":
        middle_code = scale_codes[len(scale_codes)//2] if len(scale_codes) % 2 == 1 else None

        def build_summary(label, codes):
            row = {"label": label, "cells": []}
            for var_name, _ in attributes:
                counts = attr_data[var_name]["counts"]
                base = attr_data[var_name]["base"]
                count = sum(int(counts.get(c, 0)) for c in codes)
                pct = f"{round((count/base)*100,1)}%" if base > 0 else "0%"
                row["cells"].append({"count": count, "pct": pct})
            return row

        def build_summary_table(summary_label, codes):
            return {
                "base": total_base,
                "columns": [label for _, label in attributes],
                "rows": [build_summary(summary_label, codes)]
            }

        # ✅ per-column summaries
        result["Top Summary"] = build_summary_table("Top Summary", [scale_codes[-1]])
        result["Top2 Summary"] = build_summary_table("Top2 Summary", scale_codes[-2:])
        result["Bottom Summary"] = build_summary_table("Bottom Summary", [scale_codes[0]])
        result["Bottom2 Summary"] = build_summary_table("Bottom2 Summary", scale_codes[:2])

        if middle_code is not None:
            result["Middle Summary"] = build_summary_table("Middle Summary", [middle_code])

        # ✅ Mean Summary (per column)
        mean_row = {"label": "Mean", "cells": []}
        for var_name, _ in attributes:
            series = attr_data[var_name]["series"]
            mean_row["cells"].append({"count": round(series.mean(), 2) if len(series) > 0 else 0})
        result["Mean Summary"] = {
            "base": total_base,
            "columns": [label for _, label in attributes],
            "rows": [mean_row]
        }

        # ✅ Apply average/total row adjustments
        summary_keys = [
            "Top Summary",
            "Top2 Summary",
            "Bottom Summary",
            "Bottom2 Summary",
            "Middle Summary",
            "Mean Summary"
        ]

        for key in summary_keys:
            if key in result and result[key]:
                result[key] = add_average_or_total_row(result[key], key)
    return result




# --- Load Project Meta ---
def load_project_meta(project_id):
    """
    Load the project's meta Excel as a DataFrame.
    Returns (meta_df, error_message).
    """
    project = Project.objects(id=project_id).first()
    if not project or not project.meta_file:
        return None, "Meta Excel not found"

    meta_path = os.path.join(settings.MEDIA_ROOT, project.meta_file)
    try:
        meta_df = pd.read_excel(meta_path)
        return meta_df, None
    except Exception as e:
        return None, str(e)
    

