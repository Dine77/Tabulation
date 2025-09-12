from .models import ActivityLog
from datetime import datetime
import pandas as pd
import pyreadstat
import math
import re


def log_activity(user, action, project_name="", file_name="", details=""):
    ActivityLog(
        user=user,
        action=action,
        project_name=project_name,
        file_name=file_name,
        details=details,
        timestamp=datetime.utcnow()
    ).save()


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
            merged["pct"] = f"{pct:.1f}"
        cleaned.append(merged)

    return cleaned


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
            merged[group_key]["pct"] += round(float(freq_pct[code]), 1)

    rows = list(merged.values())
    rows = merge_others(rows, total_base)   # ✅ merge here

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
        merged[key]["pct"] += pct

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
        rows.append({"label": row["response"], "count": int(row["count"]), "pct": round((row["count"]/base)*100, 1)})

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

    if not add_type:
        if all(str(k).isdigit() for k in value_labels.keys()):
            add_type = "rating"
        else:
            add_type = "categorical"

    ordered_codes = sorted(value_labels.keys(), reverse=True)
    rows = []

    total_row = {"label": "Total", "cells": []}
    for var_name, _ in attributes:
        base = attr_data[var_name]["base"]
        total_row["cells"].append({"count": base, "pct": "100%"})
    rows.append(total_row)

    if add_type == "rating":
        bottom_row = {"label": "Bottom 2", "cells": []}
        for var_name, _ in attributes:
            counts = attr_data[var_name]["counts"]
            base = attr_data[var_name]["base"]
            count = int(counts.get(1, 0)) + int(counts.get(2, 0))
            pct = f"{round((count/base)*100,1)}%" if base > 0 else "0%"
            bottom_row["cells"].append({"count": count, "pct": pct})
        rows.append(bottom_row)

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
        top_row = {"label": "Top 2", "cells": []}
        for var_name, _ in attributes:
            counts = attr_data[var_name]["counts"]
            base = attr_data[var_name]["base"]
            count = int(counts.get(4, 0)) + int(counts.get(5, 0))
            pct = f"{round((count/base)*100,1)}%" if base > 0 else "0%"
            top_row["cells"].append({"count": count, "pct": pct})
        rows.append(top_row)

        mean_row, median_row, std_row = {"label": "Mean", "cells": []}, {"label": "Median", "cells": []}, {"label": "StdDev", "cells": []}
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

    # ✅ merge "Others" in SRG rows
    rows = merge_others(rows, total_base, is_cells=True)

    return {
        "Total": {
            "base": total_base,
            "columns": [label for _, label in attributes],
            "rows": rows,
            "type": add_type
        }
    }
