#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Name: Regulatory & Compliance Complexity
# Author: Rod Villalobos using rapid prototyping.
# Purpose: Show an AI prototype associated with regulatory & compliance complexity.

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(33)

# --- (Optional) CSV inputs (set paths to use your own) ---
REQS_CSV = None      # e.g., "requirements.csv"
CTRLS_CSV = None     # e.g., "controls.csv"

# Expected schema if you load CSVs:
# requirements.csv columns: req_id, regulation, section, requirement, criticality [High/Med/Low], evidence_type, jurisdiction
# controls.csv     columns: control_id, control_type, mapped_sections (semicolon-joined), regulation, status [Implemented/Planned], last_review_date (YYYY-MM-DD)

# --- 1) Requirements (synthetic if no CSVs) ---
if REQS_CSV and os.path.exists(REQS_CSV):
    reqs = pd.read_csv(REQS_CSV)
else:
    regs = [
        ("Part 11",  ["11.10(a)","11.10(b)","11.10(c)","11.50","11.70"]),
        ("GCP",      ["ICH E6 2.10","ICH E6 5.5.3","ICH E6 8.3.13","ICH E6 8.3.14"]),
        ("GMP",      ["211.68","211.100(a)","211.194","211.25"]),
        ("Privacy",  ["Access","Retention","Audit","Breach"]),
    ]
    rows = []
    evid_types = ["SOP","Validation","AuditTrail","AccessControl","Training","VendorQual"]
    crits = ["High","Med","Low"]
    for reg, sections in regs:
        for s in sections:
            rows.append({
                "req_id": f"{reg}-{s}",
                "regulation": reg,
                "section": s,
                "requirement": f"Comply with {reg} section {s} and retain evidence.",
                "criticality": np.random.choice(crits, p=[0.45,0.4,0.15]),
                "evidence_type": np.random.choice(evid_types),
                "jurisdiction": np.random.choice(["US","EU","Global"], p=[0.5,0.2,0.3]),
            })
    reqs = pd.DataFrame(rows)

# --- 2) Controls inventory (synthetic if no CSVs) ---
if CTRLS_CSV and os.path.exists(CTRLS_CSV):
    ctrls = pd.read_csv(CTRLS_CSV)
else:
    ctrl_rows = []
    all_req_pairs = reqs[["regulation","section"]].drop_duplicates().values.tolist()
    control_types = ["SOP","Validation","AuditTrail","AccessControl","Training","VendorQual"]
    for i in range(1, 22):  # ~21 controls
        # Map each control to 1–3 sections from the universe
        mapped_n = np.random.choice([1,1,2,2,3], p=[0.3,0.3,0.2,0.15,0.05])
        mapped = np.random.choice(len(all_req_pairs), size=mapped_n, replace=False)
        mapped_pairs = [all_req_pairs[j] for j in mapped]
        # Assign a regulation label (primary of the first mapped)
        reg_lab = mapped_pairs[0][0]

        ctrl_rows.append({
            "control_id": f"CTRL-{i:03d}",
            "control_type": np.random.choice(control_types),
            "mapped_sections": ";".join([f"{p[0]}::{p[1]}" for p in mapped_pairs]),
            "regulation": reg_lab,
            "status": np.random.choice(["Implemented","Planned"], p=[0.7,0.3]),
            "last_review_date": (datetime(2025,1,1) - timedelta(days=int(np.random.choice([60,120,200,380,800])))).strftime("%Y-%m-%d"),
            "owner": np.random.choice(["QA","IT","ClinicalOps","RA"]),
        })
    ctrls = pd.DataFrame(ctrl_rows)

# --- 3) Normalize + explode controls to (regulation, section) rows ---
def explode_mapped_sections(df):
    df = df.copy()
    df["mapped_sections"] = df["mapped_sections"].fillna("")
    df = df.assign(mapped=df["mapped_sections"].str.split(";"))
    df = df.explode("mapped")
    df[["map_reg","map_section"]] = df["mapped"].str.split("::", n=1, expand=True)
    df["map_reg"] = df["map_reg"].fillna(df["regulation"])
    df["map_section"] = df["map_section"].fillna("")
    return df.drop(columns=["mapped"])

ctrls_exp = explode_mapped_sections(ctrls)

# --- 4) Join controls to requirements on (regulation, section) ---
joined = reqs.merge(
    ctrls_exp,
    left_on=["regulation","section"],
    right_on=["map_reg","map_section"],
    how="left",
    suffixes=("","_ctrl")
)

# --- 5) Flags per requirement ---
def days_since(date_str):
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d")
        return (datetime(2025,10,15) - d).days  # fixed “today” for reproducibility
    except Exception:
        return None

agg = joined.groupby(["req_id","regulation","section","criticality","evidence_type"])    .apply(lambda g: pd.Series({
        "controls_total": g["control_id"].notna().sum(),
        "implemented": (g["status"]=="Implemented").sum(),
        "planned": (g["status"]=="Planned").sum(),
        "has_matching_evidence_type": ((g["control_type"]==g["evidence_type"]) & g["control_id"].notna()).any(),
        "max_days_since_review": np.nanmax([days_since(x) for x in g["last_review_date"].fillna("").tolist()]) if g["control_id"].notna().any() else np.nan,
    })).reset_index()

# Conditions
agg["missing_control"]  = agg["controls_total"].eq(0)
agg["planned_only"]     = (agg["implemented"].eq(0)) & (agg["planned"].gt(0))
agg["outdated_review"]  = agg["max_days_since_review"].fillna(9999).gt(365)  # > 1 year
agg["ambiguous"]        = (agg["controls_total"].gt(1)) & (~agg["has_matching_evidence_type"])

# --- 6) Risk scoring ---
crit_weight = agg["criticality"].map({"High":3,"Med":2,"Low":1}).fillna(1)
risk = (
    2*agg["missing_control"].astype(int) +
    1*agg["planned_only"].astype(int) +
    1*agg["outdated_review"].astype(int) +
    1*agg["ambiguous"].astype(int)
)
agg["risk_score"] = (crit_weight * (1 + risk)).astype(int)

# Category label
def bucket(x):
    return "Critical" if x>=8 else ("High" if x>=6 else ("Med" if x>=4 else "Low"))
agg["risk_bucket"] = agg["risk_score"].apply(bucket)

# Coverage %
coverage_pct = round(100.0 * (1 - agg["missing_control"].mean()), 1)

# --- 7) Summaries ---
print("=== REGULATORY / COMPLIANCE SUMMARY ===")
print({
    "requirements_total": int(len(agg)),
    "coverage_percent": coverage_pct,
    "critical_risks": int((agg["risk_bucket"]=="Critical").sum()),
    "high_risks": int((agg["risk_bucket"]=="High").sum()),
    "med_risks": int((agg["risk_bucket"]=="Med").sum()),
    "low_risks": int((agg["risk_bucket"]=="Low").sum()),
}, "\n")

# Top risky requirements
top = agg.sort_values(["risk_score","criticality"], ascending=[False,True]).head(10)
print("=== TOP RISKY REQUIREMENTS (head) ===")
print(top[[
    "req_id","criticality","risk_score","missing_control","planned_only","outdated_review","ambiguous","has_matching_evidence_type"
]].to_string(index=False), "\n")

# Per-regulation counts (optional print)
by_reg = agg.groupby("regulation")["risk_bucket"].value_counts().unstack(fill_value=0)
print("=== RISK BUCKETS BY REGULATION ===")
print(by_reg, "\n")

# --- 8) Visualization: Top 10 risk scores ---
plt.figure(figsize=(8,4))
plt.bar(top["req_id"], top["risk_score"])
plt.title("Top Regulatory Risks (by Requirement)")
plt.ylabel("Risk Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:




