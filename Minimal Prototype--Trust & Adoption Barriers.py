#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: Minimal Prototype--Trust & Adoption Barriers
# Author: Rod Villalobos using rapid prototyping.
# Purpose: Show an AI prototype associated with trust and adoption barriers.

import numpy as np, pandas as pd, matplotlib.pyplot as plt

np.random.seed(222)

# -----------------------------
# 0) Config (tweak to your context)
# -----------------------------
N_USERS = 300
TARGETS = {
    "perceived_accuracy": 0.80,   # target user-perceived accuracy (0..1)
    "transparency":       0.75,   # explanatory depth, citations, why/how (0..1)
    "control":            0.70,   # opt-in/out, human-in-loop, revision controls (0..1)
    "helpfulness":        0.78,   # perceived task support (0..1)
    "privacy_safety":     0.80,   # low concern → high safety perception (we invert concern below)
    "latency_ms":         800,    # p95 acceptable
    "onboarding":         0.70,   # training / guidance quality (0..1)
}

WEIGHTS = {  # weights for Trust Score (sum doesn't need to be 1; relative importance matters)
    "perceived_accuracy": 0.25,
    "transparency":       0.18,
    "control":            0.15,
    "helpfulness":        0.15,
    "privacy_safety":     0.14,
    "latency":            0.07,   # lower latency → more trust
    "onboarding":         0.06,
}

# -----------------------------
# 1) Simulate user perceptions (0..1 except latency in ms)
# -----------------------------
def clamp01(x): return np.clip(x, 0, 1)

# Baselines with variance; you can skew these to mirror your org
perceived_accuracy = clamp01(np.random.normal(0.72, 0.10, N_USERS))
transparency       = clamp01(np.random.normal(0.55, 0.15, N_USERS))
control            = clamp01(np.random.normal(0.58, 0.18, N_USERS))
helpfulness        = clamp01(np.random.normal(0.68, 0.12, N_USERS))
privacy_concern    = clamp01(np.random.beta(2.5, 3.5, N_USERS))   # higher = more concern
privacy_safety     = clamp01(1 - privacy_concern)                 # invert to align with trust
latency_ms         = np.random.normal(900, 180, N_USERS).clip(200, 2000)
onboarding         = clamp01(np.random.normal(0.62, 0.15, N_USERS))

# Friction proxies for adoption (0..1 higher = more friction)
time_to_value      = clamp01(np.random.normal(0.40, 0.18, N_USERS))   # how long until benefit felt
training_needed    = clamp01(np.random.normal(0.45, 0.20, N_USERS))

df = pd.DataFrame({
    "perceived_accuracy": perceived_accuracy,
    "transparency": transparency,
    "control": control,
    "helpfulness": helpfulness,
    "privacy_safety": privacy_safety,
    "latency_ms": latency_ms,
    "onboarding": onboarding,
    "time_to_value": time_to_value,
    "training_needed": training_needed,
})

# -----------------------------
# 2) Trust Score (0..100) + Adoption Likelihood (0..1)
# -----------------------------
# Normalize latency so that lower latency → higher score in 0..1
lat_norm = (TARGETS["latency_ms"] / df["latency_ms"]).clip(0, 1.5)
lat_norm = np.minimum(lat_norm, 1.0)  # cap at 1

trust_0to1 = (
    WEIGHTS["perceived_accuracy"] * df["perceived_accuracy"] +
    WEIGHTS["transparency"]       * df["transparency"] +
    WEIGHTS["control"]            * df["control"] +
    WEIGHTS["helpfulness"]        * df["helpfulness"] +
    WEIGHTS["privacy_safety"]     * df["privacy_safety"] +
    WEIGHTS["latency"]            * lat_norm +
    WEIGHTS["onboarding"]         * df["onboarding"]
) / sum(WEIGHTS.values())

df["trust_score"] = (trust_0to1 * 100).clip(0, 100)

# Adoption likelihood uses trust and subtracts friction (time_to_value + training_needed)
# Sigmoid for smooth 0..1 mapping; tweak k and bias for strictness
k, bias = 6.0, -3.0
signal = (trust_0to1 - 0.60) - 0.25*(df["time_to_value"] + df["training_needed"] - 0.6)  # center friction around 0.6
df["adoption_likelihood"] = 1 / (1 + np.exp(- (k*signal + bias)))

# -----------------------------
# 3) Barriers: gaps vs targets + correlation with adoption
# -----------------------------
# Gap vs target (positive gap = shortfall to target)
gaps = {
    "perceived_accuracy_gap": TARGETS["perceived_accuracy"] - df["perceived_accuracy"].mean(),
    "transparency_gap":       TARGETS["transparency"]       - df["transparency"].mean(),
    "control_gap":            TARGETS["control"]            - df["control"].mean(),
    "helpfulness_gap":        TARGETS["helpfulness"]        - df["helpfulness"].mean(),
    "privacy_safety_gap":     TARGETS["privacy_safety"]     - df["privacy_safety"].mean(),
    "latency_gap":            (df["latency_ms"].mean() - TARGETS["latency_ms"]) / TARGETS["latency_ms"],  # >0 is worse
    "onboarding_gap":         TARGETS["onboarding"]         - df["onboarding"].mean(),
}

# Simple correlations (Pearson) with adoption likelihood
def corr(x, y):
    if np.std(x) < 1e-9 or np.std(y) < 1e-9: return 0.0
    return float(np.corrcoef(x, y)[0,1])

correls = {
    "perceived_accuracy": corr(df["perceived_accuracy"], df["adoption_likelihood"]),
    "transparency":       corr(df["transparency"],       df["adoption_likelihood"]),
    "control":            corr(df["control"],            df["adoption_likelihood"]),
    "helpfulness":        corr(df["helpfulness"],        df["adoption_likelihood"]),
    "privacy_safety":     corr(df["privacy_safety"],     df["adoption_likelihood"]),
    "latency_ms":         corr(-df["latency_ms"],        df["adoption_likelihood"]),  # negative latency helps adoption
    "onboarding":         corr(df["onboarding"],         df["adoption_likelihood"]),
}

# Rank barriers by a composite of (gap normalized * negative impact)
# For latency, treat positive latency_gap as a gap directly
gap_series = pd.Series(gaps)
# Normalize gaps to comparable scale
gap_norm = (gap_series - gap_series.min()) / (gap_series.max() - gap_series.min() + 1e-9)
# Map correlation keys to gaps
impact_map = {
    "perceived_accuracy_gap": abs(correls["perceived_accuracy"]),
    "transparency_gap":       abs(correls["transparency"]),
    "control_gap":            abs(correls["control"]),
    "helpfulness_gap":        abs(correls["helpfulness"]),
    "privacy_safety_gap":     abs(correls["privacy_safety"]),
    "latency_gap":            abs(correls["latency_ms"]),
    "onboarding_gap":         abs(correls["onboarding"]),
}
impact_series = pd.Series(impact_map).reindex(gap_series.index)
barrier_score = (gap_norm * impact_series).sort_values(ascending=False)

top_barriers = barrier_score.head(5)

# -----------------------------
# 4) Recommendations (rule-based from barriers)
# -----------------------------
def interventions(barrier_name):
    b = barrier_name
    if "transparency" in b:
        return ["Add inline ‘Why/How’ explanations", "Show citations/sources", "Expose model confidence"]
    if "control" in b:
        return ["Human-in-loop approvals for risky actions", "Granular opt-out & data controls", "Undo/Version history"]
    if "perceived_accuracy" in b:
        return ["Pilot with high-quality exemplars", "Tighten evaluation & QA gates", "Flag low-confidence outputs"]
    if "helpfulness" in b:
        return ["Task-specific prompts/templates", "Context prefill from user data", "Improve retrieval grounding"]
    if "privacy_safety" in b:
        return ["Clarify data retention & PII handling", "Local processing for sensitive fields", "Add safe modes"]
    if "latency" in b:
        return ["Async jobs for heavy tasks", "Batch & cache common queries", "Autoscale hot paths"]
    if "onboarding" in b:
        return ["90-second guided tour", "Role-based quickstarts", "In-product tips & examples"]
    return ["Interview users to pinpoint friction", "Run A/B on UI cues"]

reco_table = [(bn, round(barrier_score[bn],3), interventions(bn)[:2]) for bn in top_barriers.index]

# -----------------------------
# 5) Summary prints
# -----------------------------
print("=== TRUST & ADOPTION SUMMARY ===")
print({
    "avg_trust_score": round(float(df['trust_score'].mean()), 1),
    "median_trust_score": round(float(df['trust_score'].median()), 1),
    "adoption_rate_>50%": f"{round(100*float((df['adoption_likelihood']>=0.5).mean()),1)}%",
    "top_barriers": list(top_barriers.index),
})

print("\n=== TOP BARRIERS (score & first two interventions) ===")
for name, score, recs in reco_table:
    print(f"- {name}: {score}  →  {recs[0]}; {recs[1]}")

print("\n=== CORRELATIONS (feature → adoption) ===")
print({k: round(v,3) for k,v in correls.items()})

# -----------------------------
# 6) Visuals
# -----------------------------
plt.figure(figsize=(7,4))
plt.hist(df["trust_score"], bins=20)
plt.title("Trust Score Distribution")
plt.xlabel("Trust Score (0..100)")
plt.ylabel("Users")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
sel = barrier_score.head(6)
plt.bar(sel.index, sel.values)
plt.title("Top Barrier Scores (Gap × Impact)")
plt.ylabel("Composite Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# -----------------------------
# 7) (Optional) Scenario: what if we improve top-1 barrier by +20%?
# -----------------------------
top1 = top_barriers.index[0]
df2 = df.copy()
if top1 == "transparency_gap":
    df2["transparency"] = np.clip(df2["transparency"]*1.2, 0, 1)
elif top1 == "control_gap":
    df2["control"] = np.clip(df2["control"]*1.2, 0, 1)
elif top1 == "perceived_accuracy_gap":
    df2["perceived_accuracy"] = np.clip(df2["perceived_accuracy"]*1.2, 0, 1)
elif top1 == "helpfulness_gap":
    df2["helpfulness"] = np.clip(df2["helpfulness"]*1.2, 0, 1)
elif top1 == "privacy_safety_gap":
    df2["privacy_safety"] = np.clip(df2["privacy_safety"]*1.2, 0, 1)
elif top1 == "latency_gap":
    df2["latency_ms"] = np.maximum(200, df2["latency_ms"]*0.8)
elif top1 == "onboarding_gap":
    df2["onboarding"] = np.clip(df2["onboarding"]*1.2, 0, 1)

lat2 = (TARGETS["latency_ms"] / df2["latency_ms"]).clip(0, 1.5)
lat2 = np.minimum(lat2, 1.0)
trust2 = (
    WEIGHTS["perceived_accuracy"] * df2["perceived_accuracy"] +
    WEIGHTS["transparency"]       * df2["transparency"] +
    WEIGHTS["control"]            * df2["control"] +
    WEIGHTS["helpfulness"]        * df2["helpfulness"] +
    WEIGHTS["privacy_safety"]     * df2["privacy_safety"] +
    WEIGHTS["latency"]            * lat2 +
    WEIGHTS["onboarding"]         * df2["onboarding"]
) / sum(WEIGHTS.values())
adopt2 = 1 / (1 + np.exp(- (6.0*((trust2 - 0.60) - 0.25*(df2["time_to_value"] + df2["training_needed"] - 0.6)) - 3.0)))

print("\n=== WHAT-IF SCENARIO (improve top barrier +20%) ===")
print({
    "avg_trust_before": round(float(df['trust_score'].mean()),1),
    "avg_trust_after":  round(float((trust2*100).mean()),1),
    "adoption_rate_>50%_before": f"{round(100*float((df['adoption_likelihood']>=0.5).mean()),1)}%",
    "adoption_rate_>50%_after":  f"{round(100*float((adopt2>=0.5).mean()),1)}%",
})


# In[ ]:




