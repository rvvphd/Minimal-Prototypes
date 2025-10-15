#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: Minimal Prototype--Competition & Differentiation
# Author: Rod Villalobos using rapid prototyping
# Purpose: Show an AI prototype associated with competition & differentiation.

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

# --- 0) Config ---
CSV_PATH = None  # e.g., "competitors.csv" with columns listed above
OUR_NAME = "OurProduct"

# Feature weights (sum is arbitrary; relative importance matters)
WEIGHTS = {
    "performance": 0.25,
    "integration": 0.20,
    "compliance":  0.20,
    "ux":          0.15,
    "support":     0.10,
    "ecosystem":   0.10,
}
PRICE_WEIGHT = 0.25  # used only for positioning/plot; not part of differentiation by feature

# --- 1) Data (load or synthesize) ---
cols = ["name","price","performance","integration","compliance","ux","support","ecosystem"]
if CSV_PATH and os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    assert all(c in df.columns for c in cols), f"CSV must include: {cols}"
else:
    np.random.seed(7)
    df = pd.DataFrame({
        "name": ["OurProduct","Alpha","Beta","Gamma","Delta","Epsilon"],
        "price":       [120, 150, 95, 180, 140, 110],      # lower is better
        "performance": [8.6, 7.4, 6.9, 8.2, 7.0, 6.8],
        "integration": [8.2, 6.4, 6.0, 7.1, 6.2, 6.5],
        "compliance":  [8.8, 6.8, 6.2, 7.5, 6.6, 6.4],
        "ux":          [7.9, 7.1, 6.5, 7.3, 6.7, 6.8],
        "support":     [8.1, 6.9, 6.3, 7.2, 6.5, 6.6],
        "ecosystem":   [7.6, 6.1, 6.7, 7.0, 6.2, 6.4],
    })

# --- 2) Basic sanity and normalization ---
features = list(WEIGHTS.keys())
assert OUR_NAME in df["name"].values, f"'{OUR_NAME}' must be present in 'name' column."

# Normalize each feature to 0..1 using min-max (if constant, set to 0.5 to avoid div/0)
norm = df.copy()
for c in features:
    lo, hi = df[c].min(), df[c].max()
    norm[c] = 0.5 if hi == lo else (df[c] - lo) / (hi - lo)

# For price, invert and normalize so "lower price → higher normalized value"
lo, hi = df["price"].min(), df["price"].max()
norm["price_inv"] = 0.5 if hi == lo else (hi - df["price"]) / (hi - lo)

# --- 3) Scores ---
# Positioning Score combines capability features using WEIGHTS (not price)
cap_scores = np.zeros(len(df))
for f, w in WEIGHTS.items():
    cap_scores += norm[f].values * w

# Differentiation Score (per feature): OurProduct normalized feature minus competitors' mean
market_means = norm[features].mean(axis=0)
our_row = norm[norm["name"] == OUR_NAME].iloc[0]
diff_vector = (our_row[features] - market_means).sort_values(ascending=False)

# Simple combined Position score for plotting (capability vs price inverted)
position_df = norm[["name"]].copy()
position_df["capability_score"] = cap_scores
position_df["price_score_inv"] = norm["price_inv"]

# --- 4) Text summary: strengths, weaknesses, whitespace ---
top_strengths = diff_vector.head(3)
top_weaknesses = diff_vector.tail(3)

# "Whitespace": features where the ENTIRE market (excluding us) is weak (mean below threshold)
# Threshold heuristic: < 0.45 normalized → generally weak; adjust to taste
market_means_ex_ours = norm[norm["name"] != OUR_NAME][features].mean(axis=0)
whitespace_features = list(market_means_ex_ours[market_means_ex_ours < 0.45].index)

print("=== COMPETITION & DIFFERENTIATION SUMMARY ===")
print("Top Strengths (normalized gaps vs market):")
print(top_strengths.to_dict(), "\n")
print("Top Weaknesses (normalized gaps vs market):")
print(top_weaknesses.to_dict(), "\n")
print("Whitespace Opportunities (market-wide underperformance):")
print(whitespace_features if whitespace_features else "None", "\n")

# --- 5) Bar chart: Your feature gaps (top + bottom, combined) ---
sel = pd.concat([top_strengths, top_weaknesses]).sort_values()
plt.figure(figsize=(7,4))
plt.bar(sel.index, sel.values)
plt.title("Our Differentiation Gaps vs Market (Normalized)")
plt.ylabel("Gap (Our - Market Mean)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# --- 6) 2D Positioning Map (Capability vs Price) ---
plt.figure(figsize=(6,5))
x = position_df["capability_score"].values
y = position_df["price_score_inv"].values
plt.scatter(x, y)
for i, nm in enumerate(position_df["name"].values):
    plt.annotate(nm, (x[i]+0.002, y[i]+0.002), fontsize=9)
plt.xlabel("Capability (Weighted, Normalized)")
plt.ylabel("Price (Lower → Higher Score)")
plt.title("Positioning Map: Capability vs Price")
plt.tight_layout()
plt.show()

# --- 7) Optional artifact prints ---
print("=== Positioning Table (head) ===")
print(position_df.sort_values("capability_score", ascending=False).head(10).to_string(index=False))



# In[ ]:




