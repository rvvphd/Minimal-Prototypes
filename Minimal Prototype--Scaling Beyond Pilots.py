#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: Minimal Prototype--Scaling Beyond Pilots
# Author: Rod Villalobos using rapid prototyyping.
# Purpose: Show an AI prototype associated with scaling beyond pilots.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(101)

# -----------------------------
# 0) Config (tweak to your case)
# -----------------------------
DAYS              = 30
SLO_AVAIL_TARGET  = 0.995      # 99.5% monthly availability
LAT_P95_TARGET_MS = 600        # p95 latency SLO (ms)
ERR_BUDGET_WIN_D  = 30         # window for error budget
BASE_LAT_MS       = 250        # base p95 latency at low utilization
BASE_FAIL_RATE    = 0.001      # 0.1% baseline failure
ALPHA_FAIL_UP     = 0.020      # failure increase factor with utilization
BETA_LAT_UP       = 2.00       # latency multiplier with utilization
UTIL_TARGET       = 0.70       # planned headroom target for sizing

# Demand model (requests/sec average per day)
pilot_rps   = 8
day_growth  = 1.10             # 10% growth compounding
noise_pct   = 0.08             # day-to-day noise

# Capacity & cost model
inst_rps_peak     = 20         # max sustainable RPS per instance at 100% util
inst_cost_per_hr  = 0.18
min_instances     = 2
max_instances     = 80
scale_step        = 4          # scaler adds/removes in blocks

# -----------------------------
# 1) Generate demand per day
# -----------------------------
days = np.arange(DAYS)
demand_rps = pilot_rps * (day_growth ** days)
demand_rps *= (1 + np.random.normal(0, noise_pct, size=DAYS)).clip(0.8, 1.25)
# requests per day (approx): rps * 86,400
reqs_total = (demand_rps * 86400).astype(int)

# -----------------------------
# 2) Autoscaling policy (simple)
# -----------------------------
# Plan for UTIL_TARGET headroom: each instance should run at <= UTIL_TARGET
inst_rps_at_target = inst_rps_peak * UTIL_TARGET

def plan_instances(rps):
    raw = np.ceil(rps / inst_rps_at_target)
    # scaling in steps + clamp to bounds
    stepped = np.ceil(raw / scale_step) * scale_step
    return int(np.clip(stepped, min_instances, max_instances))

instances = np.array([plan_instances(r) for r in demand_rps])
capacity_rps = instances * inst_rps_peak           # absolute peak (100% util)
utilization  = np.minimum(1.0, demand_rps / (instances * inst_rps_at_target))

# -----------------------------
# 3) SLO modeling
# -----------------------------
# Latency inflation with utilization (convex curve)
p95_latency_ms = BASE_LAT_MS * (1 + BETA_LAT_UP * (utilization ** 2.0))
latency_ok = (p95_latency_ms <= LAT_P95_TARGET_MS)

# Failures increase with utilization (beyond baseline)
fail_rate = BASE_FAIL_RATE + ALPHA_FAIL_UP * (utilization ** 2.2)
fail_rate = np.clip(fail_rate, 0, 0.5)

# Availability approximation = 1 - effective error rate
# (includes soft drops if demand > capacity at UTIL_TARGET as extra error)
soft_drop_rate = np.maximum(0.0, demand_rps - instances * inst_rps_at_target) / np.maximum(demand_rps, 1e-9)
soft_drop_rate = np.clip(soft_drop_rate, 0, 1)
eff_error_rate = 1 - ((1 - fail_rate) * (1 - soft_drop_rate))
availability   = 1 - eff_error_rate

# -----------------------------
# 4) Error budget & costs
# -----------------------------
allowed_err = 1 - SLO_AVAIL_TARGET
daily_calls = reqs_total
daily_errors = (eff_error_rate * daily_calls).astype(int)

# Burn as fraction of budget per day (budget = allowed_err * total requests over window)
window_calls = daily_calls.sum()
budget_total = allowed_err * window_calls
budget_burn  = daily_errors / max(budget_total, 1.0)

# Cost model
daily_cost = instances * inst_cost_per_hr * 24
monthly_cost = daily_cost.sum()
cost_per_million = (monthly_cost / (window_calls / 1_000_000)) if window_calls > 0 else np.nan

# -----------------------------
# 5) Rollout gates (heuristics)
# -----------------------------
avail_mean = availability.mean()
latency_pass_rate = latency_ok.mean()                 # % days meeting p95 target
burn_total = budget_burn.sum()                        # fraction of monthly error budget used
headroom_ok_days = (utilization <= UTIL_TARGET).mean()

def rollout_reco(avail, lat_ok, burn, head_ok):
    if (avail >= SLO_AVAIL_TARGET and lat_ok >= 0.95 and burn <= 0.75 and head_ok >= 0.9):
        return "✅ Ready for GA: SLO met, budget healthy, capacity headroom sufficient."
    if (avail >= SLO_AVAIL_TARGET*0.995 and lat_ok >= 0.85):
        return "🟡 Limited GA: watch latency spikes and error-budget burn; increase capacity or optimize."
    return "🧪 Pilot only: add capacity, optimize hot paths, or reduce variance before broader rollout."

recommendation = rollout_reco(avail_mean, latency_pass_rate, burn_total, headroom_ok_days)

# -----------------------------
# 6) Summary prints
# -----------------------------
print("=== SCALING READINESS SUMMARY ===")
print({
    "avg_demand_rps": round(float(demand_rps.mean()), 2),
    "avg_utilization": round(float(utilization.mean()), 3),
    "avg_p95_latency_ms": round(float(p95_latency_ms.mean()), 1),
    "availability_mean": round(float(avail_mean), 5),
    "p95_latency_pass_rate": f"{round(100*float(latency_pass_rate),1)}%",
    "error_budget_used_%": f"{round(100*float(burn_total),1)}%",
    "capacity_headroom_ok_days_%": f"{round(100*float(headroom_ok_days),1)}%",
    "instances_min_max": (int(instances.min()), int(instances.max())),
    "monthly_cost_$": round(float(monthly_cost), 2),
    "cost_per_million_req_$": round(float(cost_per_million), 2),
})
print("\n=== RECOMMENDATION ===")
print(recommendation)

# -----------------------------
# 7) DataFrame head (optional)
# -----------------------------
df = pd.DataFrame({
    "day": days+1,
    "demand_rps": demand_rps,
    "instances": instances,
    "utilization": utilization,
    "p95_latency_ms": p95_latency_ms,
    "availability": availability,
    "latency_ok": latency_ok,
    "daily_calls": daily_calls,
    "daily_errors": daily_errors,
    "daily_cost_$": daily_cost
})
print("\n=== DAILY SNAPSHOT (head) ===")
print(df.head(5).round(3).to_string(index=False))

# -----------------------------
# 8) Plots
# -----------------------------
plt.figure(figsize=(8,4))
plt.plot(days+1, demand_rps, label="Demand (RPS)")
plt.plot(days+1, capacity_rps, label="Capacity (RPS @100%)")
plt.plot(days+1, instances*inst_rps_at_target, label=f"Planned Capacity (@{int(UTIL_TARGET*100)}% util)")
plt.title("Demand vs Capacity")
plt.xlabel("Day"); plt.ylabel("RPS"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
cum_burn = np.cumsum(budget_burn)
plt.plot(days+1, cum_burn, label="Cumulative Error Budget Burn (fraction)")
plt.axhline(1.0, ls="--", label="100% of Monthly Budget")
plt.title("Error Budget Burn (30-day window)")
plt.xlabel("Day"); plt.ylabel("Fraction of Budget"); plt.legend(); plt.tight_layout(); plt.show()


# In[ ]:




