"""
run_simulator_example.py
- Tries to import the repository's dwave_classical_portfolio class and use it.
- If unavailable, builds a small local BQM (same modelling idea) and solves it with neal or dimod sampler.
- Prints top feasible portfolios (within budget) and a summary.
"""

import sys
import numpy as np
import pandas as pd
import dimod

# try sampler imports (neal preferred, fallback to dimod reference)
try:
    import neal
    Sampler = neal.SimulatedAnnealingSampler
    sampler_name = "neal.SimulatedAnnealingSampler"
except Exception:
    try:
        from dwave.samplers import SimulatedAnnealingSampler
        Sampler = SimulatedAnnealingSampler
        sampler_name = "dwave.samplers.SimulatedAnnealingSampler"
    except Exception:
        from dimod.reference.samplers import SimulatedAnnealingSampler
        Sampler = SimulatedAnnealingSampler
        sampler_name = "dimod.reference.samplers.SimulatedAnnealingSampler"

print(f"Using sampler: {sampler_name}")

# ---------------------------------------
# Replace these example inputs as needed
# ---------------------------------------
stocks = ["ADANIPORTS.NS", "AXISBANK.NS", "BHARTIARTL.NS", "COALINDIA.NS"]
prices = np.array([300.0, 900.0, 650.0, 120.0])     # example per-share price
exp_returns = np.array([0.07, 0.10, 0.09, 0.06])    # expected returns
cov = np.array([
    [0.04, 0.01, 0.005, 0.002],
    [0.01, 0.05, 0.007, 0.003],
    [0.005, 0.007, 0.045, 0.002],
    [0.002, 0.003, 0.002, 0.03]
])
budget = 20000
risk_factor = 1.0

# ---------------------------------------
# Try to use repo's class if available
# ---------------------------------------
try:
    # adjust import path if repo uses different module name
    from portfolio import dwave_classical_portfolio
    print("Found dwave_classical_portfolio in repo — using it.")
    portfolio = dwave_classical_portfolio(stocks, risk_factor, budget)
    result = portfolio.portfolio_dwave(simulate_classical=True)  # if repo supports simulate flag
    print("Result from repo class:")
    print(result)
    sys.exit(0)
except Exception as e:
    print("Repo class not used (not found or not compatible). Falling back to local simulator.")
    # print(e)  # uncomment for debug

# ---------------------------------------
# Build local BQM (0/1 per stock)
# ---------------------------------------
n = len(stocks)

# Linear: negative returns (we minimize)
linear = {i: -float(exp_returns[i]) for i in range(n)}

# Quadratic: risk term
quad = {}
for i in range(n):
    for j in range(i, n):
        val = float(risk_factor * cov[i, j])
        if i == j:
            linear[i] = linear.get(i, 0.0) + val
        else:
            quad[(i, j)] = quad.get((i, j), 0.0) + val

# Penalty for budget constraint: P * (sum p_i x_i - budget)^2
# Expand: sum p_i^2 x_i + 2 sum_{i<j} p_i p_j x_i x_j - 2 budget sum p_i x_i + budget^2
P = 1e-3 * 1e6  # tuned scale (adjust as needed)
for i in range(n):
    linear[i] = linear.get(i, 0.0) + P * (prices[i] ** 2) - P * 2 * budget * prices[i]
for i in range(n):
    for j in range(i + 1, n):
        quad[(i, j)] = quad.get((i, j), 0.0) + P * 2 * prices[i] * prices[j]

bqm = dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype=dimod.BINARY)

# ---------------------------------------
# Sample with chosen sampler
# ---------------------------------------
sampler = Sampler()
sampleset = sampler.sample(bqm, num_reads=1000)

# Aggregate unique and evaluate
agg = sampleset.aggregate()
rows = []
for rec in agg.record:
    sample_values = rec[0]   # array like
    energy = float(rec[1])
    sample = {i: int(sample_values[idx]) for idx, i in enumerate(agg.variables)}
    x = np.array([sample[i] for i in range(n)])
    total_price = float(np.dot(prices, x))
    exp_ret = float(np.dot(exp_returns, x))
    risk = float(x @ cov @ x)
    rows.append({
        "x": x,
        "picks": [stocks[i] for i in range(n) if x[i] == 1],
        "price": total_price,
        "exp_return": exp_ret,
        "risk": risk,
        "energy": energy
    })

df = pd.DataFrame(rows)
df["feasible"] = df["price"] <= budget
df = df.sort_values(["feasible", "energy"], ascending=[False, True]).reset_index(drop=True)

print("\nTop feasible solutions:")
feasible = df[df["feasible"]].head(10)
if feasible.empty:
    print("No feasible solutions found. Try increasing penalty P or adjusting inputs.")
else:
    for idx, r in feasible.iterrows():
        print(f"Solution #{idx+1}: picks={r['picks']}, price={r['price']:.2f}, exp_return={r['exp_return']:.4f}, risk={r['risk']:.4f}, energy={r['energy']:.4f}")

print("\nTop overall by energy (may be infeasible):")
for i, r in df.head(5).iterrows():
    print(f"{i+1}: picks={r['picks']}, price={r['price']:.2f}, feasible={r['feasible']}, energy={r['energy']:.4f}")

# done

