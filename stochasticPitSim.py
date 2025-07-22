import fastf1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# load session data
year, gp, session_type = 2024, 'Belgian Grand Prix', 'R'
session = fastf1.get_session(year, gp, session_type)
session.load()

laps = session.laps
stints = laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime"]].copy()
stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
stints.dropna(subset=["LapTime_s"], inplace=True)
stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1

# tire degradation model 
def build_tire_model(compound_data):
    x = compound_data["StintLap"].values
    y = compound_data["LapTime_s"].values

    def model(x, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(80, 5))
        beta = numpyro.sample("beta", dist.Normal(0.03, 0.02)) 
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
        mu = alpha + beta * x
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), x, y)
    return mcmc

# build models with compound-specific degradation
compound_models = {}
degradation_rates = {
    'SOFT': 0.08,     
    'MEDIUM': 0.04,   
    'HARD': 0.02      
}

for compound in stints["Compound"].unique():
    data = stints[stints["Compound"] == compound]
    if len(data) > 5:  # only build model if enough data
        mcmc = build_tire_model(data)
        compound_models[compound] = mcmc

# tire degradation function
def get_tire_performance(compound, lap_in_stint, base_pace=80):
    """Calculate lap time based on compound and stint lap with realistic degradation"""
    
    # base pace differences between compounds
    compound_offsets = {
        'SOFT': -0.8,    # softs are fastest initially
        'MEDIUM': 0.0,   # baseline
        'HARD': +0.5     # hards are slowest initially but degrade less
    }
    
    # degradation rates (seconds per lap)
    deg_rates = {
        'SOFT': 0.12,    # heavy degradation
        'MEDIUM': 0.06,  # moderate degradation
        'HARD': 0.03     # light degradation
    }
    
    # non-linear degradation
    base_time = base_pace + compound_offsets.get(compound, 0)
    degradation = deg_rates.get(compound, 0.05) * lap_in_stint
    
    # exponential component for extreme stint lengths
    if lap_in_stint > 20:
        degradation += 0.02 * (lap_in_stint - 20) ** 1.5
    
    return base_time + degradation

# SC simulation
def generate_sc_laps(num_laps):
    sc_laps = set()
    
    # higher probability of SC at Spa due to weather and crashes
    if np.random.rand() < 0.3:  # 30% chance of early SC
        sc_start = np.random.choice(range(1, 8))
        sc_laps.update(range(sc_start, sc_start + 3))
    
    if np.random.rand() < 0.25:  # 25% chance of mid-race SC
        sc_start = np.random.choice(range(15, 35))
        sc_laps.update(range(sc_start, sc_start + 4))
    
    if np.random.rand() < 0.15:  # 15% chance of late SC
        sc_start = np.random.choice(range(35, num_laps-3))
        sc_laps.update(range(sc_start, sc_start + 2))
    
    return sc_laps

# race simulation
def simulate_race(strategy, compound_models, num_laps=44, base_pace=80):
    race_time = 0
    current_lap = 1
    pit_time_loss = 22  
    sc_laps = generate_sc_laps(num_laps)
    
    for stint_idx, stint in enumerate(strategy):
        comp = stint["compound"]
        stint_len = stint["laps"]
        
        # adjust stint length if it would exceed race distance
        remaining_laps = num_laps - current_lap + 1
        stint_len = min(stint_len, remaining_laps)
        
        for lap_in_stint in range(1, stint_len + 1):
            if current_lap > num_laps:
                break
                
            # get base lap time from tire model
            lap_time = get_tire_performance(comp, lap_in_stint, base_pace)
            
            # add random variation
            lap_time += np.random.normal(0, 0.3)
            
            # safety car effect
            if current_lap in sc_laps:
                lap_time *= 1.4  # Larger SC time penalty
            
            # traffic effect
            if stint_idx == 0 and lap_in_stint > 15:  # Long first stint
                traffic_penalty = min(0.5, (lap_in_stint - 15) * 0.05)
                lap_time += traffic_penalty
            
            race_time += lap_time
            current_lap += 1
        
        # add pit stop time (except for last stint)
        if stint_idx < len(strategy) - 1:
            # reduced pit loss during SC
            if any(lap in sc_laps for lap in range(max(1, current_lap-3), current_lap+1)):
                race_time += pit_time_loss * 0.4  
            else:
                race_time += pit_time_loss
    
    return race_time

# Monte Carlo simulation
def run_monte_carlo(strategies, compound_models, num_sims=1000):
    results = {name: [] for name in strategies.keys()}
    for _ in trange(num_sims):
        # vary base pace slightly between simulations
        base_pace = np.random.normal(80, 1)
        for name, strat in strategies.items():
            t = simulate_race(strat, compound_models, base_pace=base_pace)
            results[name].append(t)
    return results

# strategy definitions
strategies = {
    "1-stop (M-H)": [
        {"compound": "MEDIUM", "laps": 20},  
        {"compound": "HARD", "laps": 24}
    ],
    "2-stop (M-M-H)": [
        {"compound": "MEDIUM", "laps": 15},
        {"compound": "MEDIUM", "laps": 15}, 
        {"compound": "HARD", "laps": 14}
    ],
    "2-stop (S-M-H)": [  
        {"compound": "SOFT", "laps": 12},
        {"compound": "MEDIUM", "laps": 16},
        {"compound": "HARD", "laps": 16}
    ],
    "1-stop (H-M)": [
        {"compound": "HARD", "laps": 25},   
        {"compound": "MEDIUM", "laps": 19}
    ],
    "2-stop (M-H-M)": [
        {"compound": "MEDIUM", "laps": 12},
        {"compound": "HARD", "laps": 18},
        {"compound": "MEDIUM", "laps": 14}
    ]
}

# simulation
print("Running Monte Carlo simulation...")
results = run_monte_carlo(strategies, compound_models, num_sims=1000)

# visualization
plt.figure(figsize=(14, 8))

colors = {
    "1-stop (M-H)": "steelblue",
    "2-stop (M-M-H)": "sandybrown", 
    "2-stop (S-M-H)": "crimson",
    "1-stop (H-M)": "indianred",
    "2-stop (M-H-M)": "seagreen"
}

for name, times in results.items():
    times = np.array(times)
    plt.hist(times, bins=40, alpha=0.4, label=name, color=colors.get(name))
    
    median = np.median(times)
    p10 = np.percentile(times, 10)
    p90 = np.percentile(times, 90)
    
    plt.axvline(median, color=colors.get(name, 'gray'), linestyle='--', linewidth=2)
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]*0.8], p10, p90, 
                      color=colors.get(name, 'gray'), alpha=0.1)

plt.xlabel("Race Time (s)")
plt.ylabel("Frequency")
plt.title("F1 Pit Strategy Simulation - Belgian Grand Prix")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# strategy comparison analysis
print("\n" + "="*60)
print("STRATEGY PERFORMANCE SUMMARY")
print("="*60)

summary_data = []
for name, times in results.items():
    times = np.array(times)
    median = np.median(times)
    mean = np.mean(times)
    std = np.std(times)
    p5 = np.percentile(times, 5)
    p95 = np.percentile(times, 95)
    
    summary_data.append({
        "Strategy": name,
        "Median (s)": round(median, 1),
        "Mean (s)": round(mean, 1),
        "Std Dev": round(std, 1),
        "5th %ile": round(p5, 1),
        "95th %ile": round(p95, 1),
        "Range": round(p95 - p5, 1)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values("Median (s)")
print(summary_df.to_string(index=False))

# head-to-head probabilities
print("\n" + "="*50)
print("HEAD-TO-HEAD WIN PROBABILITIES")
print("="*50)

strategy_names = list(results.keys())
for i, strat1 in enumerate(strategy_names):
    for j, strat2 in enumerate(strategy_names[i+1:], i+1):
        times1 = np.array(results[strat1])
        times2 = np.array(results[strat2])
        prob = np.mean(times1 < times2)
        print(f"{strat1} beats {strat2}: {prob:.1%}")

# risk analysis
print("\n" + "="*40)
print("RISK ANALYSIS")
print("="*40)

best_median = min(summary_df["Median (s)"])
for _, row in summary_df.iterrows():
    time_penalty = row["Median (s)"] - best_median
    risk = row["Range"] / 2  
    print(f"{row['Strategy']:15} | Time Penalty: +{time_penalty:4.1f}s | Risk (±): {risk:4.1f}s")