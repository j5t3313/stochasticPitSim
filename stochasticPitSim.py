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
year, gp, session_type = 2025, 'Austrian Grand Prix', 'R'
session = fastf1.get_session(year, gp, session_type)
session.load()

laps = session.laps
stints = laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime"]].copy()
stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
stints.dropna(subset=["LapTime_s"], inplace=True)
stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1

# build deg models
def build_tire_model(compound_data):
    x = compound_data["StintLap"].values
    y = compound_data["LapTime_s"].values

    def model(x, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(80, 5))
        beta = numpyro.sample("beta", dist.Normal(0, 0.1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
        mu = alpha + beta * x
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), x, y)
    return mcmc

compound_models = {}
for compound in stints["Compound"].unique():
    data = stints[stints["Compound"] == compound]
    mcmc = build_tire_model(data)
    compound_models[compound] = mcmc

# generate sc sim
def generate_sc_laps(num_laps):
    sc_laps = set()
    if np.random.rand() < 0.4:
        sc_laps.add(1)
    if np.random.rand() < 0.2:
        sc_laps.update(np.random.choice(range(18, 32), size=3, replace=False))
    if np.random.rand() < 0.2:
        sc_laps.update(np.random.choice(range(38, 52), size=3, replace=False))
    return sc_laps

# race sim
def simulate_race(strategy, compound_models, num_laps=71):
    race_time = 0
    current_lap = 1
    pit_time = 25
    sc_laps = generate_sc_laps(num_laps)

    for stint in strategy:
        comp = stint["compound"]
        stint_len = stint["laps"]
        model = compound_models[comp]
        samples = model.get_samples()

        alpha = np.random.choice(samples["alpha"])
        beta = np.random.choice(samples["beta"])
        sigma = np.random.choice(samples["sigma"])

        for lap_in_stint in range(1, stint_len + 1):
            if current_lap > num_laps:
                break
            base_time = alpha + beta * lap_in_stint
            noise = np.random.normal(0, sigma)
            lap_time = base_time + noise
            if current_lap in sc_laps:
                lap_time *= 1.3
            race_time += lap_time
            current_lap += 1

        # adjust pit time if SC occurs in pit window
        if any(lap in sc_laps for lap in range(current_lap - stint_len, current_lap)):
            race_time += pit_time * 0.7  # SC pit loss reduced
        else:
            race_time += pit_time

    return race_time

# monte carlo
def run_monte_carlo(strategies, compound_models, num_sims=1000):
    results = {name: [] for name in strategies.keys()}
    for _ in trange(num_sims):
        for name, strat in strategies.items():
            t = simulate_race(strat, compound_models)
            results[name].append(t)
    return results

# define strats
strategies = {
    "1-stop (M-H)": [
        {"compound": "MEDIUM", "laps": 30},
        {"compound": "HARD", "laps": 41}
    ],
    "2-stop (M-M-H)": [
        {"compound": "MEDIUM", "laps": 20},
        {"compound": "MEDIUM", "laps": 25},
        {"compound": "HARD", "laps": 26}
    ],
    "1-stop (M-H-M)": [
        {"compound": "MEDIUM", "laps": 30},
        {"compound": "HARD", "laps": 30},
        {"compound": "MEDIUM", "laps": 11}
    ],
    "1-stop (H-M)": [
        {"compound": "HARD", "laps": 40},
        {"compound": "MEDIUM", "laps": 31}
    ]
}

# run and visualize
results = run_monte_carlo(strategies, compound_models)

plt.figure(figsize=(10,6))

colors = {
    "1-stop (M-H)": "steelblue",
    "2-stop (M-M-H)": "sandybrown",
    "1-stop (M-H-M)": "seagreen",
    "1-stop (H-M)": "indianred"
}

for name, times in results.items():
    
    plt.hist(times, bins=50, alpha=0.4, label=name, color=colors.get(name, None))
    
    # calculate key stats
    median = np.median(times)
    p10 = np.percentile(times, 10)
    p90 = np.percentile(times, 90)
    
    # median line
    plt.axvline(median, color=colors.get(name, 'gray'), linestyle='--', linewidth=2)
    
    # 10-90 percentile band
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], p10, p90, color=colors.get(name, 'gray'), alpha=0.1)

plt.xlabel("Race Time (s)")
plt.ylabel("Frequency")
plt.title("Stochastic Pit Strategy Simulation\n(w/ Median Lines + 90% Confidence Bands)")
plt.legend()
plt.show()


# probabilities
mh_times = np.array(results["1-stop (M-H)"])
mmh_times = np.array(results["2-stop (M-M-H)"])
prob = np.mean(mh_times < mmh_times)
print(f"Probability 1-stop (M-H) beats 2-stop (M-M-H): {prob:.3f}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.figure(figsize=(12, 7))

for name, times in results.items():
    times = np.array(times)
   
    plt.hist(times, bins=50, alpha=0.3, density=True, label=f"{name} hist")
    sns.kdeplot(times, lw=2, label=f"{name} KDE")
    median = np.median(times)
    plt.axvline(median, linestyle='--', linewidth=2, label=f"{name} median")
    plt.text(median + 5, plt.ylim()[1]*0.02, f"{int(median)}s", rotation=90, va='bottom')

plt.xlabel("Race Time (s)")
plt.ylabel("Density")
plt.title("Stochastic Pit Strategy Simulation\n(w/ KDE + Median Annotations)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(12, 7))

for name, times in results.items():
    times = np.array(times)
    sorted_times = np.sort(times)
    cdf = np.arange(1, len(sorted_times)+1) / len(sorted_times)
    plt.plot(sorted_times, cdf, label=name)

plt.xlabel("Race Time (s)")
plt.ylabel("Cumulative Probability")
plt.title("Stochastic Pit Strategy Simulation CDF")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

summary_data = []
for name, times in results.items():
    times = np.array(times)
    median = np.median(times)
    p5 = np.percentile(times, 5)
    p95 = np.percentile(times, 95)
    summary_data.append({
        "Strategy": name,
        "Median (s)": round(median, 1),
        "5th %ile (s)": round(p5, 1),
        "95th %ile (s)": round(p95, 1)
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df)
