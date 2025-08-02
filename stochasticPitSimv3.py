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

# load 2024 Hungarian GP data for tire deg modeling
session = fastf1.get_session(2024, 'Hungarian Grand Prix', 'R')
session.load()

# load 2025 qualifying results for grid positions
quali = fastf1.get_session(2025, 'Hungarian Grand Prix', 'Q')
quali.load()

# process race data
laps = session.laps
stints = laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime"]].copy()
stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
stints.dropna(subset=["LapTime_s"], inplace=True)
stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1

# process qualifying results for grid analysis
quali_results = quali.results[['Abbreviation', 'Position']].copy()
quali_results = quali_results.dropna()
quali_results['GridPosition'] = quali_results['Position'].astype(int)

print("Grid positions from 2025 Hungarian GP qualifying:")
print(quali_results[['Abbreviation', 'GridPosition']].head(10))

# tire deg model
def build_tire_model(compound_data):
    if len(compound_data) < 5:
        return None
    
    x = compound_data["StintLap"].values
    y = compound_data["LapTime_s"].values

    def model(x, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(77, 5))  
        beta = numpyro.sample("beta", dist.Normal(0.04, 0.02))  
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
        mu = alpha + beta * x
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), x, y)
    return mcmc

# build models with compound-specific deg
compound_models = {}
for compound in stints["Compound"].unique():
    data = stints[stints["Compound"] == compound]
    if len(data) > 5:
        mcmc = build_tire_model(data)
        compound_models[compound] = mcmc

# tire performance 
def get_tire_performance(compound, lap_in_stint, base_pace=77, weather='dry', track_evolution=0):
    """Calculate lap time based on compound, stint lap, weather, and track evolution"""
    
    if weather == 'dry':
        # dry compound characteristics 
        compound_offsets = {
            'SOFT': -0.6,    
            'MEDIUM': 0.0,   
            'HARD': +0.4     
        }
        
        deg_rates = {
            'SOFT': 0.15,    
            'MEDIUM': 0.08,  
            'HARD': 0.04     
        }
        
        base_time = base_pace + compound_offsets.get(compound, 0)
        degradation = deg_rates.get(compound, 0.08) * lap_in_stint
        
    else:  # wet weather
        compound_offsets = {
            'INTERMEDIATE': 0.0,  # baseline for wet
            'WET': +2.0          
        }
        
        deg_rates = {
            'INTERMEDIATE': 0.02,  
            'WET': 0.01           
        }
        
        # base wet pace 
        base_time = base_pace + 8 + compound_offsets.get(compound, 0)
        degradation = deg_rates.get(compound, 0.02) * lap_in_stint
    
    # track evolution (negative = getting faster)
    base_time += track_evolution
    
    # non-linear deg for long stints
    if lap_in_stint > 25:
        degradation += 0.03 * (lap_in_stint - 25) ** 1.3
    
    return base_time + degradation

# weather simulation 
def generate_weather_conditions(num_laps, rain_probability=0.5):
    """Generate weather conditions throughout the race"""
    weather_laps = ['dry'] * num_laps
    
    if np.random.rand() < rain_probability:
        # rain scenarios
        rain_scenarios = [
            'early_shower',    # rain laps 5-15
            'mid_race_rain',   # rain laps 20-35
            'late_drama',      # rain laps 40-50
            'intermittent'     # multiple short showers
        ]
        
        scenario = np.random.choice(rain_scenarios)
        
        if scenario == 'early_shower':
            rain_start = np.random.choice(range(3, 8))
            rain_end = rain_start + np.random.choice(range(8, 15))
            for lap in range(rain_start, min(rain_end, num_laps)):
                weather_laps[lap] = 'wet'
                
        elif scenario == 'mid_race_rain':
            rain_start = np.random.choice(range(18, 25))
            rain_end = rain_start + np.random.choice(range(12, 20))
            for lap in range(rain_start, min(rain_end, num_laps)):
                weather_laps[lap] = 'wet'
                
        elif scenario == 'late_drama':
            rain_start = np.random.choice(range(35, 45))
            for lap in range(rain_start, num_laps):
                weather_laps[lap] = 'wet'
                
        elif scenario == 'intermittent':
            # 2-3 short showers
            num_showers = np.random.choice([2, 3])
            for _ in range(num_showers):
                shower_start = np.random.choice(range(5, num_laps-8))
                shower_length = np.random.choice(range(3, 8))
                for lap in range(shower_start, min(shower_start + shower_length, num_laps)):
                    weather_laps[lap] = 'wet'
    
    return weather_laps

# safety car simulation 
def generate_sc_laps(num_laps, weather_conditions):
    """Generate safety car periods based on track and weather"""
    sc_laps = set()
    
    # base SC probability at Hungary
    base_sc_prob = 0.15
    
    # weather-influenced SC probability
    wet_laps = sum(1 for w in weather_conditions if w == 'wet')
    weather_sc_prob = min(0.4, wet_laps / num_laps * 0.8)
    
    total_sc_prob = min(0.6, base_sc_prob + weather_sc_prob)
    
    if np.random.rand() < total_sc_prob:
        # determine SC cause and timing
        if wet_laps > 0:
            # weather-related incident
            wet_lap_indices = [i for i, w in enumerate(weather_conditions) if w == 'wet']
            if wet_lap_indices:
                sc_start = np.random.choice(wet_lap_indices[:len(wet_lap_indices)//2])
                sc_duration = np.random.choice(range(3, 6))
        else:
            # regular incident
            sc_start = np.random.choice(range(8, num_laps-5))
            sc_duration = np.random.choice(range(2, 5))
        
        sc_laps.update(range(sc_start, min(sc_start + sc_duration, num_laps)))
    
    return sc_laps

# position-based performance model
def get_position_penalty(grid_pos, current_lap, total_laps):
    """Calculate time penalty based on starting grid position and traffic"""
    
    base_penalty = {
        1: 0.0, 2: 0.1, 3: 0.15, 4: 0.2, 5: 0.25,
        6: 0.3, 7: 0.35, 8: 0.4, 9: 0.45, 10: 0.5
    }
    
    # traffic penalty reduces over time as field spreads
    traffic_factor = max(0.3, 1.0 - (current_lap / total_laps) * 0.7)
    
    penalty = base_penalty.get(min(grid_pos, 10), 0.5 + (grid_pos - 10) * 0.05)
    
    return penalty * traffic_factor

# race simulation 
def simulate_race(strategy, grid_position, compound_models, num_laps=70, 
                 base_pace=77, rain_probability=0.5, car_performance_factor=1.0):
    """Simulate a complete race with weather and position effects"""
    
    race_time = 0
    current_lap = 1
    pit_time_loss = 20  
    current_position = grid_position
    fuel_load = 110  # starting fuel load in kg
    fuel_consumption_rate = 1.6  # kg per lap
    
    # generate race conditions
    weather_conditions = generate_weather_conditions(num_laps, rain_probability)
    sc_laps = generate_sc_laps(num_laps, weather_conditions)
    
    # track evolution (gets faster over time in dry conditions)
    track_evolution_rate = -0.002  
    
    # VSC simulation 
    vsc_laps = set()
    if np.random.rand() < 0.25:  
        vsc_start = np.random.choice(range(10, num_laps-5))
        vsc_laps.update(range(vsc_start, vsc_start + 2))
    
    for stint_idx, stint in enumerate(strategy):
        comp = stint["compound"]
        stint_len = stint["laps"]
        
        # adjust stint length if it would exceed race distance
        remaining_laps = num_laps - current_lap + 1
        stint_len = min(stint_len, remaining_laps)
        
        for lap_in_stint in range(1, stint_len + 1):
            if current_lap > num_laps:
                break
            
            # current weather
            current_weather = weather_conditions[current_lap - 1]
            
            # track evolution
            track_evolution = track_evolution_rate * current_lap if current_weather == 'dry' else 0
            
            # base lap time from tire model
            lap_time = get_tire_performance(comp, lap_in_stint, base_pace, 
                                          current_weather, track_evolution)
            
            # car performance factor 
            lap_time *= car_performance_factor
            
            # fuel effect (car gets lighter and faster)
            fuel_effect = -(fuel_load * 0.035)  
            lap_time += fuel_effect
            fuel_load = max(5, fuel_load - fuel_consumption_rate)  # min 5kg reserve
            
            # position-based penalty (traffic, dirty air)
            position_penalty = get_position_penalty(current_position, current_lap, num_laps)
            lap_time += position_penalty
            
            # DRS effect for following cars (only in dry conditions)
            if current_weather == 'dry' and current_position > 1:
                # probability of being within DRS range
                drs_probability = max(0.1, 0.4 - (current_position * 0.02))
                if np.random.rand() < drs_probability:
                    lap_time -= 0.3  # DRS time gain
            
            # tire temperature effects
            if current_weather == 'dry':
                
                if comp == 'SOFT' and lap_in_stint > 10:
                    lap_time += min(0.5, (lap_in_stint - 10) * 0.02)  # overheating
                elif comp == 'HARD' and lap_in_stint < 5:
                    lap_time += max(0, (5 - lap_in_stint) * 0.1)  # cold tires
            
            # weather transition penalties
            if current_lap > 1:
                prev_weather = weather_conditions[current_lap - 2]
                if prev_weather != current_weather:
                    # weather change penalty
                    if current_weather == 'wet' and comp in ['SOFT', 'MEDIUM', 'HARD']:
                        lap_time += np.random.uniform(3.0, 8.0)  
                    elif current_weather == 'dry' and comp in ['INTERMEDIATE', 'WET']:
                        lap_time += np.random.uniform(1.5, 3.0)  
            
            # driver error probability (increases with stint length and weather)
            error_probability = 0.01 + (lap_in_stint * 0.001)
            if current_weather == 'wet':
                error_probability *= 2.5
            
            if np.random.rand() < error_probability:
                lap_time += np.random.uniform(1.0, 3.0)  
            
            # random variation (car setup, track conditions)
            lap_time += np.random.normal(0, 0.3)
            
            # safety car effect
            if current_lap in sc_laps:
                lap_time *= 1.3  # SC factor
            elif current_lap in vsc_laps:
                lap_time *= 1.15  # VSC factor
            
            race_time += lap_time
            current_lap += 1
        
        # pit stop execution
        if stint_idx < len(strategy) - 1:
            # pit stop time variability
            pit_execution = np.random.normal(pit_time_loss, 1.5)  
            
            # reduced pit loss during SC/VSC
            if any(lap in sc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                pit_execution *= 0.25  # SC
            elif any(lap in vsc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                pit_execution *= 0.6   # VSC
            
            # tire change complexity
            prev_compound = strategy[stint_idx]["compound"]
            next_compound = strategy[stint_idx + 1]["compound"]
            if prev_compound != next_compound:
                pit_execution += 0.5 
            
            race_time += max(15, pit_execution)  
            
            # strategic position changes
            strategy_aggressiveness = len([s for s in strategy if s["compound"] in ["SOFT", "INTERMEDIATE"]])
            undercut_probability = 0.3 if stint_idx == 0 else 0.15
            
            position_change = 0
            if np.random.rand() < undercut_probability:
                position_change = np.random.choice([-2, -1], p=[0.3, 0.7])  # undercut gain
            else:
                position_change = np.random.choice([-1, 0, 1, 2], p=[0.15, 0.4, 0.35, 0.1])
            
            if strategy_aggressiveness >= 2:  
                position_change += np.random.choice([-1, 0], p=[0.4, 0.6])
            
            current_position = max(1, min(20, current_position + position_change))
    
    return race_time, current_position

# tire strategies
dry_strategies = {
    "1-stop (M-H)": [
        {"compound": "MEDIUM", "laps": 35},
        {"compound": "HARD", "laps": 35}
    ],
    "2-stop (S-M-H)": [
        {"compound": "SOFT", "laps": 20},
        {"compound": "MEDIUM", "laps": 25},
        {"compound": "HARD", "laps": 25}
    ],
    "2-stop (M-M-H)": [
        {"compound": "MEDIUM", "laps": 23},
        {"compound": "MEDIUM", "laps": 23},
        {"compound": "HARD", "laps": 24}
    ],
    "1-stop (H-M)": [
        {"compound": "HARD", "laps": 40},
        {"compound": "MEDIUM", "laps": 30}
    ],
    "3-stop Aggressive": [
        {"compound": "SOFT", "laps": 15},
        {"compound": "MEDIUM", "laps": 18},
        {"compound": "MEDIUM", "laps": 18},
        {"compound": "SOFT", "laps": 19}
    ]
}

wet_strategies = {
    "Wet Conservative": [
        {"compound": "INTERMEDIATE", "laps": 25},
        {"compound": "INTERMEDIATE", "laps": 25},
        {"compound": "MEDIUM", "laps": 20}
    ],
    "Wet Aggressive": [
        {"compound": "INTERMEDIATE", "laps": 15},
        {"compound": "WET", "laps": 20},
        {"compound": "INTERMEDIATE", "laps": 15},
        {"compound": "SOFT", "laps": 20}
    ],
    "Gamble on Dry": [
        {"compound": "MEDIUM", "laps": 35},
        {"compound": "HARD", "laps": 35}
    ]
}

# combine strategies
all_strategies = {**dry_strategies, **wet_strategies}

# points system
def get_f1_points(position):
    """Return F1 championship points for finishing position"""
    points_map = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    return points_map.get(position, 0)

# Monte Carlo simulation with car performance differentiation
def run_monte_carlo_with_grid(strategies, compound_models, grid_positions, num_sims=500):
    """Run simulation across different grid positions with car performance factors"""
    results = {}
    
    # car performance factors based on 2025 grid positions (approximation)
    car_performance_map = {
        1: 0.98,   
        2: 0.985,  
        3: 0.99,   
        4: 0.995,
        5: 1.00,   
        6: 1.005,
        7: 1.01,
        8: 1.015,  
        9: 1.02,
        10: 1.025,
        11: 1.03,  
        12: 1.035,
        13: 1.04,
        14: 1.045,
        15: 1.05,  
    }
    
    for grid_pos in grid_positions:
        print(f"\nSimulating from grid position {grid_pos}...")
        pos_results = {name: {'times': [], 'final_positions': [], 'points': []} for name in strategies.keys()}
        
        car_factor = car_performance_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
        
        for _ in trange(num_sims, desc=f"P{grid_pos}"):
            # vary base pace and weather probability
            base_pace = np.random.normal(77, 0.6)
            sim_rain_prob = np.random.uniform(0.4, 0.6)  # 40-60% rain chance
            
            for name, strat in strategies.items():
                race_time, final_pos = simulate_race(
                    strat, grid_pos, compound_models, 
                    base_pace=base_pace, rain_probability=sim_rain_prob,
                    car_performance_factor=car_factor
                )
                
                # calculate points 
                points = get_f1_points(final_pos)
                
                pos_results[name]['times'].append(race_time)
                pos_results[name]['final_positions'].append(final_pos)
                pos_results[name]['points'].append(points)
        
        results[grid_pos] = pos_results
    
    return results

# visualization
def plot_strategy_analysis(results, grid_positions):
    """Create comprehensive strategy analysis plots"""
    
    # Figure 1: Race time distributions
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle('Hungarian GP Strategy Analysis - Race Time Distributions (50% Rain Probability)', fontsize=14)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_strategies)))
    strategy_colors = dict(zip(all_strategies.keys(), colors))
    
    for i, grid_pos in enumerate([1, 5, 10]):
        ax = axes1[i]
        for strategy_name in all_strategies.keys():
            times = results[grid_pos][strategy_name]['times']
            ax.hist(times, bins=25, alpha=0.6, label=strategy_name, 
                   color=strategy_colors[strategy_name])
        
        ax.set_title(f'Race Time Distribution - Grid P{grid_pos}')
        ax.set_xlabel('Race Time (s)')
        ax.set_ylabel('Frequency')
        if i == 2:  
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Final position distributions
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle('Hungarian GP Strategy Analysis - Final Position Distributions (50% Rain Probability)', fontsize=14)
    
    for i, grid_pos in enumerate([1, 5, 10]):
        ax = axes2[i]
        
        final_pos_data = []
        strategy_names = []
        
        for strategy_name in all_strategies.keys():
            positions = results[grid_pos][strategy_name]['final_positions']
            final_pos_data.extend(positions)
            strategy_names.extend([strategy_name] * len(positions))
        
        df_pos = pd.DataFrame({'Strategy': strategy_names, 'Final_Position': final_pos_data})
        
        sns.boxplot(data=df_pos, x='Strategy', y='Final_Position', ax=ax)
        ax.set_title(f'Final Positions - Grid P{grid_pos}')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Final Position')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        ax.set_ylim(0, 20)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Strategy effectiveness heatmap
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    
    heatmap_data = []
    strategy_names = list(all_strategies.keys())
    
    for strategy in strategy_names:
        row = []
        for grid_pos in [1, 3, 5, 8, 10, 15]:
            avg_pos = np.mean(results[grid_pos][strategy]['final_positions'])
            row.append(avg_pos)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    
    im = ax3.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto')
    
    ax3.set_xticks(range(len([1, 3, 5, 8, 10, 15])))
    ax3.set_xticklabels([f'P{pos}' for pos in [1, 3, 5, 8, 10, 15]])
    ax3.set_yticks(range(len(strategy_names)))
    ax3.set_yticklabels(strategy_names)
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Average Final Position', rotation=270, labelpad=20)
    
    for i in range(len(strategy_names)):
        for j in range(len([1, 3, 5, 8, 10, 15])):
            text = ax3.text(j, i, f'{heatmap_array[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax3.set_title('Strategy Effectiveness by Grid Position\n(Lower numbers = better final position)', fontsize=14)
    ax3.set_xlabel('Starting Grid Position')
    ax3.set_ylabel('Strategy')
    
    plt.tight_layout()
    plt.show()

# run simulations for key grid positions
key_positions = [1, 3, 5, 8, 10, 15]
print("Running Monte Carlo simulation for Hungarian GP...")
print("Including 50% rain probability...")

results = run_monte_carlo_with_grid(all_strategies, compound_models, key_positions, num_sims=300)

# generate analysis
plot_strategy_analysis(results, key_positions)

# results summary
print("\n" + "="*80)
print("HUNGARIAN GP STRATEGY ANALYSIS SUMMARY")
print("="*80)

for grid_pos in key_positions:
    print(f"\n GRID POSITION {grid_pos}")
    print("-" * 50)
    
    strategy_summary = []
    for strategy_name, data in results[grid_pos].items():
        times = np.array(data['times'])
        positions = np.array(data['final_positions'])
        points = np.array(data['points'])
        
        strategy_summary.append({
            'Strategy': strategy_name,
            'Avg Time': f"{np.mean(times):.1f}s",
            'Avg Final Pos': f"{np.mean(positions):.1f}",
            'Avg Points': f"{np.mean(points):.1f}",
            'Points Finish %': f"{np.mean(points > 0)*100:.1f}%",
            'Top 5 Finish %': f"{np.mean(positions <= 5)*100:.1f}%",
            'Podium %': f"{np.mean(positions <= 3)*100:.1f}%" if grid_pos <= 10 else "N/A",
            'Win %': f"{np.mean(positions == 1)*100:.1f}%" if grid_pos <= 5 else "N/A"
        })
    
    summary_df = pd.DataFrame(strategy_summary)
    summary_df = summary_df.sort_values('Avg Points', ascending=False)
    print(summary_df.to_string(index=False))

print("\n" + "="*80)
