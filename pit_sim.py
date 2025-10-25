import fastf1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
import json

from tire_model import build_models, get_tire_perf

def load_params(gp_name):
    try:
        gp_clean = gp_name.lower().replace(' ', '_').replace('grand_prix', 'gp')
        fname = f"{gp_clean}_params"
        mod = __import__(fname)
        
        params = {
            'POS_PEN': mod.POS_PEN,
            'TIRE_PERF': mod.TIRE_PERF,
            'DRV_ERR': mod.DRV_ERR,
            'DRS': mod.DRS
        }
        return params, True
    except ImportError:
        return None, False

def gen_sc_laps(n_laps, circuit):
    sc_laps = set()
    
    sc_prob = circuit['sc_prob']
    
    if np.random.rand() < sc_prob:
        sc_start = np.random.choice(range(8, n_laps-5))
        sc_dur = np.random.choice(range(2, 5))
        
        sc_laps.update(range(sc_start, min(sc_start + sc_dur, n_laps)))
    
    return sc_laps

def get_pos_pen(grid_pos, curr_lap, total_laps, params=None):
    if params and 'POS_PEN' in params:
        pen_map = params['POS_PEN']
        if grid_pos in pen_map:
            base = pen_map[grid_pos]['penalty']
        else:
            base = 0.05 * (grid_pos - 1)
    else:
        penalty_map = {
            1: 0.0, 2: 0.08, 3: 0.14, 4: 0.18, 5: 0.22,
            6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.42
        }
        base = penalty_map.get(min(grid_pos, 10), 0.42 + (grid_pos - 10) * 0.035)
    
    traffic = max(0.4, 1.0 - (curr_lap / total_laps) * 0.6)
    
    return base * traffic

def sim_race(strat, grid_pos, models, circuit, params=None, car_perf=1.0):
    n_laps = circuit['laps']
    base_pace = circuit['base_pace']
    rain_prob = circuit['rain_prob']
    
    race_time = 0
    curr_lap = 1
    pit_loss = circuit['pit_loss']
    curr_pos = grid_pos
    fuel = 110
    fuel_rate = circuit['fuel_rate']
    
    weather = ['dry'] * n_laps
    sc_laps = gen_sc_laps(n_laps, circuit)
    
    track_evo_rate = -0.0025
    
    vsc_laps = set()
    if np.random.rand() < circuit['vsc_prob']:
        vsc_start = np.random.choice(range(10, n_laps-5))
        vsc_laps.update(range(vsc_start, vsc_start + 2))
    
    for stint_idx, stint in enumerate(strat):
        comp = stint["comp"]
        stint_len = stint["laps"]
        
        remaining = n_laps - curr_lap + 1
        stint_len = min(stint_len, remaining)
        
        for lap_in_stint in range(1, stint_len + 1):
            if curr_lap > n_laps:
                break
            
            curr_weather = 'dry'
            track_evo = track_evo_rate * curr_lap
            
            lap_time = get_tire_perf(
                comp, lap_in_stint, models, base_pace, 
                curr_weather, track_evo, params
            )
            
            lap_time *= car_perf
            
            fuel_effect = -(fuel * circuit['fuel_effect'])
            lap_time += fuel_effect
            fuel = max(5, fuel - fuel_rate)
            
            pos_penalty = get_pos_pen(curr_pos, curr_lap, n_laps, params)
            lap_time += pos_penalty
            
            if curr_weather == 'dry' and curr_pos > 1:
                if params and 'DRS' in params:
                    drs_prob = params['DRS'].get('prob', 0.38)
                    drs_mean = params['DRS'].get('median', 0.40)
                    drs_std = params['DRS'].get('std', 0.18)
                else:
                    drs_prob = 0.38
                    drs_mean = 0.40
                    drs_std = 0.18
                
                pos_adj_prob = drs_prob * max(0.3, 1.0 - (curr_pos * 0.05))
                
                if np.random.rand() < pos_adj_prob:
                    drs_adv = max(0.2, np.random.normal(drs_mean, drs_std))
                    lap_time -= drs_adv
            
            if curr_weather == 'dry':
                if comp == 'SOFT' and lap_in_stint > 8:
                    lap_time += min(0.6, (lap_in_stint - 8) * 0.025)
                elif comp == 'HARD' and lap_in_stint < 6:
                    lap_time += max(0, (6 - lap_in_stint) * 0.12)
            
            if params and 'DRV_ERR' in params:
                err_rate = params['DRV_ERR'].get('dry', {}).get('rate', 0.008)
            else:
                err_rate = 0.008
            
            err_prob = err_rate + (lap_in_stint * 0.0005)
            
            if np.random.rand() < err_prob:
                lap_time += np.random.uniform(1.2, 3.5)
            
            lap_time += np.random.normal(0, 0.35)
            
            if curr_lap in sc_laps:
                lap_time *= 1.35
            elif curr_lap in vsc_laps:
                lap_time *= 1.18
            
            race_time += lap_time
            curr_lap += 1
        
        if stint_idx < len(strat) - 1:
            pit_exec = np.random.normal(pit_loss, 1.2)
            
            if any(lap in sc_laps for lap in range(max(1, curr_lap-2), curr_lap+1)):
                pit_exec *= 0.20
            elif any(lap in vsc_laps for lap in range(max(1, curr_lap-2), curr_lap+1)):
                pit_exec *= 0.55
            
            prev_comp = strat[stint_idx]["comp"]
            next_comp = strat[stint_idx + 1]["comp"]
            if prev_comp != next_comp:
                pit_exec += 0.4
            
            race_time += max(13, pit_exec)
            
            strat_agg = len([s for s in strat if s["comp"] in ["SOFT"]])
            undercut_prob = 0.25 if stint_idx == 0 else 0.12
            
            pos_change = 0
            if np.random.rand() < undercut_prob:
                pos_change = np.random.choice([-2, -1], p=[0.25, 0.75])
            else:
                pos_change = np.random.choice([-1, 0, 1, 2], p=[0.10, 0.50, 0.30, 0.10])
            
            if strat_agg >= 2:
                pos_change += np.random.choice([-1, 0], p=[0.3, 0.7])
            
            curr_pos = max(1, min(20, curr_pos + pos_change))
    
    return race_time, curr_pos

def get_strats(circuit):
    dry = {
        "1-stop (M-H)": [
            {"comp": "MEDIUM", "laps": 18},
            {"comp": "HARD", "laps": 38}
        ],
        "1-stop (H-M)": [
            {"comp": "HARD", "laps": 38},
            {"comp": "MEDIUM", "laps": 18}
        ],
        "1-stop (H-S)": [
            {"comp": "HARD", "laps": 43},
            {"comp": "SOFT", "laps": 13}
        ],
        "1-stop (S-H)": [
            {"comp": "SOFT", "laps": 14},
            {"comp": "HARD", "laps": 42}
        ],
        "1-stop (M-S)": [
            {"comp": "MEDIUM", "laps": 33},
            {"comp": "SOFT", "laps": 23}
        ],
        "1-stop (S-M)": [
            {"comp": "SOFT", "laps": 24},
            {"comp": "MEDIUM", "laps": 32}
        ],
        "2-stop (M-H-S)": [
            {"comp": "MEDIUM", "laps": 16},
            {"comp": "HARD", "laps": 32},
            {"comp": "SOFT", "laps": 8}
        ],
        "2-stop (H-M-H)": [
            {"comp": "HARD", "laps": 12},
            {"comp": "MEDIUM", "laps": 16},
            {"comp": "HARD", "laps": 28}
        ],
        "2-stop (S-H-S)": [
            {"comp": "SOFT", "laps": 16},
            {"comp": "HARD", "laps": 32},
            {"comp": "SOFT", "laps": 8}
        ],
        "2-stop (S-H-M)": [
            {"comp": "SOFT", "laps": 16},
            {"comp": "HARD", "laps": 28},
            {"comp": "MEDIUM", "laps": 12}
        ],
         "2-stop (S-M-S)": [
            {"comp": "SOFT", "laps": 16},
            {"comp": "MEDIUM", "laps": 25},
            {"comp": "SOFT", "laps": 15}
        ]
    }
    
    return dry

def get_f1_pts(pos):
    pts = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    return pts.get(pos, 0)

def run_mc(strats, models, circuit, grid_pos, params=None, n_sims=500):
    results = {}
    
    car_perf_map = {
        1: 0.98,   2: 0.985,  3: 0.99,   4: 0.995,  5: 1.00,   
        6: 1.005,  7: 1.01,   8: 1.015,  9: 1.02,   10: 1.025,
        11: 1.03,  12: 1.035, 13: 1.04,  14: 1.045, 15: 1.05,  
    }
    
    for gp in grid_pos:
        print(f"\nSimulating from grid position {gp}...")
        pos_res = {name: {'times': [], 'final_pos': [], 'pts': []} 
                   for name in strats.keys()}
        
        car_factor = car_perf_map.get(gp, 1.0 + (gp - 15) * 0.005)
        
        for _ in trange(n_sims, desc=f"P{gp}"):
            base_pace = np.random.normal(circuit['base_pace'], 0.5)
            
            temp_circuit = circuit.copy()
            temp_circuit['base_pace'] = base_pace
            
            for name, strat in strats.items():
                race_time, final_pos = sim_race(
                    strat, gp, models, temp_circuit, params, car_factor
                )
                
                pts = get_f1_pts(final_pos)
                
                pos_res[name]['times'].append(race_time)
                pos_res[name]['final_pos'].append(final_pos)
                pos_res[name]['pts'].append(pts)
        
        results[gp] = pos_res
    
    return results

def plot_analysis(results, grid_pos, circuit, model_info):
    rain_prob = circuit['rain_prob']
    model_src = "FP1 + Sprint" if model_info else "Fallback"
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 7))
    fig1.suptitle(f'{circuit["gp_name"]} Strategy Analysis - Race Time\n'
                 f'({rain_prob:.0%} Rain | {model_src} Models)', fontsize=14)
    
    strats = list(results[grid_pos[0]].keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(strats)))
    strat_colors = dict(zip(strats, colors))
    
    for i, gp in enumerate([1, 3, 5]):
        if gp not in results:
            continue
        ax = axes1[i]
        for strat_name in strats:
            times = results[gp][strat_name]['times']
            ax.hist(times, bins=25, alpha=0.6, label=strat_name, 
                   color=strat_colors[strat_name])
        
        ax.set_title(f'Race Time Distribution - Grid P{gp}')
        ax.set_xlabel('Race Time (s)')
        ax.set_ylabel('Frequency')
        if i == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results, grid_pos, circuit, model_info):
    print("\n" + "="*80)
    print(f"{circuit['gp_name'].upper()} STRATEGY ANALYSIS SUMMARY")
    print("="*80)
    
    tire_src = "FP1 + Sprint Models" if model_info else "Fallback Models"
    print(f"Tire Models: {tire_src}")
    
    if model_info:
        print(f"Practice Data Sources:")
        sess_info = model_info.get('info', {})
        if sess_info.get('fp1'):
            print(f"  FP1: {sess_info.get('fp1_laps', 0)} laps")
        if sess_info.get('sprint'):
            print(f"  Sprint: {sess_info.get('sprint_laps', 0)} laps")
        
        print(f"Model Quality:")
        for comp, q_info in model_info.get('quality', {}).items():
            quality = q_info['quality']
            n = q_info['n']
            print(f"  {comp}: {quality} ({n} laps)")
    
    for gp in grid_pos:
        print(f"\nGRID POSITION {gp}")
        print("-" * 50)
        
        strat_sum = []
        for strat_name, data in results[gp].items():
            times = np.array(data['times'])
            pos = np.array(data['final_pos'])
            pts = np.array(data['pts'])
            
            strat_sum.append({
                'Strategy': strat_name,
                'Avg Time': f"{np.mean(times):.1f}s",
                'Avg Pos': f"{np.mean(pos):.1f}",
                'Avg Pts': f"{np.mean(pts):.1f}",
                'Pts %': f"{np.mean(pts > 0)*100:.1f}%",
                'Top 5 %': f"{np.mean(pos <= 5)*100:.1f}%",
                'Podium %': f"{np.mean(pos <= 3)*100:.1f}%" if gp <= 10 else "N/A",
                'Win %': f"{np.mean(pos == 1)*100:.1f}%" if gp <= 5 else "N/A"
            })
        
        df = pd.DataFrame(strat_sum)
        df = df.sort_values('Avg Pts', ascending=False)
        print(df.to_string(index=False))

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pit_sim.py <year> <gp_name>")
        print("Example: python pit_sim.py 2025 'United States'")
        return
    
    year = int(sys.argv[1])
    gp_key = sys.argv[2]
    
    with open('circuit_config.json', 'r') as f:
        circuits = json.load(f)
    
    if gp_key not in circuits:
        print(f"Circuit '{gp_key}' not found in config")
        return
    
    circuit = circuits[gp_key]
    gp_name = circuit['gp_name']
    
    print(f"\n{'='*80}")
    print(f"MONTE CARLO SIMULATION: {gp_name.upper()}")
    print(f"{'='*80}")
    
    params, params_loaded = load_params(gp_name)
    
    print("\nBuilding tire models from practice...")
    models, model_info = build_models(year, gp_name, circuit['base_pace'])
    
    if models:
        print(f"Using tire models from FP1/Sprint")
    else:
        print(f"Using fallback tire modeling")
    
    strats = get_strats(circuit)
    
    key_pos = [1, 3, 5, 8, 10, 15]
    print(f"\nSimulating grid positions: {key_pos}")
    print(f"Parameters: {circuit['rain_prob']:.0%} rain, "
          f"{circuit['sc_prob']:.0%} SC, {circuit['vsc_prob']:.0%} VSC")
    
    results = run_mc(strats, models, circuit, key_pos, params, n_sims=1500)
    
    plot_analysis(results, key_pos, circuit, model_info)
    print_summary(results, key_pos, circuit, model_info)

if __name__ == "__main__":
    main()
