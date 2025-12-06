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
            'POS_PEN': mod.POSITION_PENALTIES,
            'TIRE_PERF': mod.TIRE_PERFORMANCE,
            'DRV_ERR': mod.DRIVER_ERROR_RATES,
            'DRS': mod.DRS_EFFECTIVENESS
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

def sim_race(strat, grid_pos, models, circuit, params=None, car_perf=1.0, tire_adjustments=None):
    n_laps = circuit['laps']
    base_pace = circuit['base_pace']
    rain_prob = circuit['rain_prob']
    
    race_time = 0
    curr_lap = 1
    pit_loss = circuit['pit_loss']
    curr_pos = grid_pos
    fuel = 110
    fuel_rate = circuit['fuel_rate']
    
    if tire_adjustments is None:
        tire_adjustments = {}
    
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
            
            if comp in tire_adjustments:
                lap_time += tire_adjustments[comp]
            
            lap_time *= car_perf
            
            fuel_effect = -(fuel * circuit['fuel_effect'])
            lap_time += fuel_effect
            fuel = max(5, fuel - fuel_rate)
            
            pos_penalty = get_pos_pen(curr_pos, curr_lap, n_laps, params)
            lap_time += pos_penalty
            
            if curr_weather == 'dry' and curr_pos > 1:
                if params and 'DRS' in params:
                    drs_prob = params['DRS'].get('usage_probability', 0.38)
                    drs_mean = params['DRS'].get('median_advantage', 0.40)
                    drs_std = params['DRS'].get('std_advantage', 0.18)
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
                err_rate = params['DRV_ERR'].get('dry', {}).get('base_error_rate', 0.008)
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
    total_laps = circuit['laps']
    
    dry = {
        "1-stop (M-H)": [
            {"comp": "MEDIUM", "laps": int(total_laps * 0.32)},
            {"comp": "HARD", "laps": int(total_laps * 0.68)}
        ],
        "1-stop (H-M)": [
            {"comp": "HARD", "laps": int(total_laps * 0.68)},
            {"comp": "MEDIUM", "laps": int(total_laps * 0.32)}
        ],
        "1-stop (H-S)": [
            {"comp": "HARD", "laps": int(total_laps * 0.76)},
            {"comp": "SOFT", "laps": int(total_laps * 0.24)}
        ],
        "1-stop (S-H)": [
            {"comp": "SOFT", "laps": int(total_laps * 0.25)},
            {"comp": "HARD", "laps": int(total_laps * 0.75)}
        ],
        "1-stop (M-S)": [
            {"comp": "MEDIUM", "laps": int(total_laps * 0.59)},
            {"comp": "SOFT", "laps": int(total_laps * 0.41)}
        ],
        "1-stop (S-M)": [
            {"comp": "SOFT", "laps": int(total_laps * 0.43)},
            {"comp": "MEDIUM", "laps": int(total_laps * 0.57)}
        ],
        "2-stop (M-H-S)": [
            {"comp": "MEDIUM", "laps": int(total_laps * 0.29)},
            {"comp": "HARD", "laps": int(total_laps * 0.57)},
            {"comp": "SOFT", "laps": int(total_laps * 0.14)}
        ],
        "2-stop (H-M-H)": [
            {"comp": "HARD", "laps": int(total_laps * 0.21)},
            {"comp": "MEDIUM", "laps": int(total_laps * 0.29)},
            {"comp": "HARD", "laps": int(total_laps * 0.50)}
        ],
        "2-stop (S-H-S)": [
            {"comp": "SOFT", "laps": int(total_laps * 0.29)},
            {"comp": "HARD", "laps": int(total_laps * 0.57)},
            {"comp": "SOFT", "laps": int(total_laps * 0.14)}
        ],
        "2-stop (S-H-M)": [
            {"comp": "SOFT", "laps": int(total_laps * 0.29)},
            {"comp": "HARD", "laps": int(total_laps * 0.50)},
            {"comp": "MEDIUM", "laps": int(total_laps * 0.21)}
        ],
        "2-stop (S-M-S)": [
            {"comp": "SOFT", "laps": int(total_laps * 0.29)},
            {"comp": "MEDIUM", "laps": int(total_laps * 0.45)},
            {"comp": "SOFT", "laps": int(total_laps * 0.26)}
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
    
    if model_info:
        sess_info = model_info.get('info', {})
        sources = []
        if sess_info.get('fp1'):
            sources.append('FP1')
        if sess_info.get('fp2'):
            sources.append('FP2')
        if sess_info.get('sprint'):
            sources.append('Sprint')
        model_src = ' + '.join(sources) if sources else 'Practice'
    else:
        model_src = 'Fallback'
    
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
    
    if model_info:
        sess_info = model_info.get('info', {})
        sources = []
        if sess_info.get('fp1'):
            sources.append('FP1')
        if sess_info.get('fp2'):
            sources.append('FP2')
        if sess_info.get('sprint'):
            sources.append('Sprint')
        tire_src = ' + '.join(sources) + ' Models' if sources else 'Practice Models'
    else:
        tire_src = 'Fallback Models'
    print(f"Tire Models: {tire_src}")
    
    if model_info:
        print(f"Practice Data Sources:")
        sess_info = model_info.get('info', {})
        if sess_info.get('fp1'):
            print(f"  FP1: {sess_info.get('fp1_laps', 0)} laps")
        if sess_info.get('fp2'):
            print(f"  FP2: {sess_info.get('fp2_laps', 0)} laps")
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

def find_crossover_pace(strat_a, strat_b, compound, grid_pos, models, circuit, 
                        params=None, n_sims=150, tolerance=0.01):
    car_perf_map = {
        1: 0.98,   2: 0.985,  3: 0.99,   4: 0.995,  5: 1.00,   
        6: 1.005,  7: 1.01,   8: 1.015,  9: 1.02,   10: 1.025,
        11: 1.03,  12: 1.035, 13: 1.04,  14: 1.045, 15: 1.05,  
    }
    car_factor = car_perf_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
    
    def get_avg_time(adjustment):
        tire_adj = {compound: adjustment}
        times_a = []
        times_b = []
        
        for _ in range(n_sims):
            base_pace = np.random.normal(circuit['base_pace'], 0.5)
            temp_circuit = circuit.copy()
            temp_circuit['base_pace'] = base_pace
            
            time_a, _ = sim_race(strat_a, grid_pos, models, temp_circuit, 
                                params, car_factor, tire_adj)
            time_b, _ = sim_race(strat_b, grid_pos, models, temp_circuit, 
                                params, car_factor, tire_adj)
            
            times_a.append(time_a)
            times_b.append(time_b)
        
        return np.mean(times_a) - np.mean(times_b)
    
    baseline_diff = get_avg_time(0.0)
    
    if abs(baseline_diff) < tolerance:
        return 0.0, baseline_diff, False
    
    low, high = -5.0, 5.0
    
    for _ in range(25):
        mid = (low + high) / 2.0
        diff = get_avg_time(mid)
        
        if abs(diff) < tolerance:
            return mid, diff, False
        
        if (diff > 0) == (baseline_diff > 0):
            low = mid
        else:
            high = mid
    
    final_crossover = (low + high) / 2.0
    at_boundary = abs(abs(final_crossover) - 5.0) < 0.1
    
    return final_crossover, get_avg_time(final_crossover), at_boundary

def analyze_tire_sensitivity(strategy_name, strategy, compound, grid_pos, 
                             models, circuit, params=None, n_sims=150):
    adjustments = np.linspace(-2.0, 2.0, 9)
    results = []
    
    car_perf_map = {
        1: 0.98,   2: 0.985,  3: 0.99,   4: 0.995,  5: 1.00,   
        6: 1.005,  7: 1.01,   8: 1.015,  9: 1.02,   10: 1.025,
        11: 1.03,  12: 1.035, 13: 1.04,  14: 1.045, 15: 1.05,  
    }
    car_factor = car_perf_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
    
    for adj in adjustments:
        tire_adj = {compound: adj}
        times = []
        positions = []
        
        for _ in range(n_sims):
            base_pace = np.random.normal(circuit['base_pace'], 0.5)
            temp_circuit = circuit.copy()
            temp_circuit['base_pace'] = base_pace
            
            race_time, final_pos = sim_race(strategy, grid_pos, models, 
                                           temp_circuit, params, car_factor, tire_adj)
            times.append(race_time)
            positions.append(final_pos)
        
        results.append({
            'adjustment': adj,
            'avg_time': np.mean(times),
            'avg_position': np.mean(positions),
            'avg_points': np.mean([get_f1_pts(p) for p in positions])
        })
    
    return pd.DataFrame(results)

def get_optimal_pace_targets(target_strategy_name, target_strategy, all_strategies, 
                             grid_pos, models, circuit, params=None, n_sims=150):
    compounds_in_target = set(stint['comp'] for stint in target_strategy)
    
    results = {}
    
    for comp in compounds_in_target:
        comp_results = {}
        
        for comp_name, comp_strategy in all_strategies.items():
            if comp_name == target_strategy_name:
                continue
            
            crossover, diff, at_boundary = find_crossover_pace(
                target_strategy, comp_strategy, comp, grid_pos, 
                models, circuit, params, n_sims
            )
            
            comp_results[comp_name] = {
                'crossover_adjustment': crossover,
                'baseline_diff': diff,
                'at_boundary': at_boundary
            }
        
        results[comp] = comp_results
    
    return results

def print_sensitivity_analysis(target_strategy_name, target_strategy, all_strategies,
                               grid_pos, models, circuit, params=None, n_sims=150):
    print("\n" + "="*80)
    print(f"STRATEGY THRESHOLDS: {target_strategy_name}")
    print(f"Grid Position: P{grid_pos}")
    print("="*80)
    
    base_pace = circuit['base_pace']
    compounds_in_target = set(stint['comp'] for stint in target_strategy)
    
    for comp in compounds_in_target:
        print(f"\n{comp} COMPOUND")
        print("-" * 60)
        
        current_laptime = get_tire_perf(comp, 10, models, base_pace, 'dry', 0, params)
        print(f"Current pace: {current_laptime:.2f}s/lap\n")
        
        thresholds = []
        
        for comp_name, comp_strategy in all_strategies.items():
            if comp_name == target_strategy_name:
                continue
            
            print(f"  Analyzing vs {comp_name}...", end=' ', flush=True)
            crossover, baseline_diff, at_boundary = find_crossover_pace(
                target_strategy, comp_strategy, comp, grid_pos,
                models, circuit, params, n_sims
            )
            print("done")
            
            threshold_laptime = current_laptime + crossover
            comp_list = [s['comp'] for s in comp_strategy]
            
            thresholds.append({
                'strategy': comp_name,
                'compounds': ' â†’ '.join(comp_list),
                'threshold': threshold_laptime,
                'at_boundary': at_boundary
            })
        
        print()
        thresholds.sort(key=lambda x: x['threshold'])
        
        print(f"If {comp} runs slower than:")
        for t in thresholds:
            if t['at_boundary']:
                print(f"  >{t['threshold']:.2f}s/lap â†’ switch to {t['strategy']} ({t['compounds']})")
            else:
                print(f"  {t['threshold']:.2f}s/lap â†’ switch to {t['strategy']} ({t['compounds']})")
        print()

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pit_sim.py <year> <gp_name> [options]")
        print()
        print("Options:")
        print("  --sensitivity <strategy>  Analyze pace thresholds for a strategy")
        print("  --targets [compound]      Calculate laptime targets for optimal strategy")
        print("  --position <1-20>         Grid position for targets (default: 1)")
        print()
        print("Examples:")
        print("  python pit_sim.py 2025 'United States'")
        print("  python pit_sim.py 2025 'United States' --sensitivity '1-stop (S-H)'")
        print("  python pit_sim.py 2025 'Monaco' --targets")
        print("  python pit_sim.py 2025 'Monaco' --targets SOFT --position 3")
        return
    
    year = int(sys.argv[1])
    gp_key = sys.argv[2]
    
    run_sensitivity = False
    sensitivity_strategy = None
    run_targets = False
    target_compound = None
    target_position = 1
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--sensitivity':
            run_sensitivity = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                sensitivity_strategy = sys.argv[i + 1]
                i += 1
            i += 1
        elif sys.argv[i] == '--targets':
            run_targets = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                target_compound = sys.argv[i + 1].upper()
                i += 1
            i += 1
        elif sys.argv[i] == '--position':
            if i + 1 < len(sys.argv):
                target_position = int(sys.argv[i + 1])
                i += 1
            i += 1
        else:
            i += 1
    
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
        sess_info = model_info.get('info', {})
        sources = []
        if sess_info.get('fp1'):
            sources.append('FP1')
        if sess_info.get('fp2'):
            sources.append('FP2')
        if sess_info.get('sprint'):
            sources.append('Sprint')
        sources_str = '/'.join(sources) if sources else 'practice'
        print(f"Using tire models from {sources_str}")
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
    
    if run_sensitivity:
        if sensitivity_strategy is None:
            print("\n" + "="*80)
            print("AVAILABLE STRATEGIES FOR SENSITIVITY ANALYSIS:")
            print("="*80)
            for name in strats.keys():
                print(f"  - {name}")
            print("\nRe-run with: python pit_sim.py <year> <gp_name> --sensitivity '<strategy_name>'")
        else:
            if sensitivity_strategy not in strats:
                print(f"\nError: Strategy '{sensitivity_strategy}' not found")
                print("Available strategies:")
                for name in strats.keys():
                    print(f"  - {name}")
            else:
                for pos in [1, 3, 5]:
                    print_sensitivity_analysis(
                        sensitivity_strategy, 
                        strats[sensitivity_strategy],
                        strats,
                        pos,
                        models,
                        circuit,
                        params,
                        n_sims=150
                    )
    
    if run_targets:
        from laptime_targets import LaptimeTargetCalculator, print_laptime_targets
        from laptime_targets import print_compound_thresholds, print_strategy_summary
        
        calculator = LaptimeTargetCalculator(circuit, models, params)
        
        if target_compound:
            print_compound_thresholds(calculator, target_compound, target_position, n_sims=100)
        else:
            print_laptime_targets(calculator, target_position, n_sims=100)
            print_strategy_summary(calculator, target_position, n_sims=100)

if __name__ == "__main__":
    main()