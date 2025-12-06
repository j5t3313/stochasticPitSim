import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys

from tire_model import build_models, get_tire_perf
from pit_sim import load_params, get_strats, sim_race, get_f1_pts

class LaptimeTargetCalculator:
    def __init__(self, circuit, models, params=None):
        self.circuit = circuit
        self.models = models
        self.params = params
        self.base_pace = circuit['base_pace']
        self.strategies = get_strats(circuit)
        
        self.car_perf_map = {
            1: 0.98,   2: 0.985,  3: 0.99,   4: 0.995,  5: 1.00,   
            6: 1.005,  7: 1.01,   8: 1.015,  9: 1.02,   10: 1.025,
            11: 1.03,  12: 1.035, 13: 1.04,  14: 1.045, 15: 1.05,  
        }
    
    def get_current_compound_pace(self, compound, stint_lap=10):
        return get_tire_perf(compound, stint_lap, self.models, self.base_pace, 
                            'dry', 0, self.params)
    
    def simulate_strategy_batch(self, strategy, grid_pos, tire_adjustments, n_sims=100):
        car_factor = self.car_perf_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
        times = []
        positions = []
        
        for _ in range(n_sims):
            base_pace = np.random.normal(self.circuit['base_pace'], 0.5)
            temp_circuit = self.circuit.copy()
            temp_circuit['base_pace'] = base_pace
            
            race_time, final_pos = sim_race(strategy, grid_pos, self.models, 
                                           temp_circuit, self.params, car_factor, 
                                           tire_adjustments)
            times.append(race_time)
            positions.append(final_pos)
        
        return {
            'avg_time': np.mean(times),
            'avg_pos': np.mean(positions),
            'avg_pts': np.mean([get_f1_pts(p) for p in positions])
        }
    
    def find_crossover_adjustment(self, strat_a, strat_b, compound, grid_pos, 
                                   n_sims=100, tolerance=0.5):
        def time_diff(adj):
            tire_adj = {compound: adj}
            res_a = self.simulate_strategy_batch(strat_a, grid_pos, tire_adj, n_sims)
            res_b = self.simulate_strategy_batch(strat_b, grid_pos, tire_adj, n_sims)
            return res_a['avg_time'] - res_b['avg_time']
        
        baseline = time_diff(0.0)
        
        if abs(baseline) < tolerance:
            return 0.0, baseline, 'equal'
        
        low, high = -5.0, 5.0
        
        for _ in range(20):
            mid = (low + high) / 2.0
            diff = time_diff(mid)
            
            if abs(diff) < tolerance:
                return mid, diff, 'found'
            
            if (diff > 0) == (baseline > 0):
                low = mid
            else:
                high = mid
        
        final = (low + high) / 2.0
        
        if abs(final) > 4.5:
            return final, time_diff(final), 'boundary'
        
        return final, time_diff(final), 'found'
    
    def calculate_all_thresholds(self, grid_pos, n_sims=100, verbose=True):
        all_compounds = set()
        for strat in self.strategies.values():
            for stint in strat:
                all_compounds.add(stint['comp'])
        
        results = {}
        strat_names = list(self.strategies.keys())
        n_comparisons = len(strat_names) * (len(strat_names) - 1) // 2 * len(all_compounds)
        
        if verbose:
            print(f"\nCalculating thresholds for P{grid_pos}...")
            print(f"Strategies: {len(strat_names)}, Compounds: {len(all_compounds)}")
            print(f"Total comparisons: {n_comparisons}")
        
        pbar = tqdm(total=n_comparisons, disable=not verbose, desc="Analyzing")
        
        for compound in all_compounds:
            compound_results = {}
            
            for i, name_a in enumerate(strat_names):
                for name_b in strat_names[i+1:]:
                    strat_a = self.strategies[name_a]
                    strat_b = self.strategies[name_b]
                    
                    a_uses_compound = any(s['comp'] == compound for s in strat_a)
                    b_uses_compound = any(s['comp'] == compound for s in strat_b)
                    
                    if not a_uses_compound and not b_uses_compound:
                        pbar.update(1)
                        continue
                    
                    crossover, diff, status = self.find_crossover_adjustment(
                        strat_a, strat_b, compound, grid_pos, n_sims
                    )
                    
                    current_pace = self.get_current_compound_pace(compound)
                    threshold_pace = current_pace + crossover
                    
                    key = f"{name_a} vs {name_b}"
                    compound_results[key] = {
                        'crossover_adj': crossover,
                        'threshold_pace': threshold_pace,
                        'baseline_diff': diff,
                        'status': status,
                        'strat_a': name_a,
                        'strat_b': name_b,
                        'a_uses': a_uses_compound,
                        'b_uses': b_uses_compound
                    }
                    
                    pbar.update(1)
            
            results[compound] = compound_results
        
        pbar.close()
        return results
    
    def calculate_strategy_windows(self, grid_pos, n_sims=100, verbose=True):
        thresholds = self.calculate_all_thresholds(grid_pos, n_sims, verbose)
        
        windows = {}
        
        for strat_name, strat in self.strategies.items():
            strat_compounds = set(s['comp'] for s in strat)
            strat_windows = {}
            
            for compound in strat_compounds:
                current_pace = self.get_current_compound_pace(compound)
                
                slower_threshold = float('inf')
                switch_to_slower = None
                faster_threshold = float('-inf')
                switch_to_faster = None
                
                for comp_key, data in thresholds.get(compound, {}).items():
                    if strat_name not in comp_key:
                        continue
                    
                    other_strat = data['strat_b'] if data['strat_a'] == strat_name else data['strat_a']
                    
                    res_current = self.simulate_strategy_batch(strat, grid_pos, {}, 50)
                    res_other = self.simulate_strategy_batch(
                        self.strategies[other_strat], grid_pos, {}, 50
                    )
                    
                    current_better = res_current['avg_time'] < res_other['avg_time']
                    
                    threshold = data['threshold_pace']
                    
                    if current_better:
                        if threshold < slower_threshold:
                            slower_threshold = threshold
                            switch_to_slower = other_strat
                    else:
                        if threshold > faster_threshold:
                            faster_threshold = threshold
                            switch_to_faster = other_strat
                
                strat_windows[compound] = {
                    'current_pace': current_pace,
                    'max_pace': slower_threshold if slower_threshold != float('inf') else None,
                    'switch_if_slower': switch_to_slower,
                    'min_pace': faster_threshold if faster_threshold != float('-inf') else None,
                    'switch_if_faster': switch_to_faster
                }
            
            windows[strat_name] = strat_windows
        
        return windows
    
    def find_best_strategy_for_pace(self, compound_paces, grid_pos, n_sims=100):
        best_strat = None
        best_time = float('inf')
        all_results = {}
        
        for strat_name, strat in self.strategies.items():
            tire_adj = {}
            for compound, pace in compound_paces.items():
                current = self.get_current_compound_pace(compound)
                tire_adj[compound] = pace - current
            
            result = self.simulate_strategy_batch(strat, grid_pos, tire_adj, n_sims)
            all_results[strat_name] = result
            
            if result['avg_time'] < best_time:
                best_time = result['avg_time']
                best_strat = strat_name
        
        return best_strat, all_results
    
    def generate_pace_sensitivity_table(self, compound, grid_pos, 
                                         pace_range=(-2.0, 2.0), steps=9, n_sims=100):
        current_pace = self.get_current_compound_pace(compound)
        adjustments = np.linspace(pace_range[0], pace_range[1], steps)
        
        results = []
        
        for adj in tqdm(adjustments, desc=f"Analyzing {compound}"):
            tire_adj = {compound: adj}
            pace = current_pace + adj
            
            strat_results = {}
            for strat_name, strat in self.strategies.items():
                if not any(s['comp'] == compound for s in strat):
                    continue
                res = self.simulate_strategy_batch(strat, grid_pos, tire_adj, n_sims)
                strat_results[strat_name] = res
            
            if strat_results:
                best = min(strat_results.items(), key=lambda x: x[1]['avg_time'])
                results.append({
                    'pace': pace,
                    'adjustment': adj,
                    'best_strategy': best[0],
                    'best_time': best[1]['avg_time'],
                    'best_position': best[1]['avg_pos'],
                    'best_points': best[1]['avg_pts']
                })
        
        return pd.DataFrame(results)

def print_laptime_targets(calculator, grid_pos, n_sims=100):
    print("\n" + "="*80)
    print(f"LAPTIME TARGETS - GRID POSITION P{grid_pos}")
    print("="*80)
    
    compounds = set()
    for strat in calculator.strategies.values():
        for stint in strat:
            compounds.add(stint['comp'])
    
    for compound in sorted(compounds):
        current_pace = calculator.get_current_compound_pace(compound)
        
        print(f"\n{compound} COMPOUND")
        print("-"*60)
        print(f"Current pace (lap 10): {current_pace:.2f}s")
        print()
        
        strat_using = [name for name, strat in calculator.strategies.items() 
                       if any(s['comp'] == compound for s in strat)]
        
        if len(strat_using) < 2:
            print(f"  Only one strategy uses {compound}")
            continue
        
        baseline_results = {}
        for strat_name in strat_using:
            res = calculator.simulate_strategy_batch(
                calculator.strategies[strat_name], grid_pos, {}, n_sims
            )
            baseline_results[strat_name] = res
        
        ranked = sorted(baseline_results.items(), key=lambda x: x[1]['avg_time'])
        best_strat = ranked[0][0]
        
        print(f"Current best strategy: {best_strat}")
        print(f"  Avg time: {ranked[0][1]['avg_time']:.1f}s, Avg pos: {ranked[0][1]['avg_pos']:.1f}")
        print()
        
        thresholds = []
        
        for other_name in strat_using:
            if other_name == best_strat:
                continue
            
            crossover, diff, status = calculator.find_crossover_adjustment(
                calculator.strategies[best_strat],
                calculator.strategies[other_name],
                compound, grid_pos, n_sims
            )
            
            threshold_pace = current_pace + crossover
            
            thresholds.append({
                'alternative': other_name,
                'threshold': threshold_pace,
                'adjustment': crossover,
                'status': status
            })
        
        thresholds.sort(key=lambda x: x['threshold'])
        
        print(f"Switch from {best_strat} when {compound} is slower than:")
        for t in thresholds:
            if t['status'] == 'boundary':
                prefix = ">" if t['adjustment'] > 0 else "<"
                print(f"  {prefix}{t['threshold']:.2f}s/lap: switch to {t['alternative']}")
            else:
                print(f"  {t['threshold']:.2f}s/lap: switch to {t['alternative']}")
        print()

def print_strategy_summary(calculator, grid_pos, n_sims=100):
    print("\n" + "="*80)
    print(f"STRATEGY OPTIMAL PACE RANGES - GRID POSITION P{grid_pos}")
    print("="*80)
    
    compounds = set()
    for strat in calculator.strategies.values():
        for stint in strat:
            compounds.add(stint['comp'])
    
    for compound in sorted(compounds):
        print(f"\n{compound} COMPOUND TARGETS")
        print("-"*60)
        
        current_pace = calculator.get_current_compound_pace(compound)
        print(f"Current modeled pace: {current_pace:.2f}s/lap")
        print()
        
        sensitivity = calculator.generate_pace_sensitivity_table(
            compound, grid_pos, pace_range=(-1.5, 1.5), steps=7, n_sims=n_sims
        )
        
        print(f"{'Laptime':<12} {'Best Strategy':<20} {'Avg Pos':<10} {'Avg Pts':<10}")
        print("-"*52)
        
        for _, row in sensitivity.iterrows():
            print(f"{row['pace']:.2f}s/lap    {row['best_strategy']:<20} "
                  f"{row['best_position']:.1f}       {row['best_points']:.1f}")
        
        print()

def print_compound_thresholds(calculator, compound, grid_pos, n_sims=100):
    print("\n" + "="*80)
    print(f"{compound} LAPTIME THRESHOLDS - GRID POSITION P{grid_pos}")
    print("="*80)
    
    current_pace = calculator.get_current_compound_pace(compound)
    print(f"\nCurrent {compound} pace (lap 10): {current_pace:.2f}s")
    
    strat_using = [name for name, strat in calculator.strategies.items() 
                   if any(s['comp'] == compound for s in strat)]
    
    print(f"Strategies using {compound}: {len(strat_using)}")
    for name in strat_using:
        stints = [s['comp'] for s in calculator.strategies[name]]
        print(f"  {name}: {' -> '.join(stints)}")
    print()
    
    if len(strat_using) < 2:
        print("Insufficient strategies for comparison")
        return
    
    print("Pairwise crossover analysis:")
    print("-"*70)
    
    for i, name_a in enumerate(strat_using):
        for name_b in strat_using[i+1:]:
            strat_a = calculator.strategies[name_a]
            strat_b = calculator.strategies[name_b]
            
            crossover, diff, status = calculator.find_crossover_adjustment(
                strat_a, strat_b, compound, grid_pos, n_sims
            )
            
            threshold_pace = current_pace + crossover
            
            if crossover > 0:
                better_now = name_a
                better_slow = name_b
            else:
                better_now = name_b
                better_slow = name_a
                threshold_pace = current_pace - crossover
            
            print(f"\n{name_a} vs {name_b}")
            print(f"  Crossover at: {threshold_pace:.2f}s/lap ({crossover:+.2f}s from current)")
            print(f"  Currently faster: {better_now}")
            if status == 'boundary':
                print(f"  Note: At analysis boundary, actual threshold may differ")

def main():
    if len(sys.argv) < 3:
        print("Usage: python laptime_targets.py <year> <gp_name> [options]")
        print()
        print("Options:")
        print("  --compound <SOFT|MEDIUM|HARD>  Analyze specific compound")
        print("  --position <1-20>              Grid position (default: 1)")
        print("  --sims <n>                     Simulations per comparison (default: 100)")
        print()
        print("Examples:")
        print("  python laptime_targets.py 2025 'United States'")
        print("  python laptime_targets.py 2025 'Monaco' --compound SOFT --position 3")
        return
    
    year = int(sys.argv[1])
    gp_key = sys.argv[2]
    
    compound_filter = None
    grid_pos = 1
    n_sims = 100
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--compound' and i + 1 < len(sys.argv):
            compound_filter = sys.argv[i + 1].upper()
            i += 2
        elif sys.argv[i] == '--position' and i + 1 < len(sys.argv):
            grid_pos = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--sims' and i + 1 < len(sys.argv):
            n_sims = int(sys.argv[i + 1])
            i += 2
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
    print(f"LAPTIME TARGET ANALYSIS: {gp_name.upper()}")
    print(f"{'='*80}")
    print(f"Grid Position: P{grid_pos}")
    print(f"Simulations per comparison: {n_sims}")
    
    params, _ = load_params(gp_name)
    
    print("\nBuilding tire models...")
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
        print(f"Using tire data from: {'/'.join(sources) if sources else 'fallback'}")
    
    calculator = LaptimeTargetCalculator(circuit, models, params)
    
    if compound_filter:
        print_compound_thresholds(calculator, compound_filter, grid_pos, n_sims)
    else:
        print_laptime_targets(calculator, grid_pos, n_sims)
        print_strategy_summary(calculator, grid_pos, n_sims)

if __name__ == "__main__":
    main()