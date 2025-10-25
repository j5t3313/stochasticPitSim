import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import json

warnings.filterwarnings('ignore')

from pit_sim import load_params, get_strats, run_mc, get_f1_pts
from tire_model import build_models

class Validator:
    def __init__(self, year, gp_key, circuit):
        self.year = year
        self.gp_key = gp_key
        self.circuit = circuit
        self.gp_name = circuit['gp_name']
        self.actual_results = None
        self.actual_data = None
        self.sim_results = None
        
    def load_actual(self):
        try:
            print(f"Loading {self.year} {self.gp_name} race data...")
            
            race = fastf1.get_session(self.year, self.gp_name, 'R')
            race.load()
            
            results = race.results
            self.actual_results = results[['Abbreviation', 'Position', 'GridPosition', 
                                          'Time', 'Points']].copy()
            self.actual_results = self.actual_results.dropna(subset=['Position'])
            self.actual_results['Position'] = self.actual_results['Position'].astype(int)
            self.actual_results['GridPosition'] = self.actual_results['GridPosition'].astype(int)
            
            laps = race.laps
            self.actual_data = {
                'laps': laps,
                'weather': self._analyze_weather(laps),
                'sc': self._detect_sc(laps),
                'strats': self._extract_strats(laps),
                'total_laps': race.total_laps,
                'race_time': race.results['Time'].iloc[0] if not race.results.empty else None
            }
            
            print(f"Loaded data for {len(self.actual_results)} drivers")
            print(f"Winner: {self.actual_results.iloc[0]['Abbreviation']} "
                  f"from P{self.actual_results.iloc[0]['GridPosition']}")
            return True
            
        except Exception as e:
            print(f"Error loading race data: {e}")
            return False
    
    def _analyze_weather(self, laps):
        weather = {
            'dry_laps': 0,
            'int_laps': 0,
            'wet_laps': 0,
            'rain': False
        }
        
        comp_counts = laps['Compound'].value_counts()
        
        dry_comps = ['SOFT', 'MEDIUM', 'HARD']
        wet_comps = ['INTERMEDIATE', 'WET']
        
        for comp in dry_comps:
            if comp in comp_counts:
                weather['dry_laps'] += comp_counts[comp]
        
        for comp in wet_comps:
            if comp in comp_counts:
                if comp == 'INTERMEDIATE':
                    weather['int_laps'] += comp_counts[comp]
                else:
                    weather['wet_laps'] += comp_counts[comp]
        
        weather['rain'] = (weather['int_laps'] + weather['wet_laps']) > 0
        weather['total_laps'] = len(laps['LapNumber'].unique())
        weather['rain_pct'] = (weather['int_laps'] + weather['wet_laps']) / len(laps) * 100
        
        return weather
    
    def _detect_sc(self, laps):
        lap_med = laps.groupby('LapNumber')['LapTime'].median()
        lap_med_sec = lap_med.dt.total_seconds()
        baseline = lap_med_sec.rolling(window=5, center=True).median()
        
        sc_thresh = 1.25
        potential_sc = []
        
        for lap_num, lap_time in lap_med_sec.items():
            if not pd.isna(baseline.loc[lap_num]):
                if lap_time > baseline.loc[lap_num] * sc_thresh:
                    potential_sc.append(lap_num)
        
        return {
            'sc_laps': potential_sc,
            'sc_occurred': len(potential_sc) > 0,
            'sc_count': len(potential_sc),
            'sc_pct': len(potential_sc) / len(lap_med) * 100 if len(lap_med) > 0 else 0
        }
    
    def _extract_strats(self, laps):
        strats = {}
        
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver]
            stints = driver_laps.groupby('Stint')
            
            driver_strat = []
            total_stops = 0
            
            for stint_num, stint_data in stints:
                if len(stint_data) > 0:
                    comp = stint_data['Compound'].iloc[0] if not stint_data['Compound'].isna().all() else 'UNKNOWN'
                    stint_len = len(stint_data)
                    
                    driver_strat.append({
                        'comp': comp,
                        'laps': stint_len,
                        'stint': stint_num
                    })
                    
                    if stint_num > 1:
                        total_stops += 1
            
            strats[driver] = {
                'strat': driver_strat,
                'stops': total_stops,
                'type': self._classify_strat(driver_strat)
            }
            
        return strats
    
    def _classify_strat(self, strat):
        if len(strat) == 1:
            return "0-stop"
        elif len(strat) == 2:
            comps = [s['comp'] for s in strat]
            return f"1-stop ({'-'.join(comps)})"
        elif len(strat) == 3:
            comps = [s['comp'] for s in strat]
            return f"2-stop ({'-'.join(comps)})"
        else:
            return f"{len(strat)-1}-stop"
    
    def gen_sim_results(self, grid_pos=[1, 3, 5, 8, 10, 15], n_sims=1500):
        print("Generating simulation results...")
        
        params, params_loaded = load_params(self.gp_name)
        
        print("Building tire models...")
        models, model_info = build_models(self.year, self.gp_name, self.circuit['base_pace'])
        
        strats = get_strats(self.circuit)
        
        sim_data = run_mc(strats, models, self.circuit, grid_pos, params, n_sims)
        
        val_results = {}
        
        for gp in grid_pos:
            val_results[gp] = {}
            
            for strat_name, data in sim_data[gp].items():
                val_results[gp][strat_name] = {
                    'final_pos': data['final_pos'],
                    'pts': data['pts'],
                    'times': data['times']
                }
        
        return val_results
    
    def validate_positions(self, grid_pos=[1, 3, 5, 8, 10, 15]):
        if self.actual_results is None or self.sim_results is None:
            print("Error: Missing data")
            return None
        
        val_results = {}
        
        print("\n" + "="*80)
        print("POSITION PREDICTION VALIDATION")
        print("="*80)
        
        weather = self.actual_data['weather'] if self.actual_data else {'rain': False}
        was_wet = weather['rain']
        
        print(f"Race conditions: {'Wet' if was_wet else 'Dry'}")
        
        for gp in grid_pos:
            actual_drv = self.actual_results[self.actual_results['GridPosition'] == gp]
            
            if len(actual_drv) == 0:
                print(f"No driver from P{gp}")
                continue
                
            actual_pos = actual_drv['Position'].iloc[0]
            actual_pts = actual_drv['Points'].iloc[0]
            drv_name = actual_drv['Abbreviation'].iloc[0]
            
            print(f"\nP{gp} - {drv_name}: Actual P{actual_pos} ({actual_pts} pts)")
            
            if gp in self.sim_results:
                sim_data = self.sim_results[gp]
                
                strat_acc = {}
                best_strat = None
                best_acc = float('inf')
                
                for strat, results in sim_data.items():
                    pred_pos = np.array(results['final_pos'])
                    pred_pts = np.array(results['pts'])
                    
                    pos_mae = np.mean(np.abs(pred_pos - actual_pos))
                    pos_rmse = np.sqrt(np.mean((pred_pos - actual_pos)**2))
                    pts_mae = np.mean(np.abs(pred_pts - actual_pts))
                    
                    acc_within_2 = np.mean(np.abs(pred_pos - actual_pos) <= 2) * 100
                    
                    mean_pred_pos = np.mean(pred_pos)
                    mean_pred_pts = np.mean(pred_pts)
                    
                    strat_acc[strat] = {
                        'pos_mae': pos_mae,
                        'pos_rmse': pos_rmse,
                        'pts_mae': pts_mae,
                        'acc_within_2': acc_within_2,
                        'mean_pred_pos': mean_pred_pos,
                        'mean_pred_pts': mean_pred_pts,
                        'pos_err': mean_pred_pos - actual_pos
                    }
                    
                    if pos_mae < best_acc:
                        best_acc = pos_mae
                        best_strat = strat
                
                val_results[gp] = {
                    'driver': drv_name,
                    'actual_pos': actual_pos,
                    'actual_pts': actual_pts,
                    'strat_acc': strat_acc,
                    'best_strat': best_strat,
                    'best_mae': best_acc,
                    'conditions': 'wet' if was_wet else 'dry'
                }
                
                print(f"Best: {best_strat} (MAE: {best_acc:.2f})")
                print(f"Pred: P{strat_acc[best_strat]['mean_pred_pos']:.1f}, "
                      f"Actual: P{actual_pos}")
        
        return val_results
    
    def validate_conditions(self):
        if self.actual_data is None:
            print("Error: No actual data")
            return None
        
        print("\n" + "="*80)
        print("RACE CONDITIONS VALIDATION")
        print("="*80)
        
        weather = self.actual_data['weather']
        sc = self.actual_data['sc']
        
        print(f"\nWeather:")
        print(f"Rain occurred: {weather['rain']}")
        print(f"Rain %: {weather['rain_pct']:.1f}%")
        print(f"Dry laps: {weather['dry_laps']}")
        print(f"Int laps: {weather['int_laps']}")
        print(f"Wet laps: {weather['wet_laps']}")
        
        print(f"\nSafety Car:")
        print(f"SC occurred: {sc['sc_occurred']}")
        print(f"SC laps: {sc['sc_count']}")
        print(f"SC %: {sc['sc_pct']:.1f}%")
        if sc['sc_laps']:
            print(f"SC laps: {sc['sc_laps']}")
        
        print(f"\nSimulation vs Actual:")
        
        sim_rain = self.circuit['rain_prob']
        sim_sc = self.circuit['sc_prob']
        sim_vsc = self.circuit['vsc_prob']
        
        print(f"Rain prob: {sim_rain:.0%}")
        print(f"SC prob: {sim_sc:.0%}")
        print(f"VSC prob: {sim_vsc:.0%}")
        
        actual_rain = weather['rain']
        actual_sc = sc['sc_occurred']
        print(f"Actual rain: {'Yes' if actual_rain else 'No'}")
        print(f"Actual SC: {'Yes' if actual_sc else 'No'}")
        
        return {
            'weather': weather,
            'sc': sc,
            'probs': {
                'rain': sim_rain,
                'sc': sim_sc,
                'vsc': sim_vsc
            }
        }
    
    def validate_strats(self):
        if self.actual_data is None:
            print("Error: No actual data")
            return None
        
        print("\n" + "="*80)
        print("STRATEGY VALIDATION")
        print("="*80)
        
        strats = self.actual_data['strats']
        
        strat_dist = {}
        for driver, data in strats.items():
            strat_type = data['type']
            if strat_type not in strat_dist:
                strat_dist[strat_type] = 0
            strat_dist[strat_type] += 1
        
        print(f"\nActual strategy distribution:")
        for strat, count in sorted(strat_dist.items()):
            print(f"{strat}: {count} drivers")
        
        wet_usage = 0
        for driver, data in strats.items():
            for stint in data['strat']:
                if stint['comp'] == 'WET':
                    wet_usage += 1
        
        print(f"\nWET tire usage: {wet_usage} stints")
        
        strat_results = {}
        for _, row in self.actual_results.iterrows():
            driver = row['Abbreviation']
            if driver in strats:
                strat_type = strats[driver]['type']
                if strat_type not in strat_results:
                    strat_results[strat_type] = []
                strat_results[strat_type].append({
                    'pos': row['Position'],
                    'grid': row['GridPosition'],
                    'pts': row['Points']
                })
        
        print(f"\nStrategy effectiveness:")
        for strat, results in strat_results.items():
            avg_fin = np.mean([r['pos'] for r in results])
            avg_pts = np.mean([r['pts'] for r in results])
            pos_gained = np.mean([r['grid'] - r['pos'] for r in results])
            print(f"{strat}: Avg P{avg_fin:.1f}, "
                  f"Pts {avg_pts:.1f}, "
                  f"{'Gained' if pos_gained > 0 else 'Lost'} {abs(pos_gained):.1f}")
        
        return {
            'dist': strat_dist,
            'results': strat_results,
            'wet_usage': wet_usage
        }
    
    def plot(self, val_results):
        if val_results is None:
            print("No validation results")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{self.gp_name} - Model Validation', fontsize=16, y=0.98)
        
        ax1 = axes[0, 0]
        grid_pos = []
        actual_pos = []
        pred_pos = []
        drivers = []
        
        for gp, data in val_results.items():
            grid_pos.append(gp)
            actual_pos.append(data['actual_pos'])
            best = data['best_strat']
            pred_pos.append(data['strat_acc'][best]['mean_pred_pos'])
            drivers.append(data['driver'])
        
        ax1.scatter(pred_pos, actual_pos, s=120, alpha=0.8)
        ax1.plot([1, 20], [1, 20], 'r--', alpha=0.6, label='Perfect', linewidth=2)
        ax1.set_xlabel('Predicted Position', fontsize=12)
        ax1.set_ylabel('Actual Position', fontsize=12)
        ax1.set_title('Predicted vs Actual Positions', fontsize=14, pad=20)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        for i, drv in enumerate(drivers):
            ax1.annotate(drv, (pred_pos[i], actual_pos[i]), 
                        xytext=(8, 8), textcoords='offset points', fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax2 = axes[0, 1]
        mae_vals = [data['best_mae'] for data in val_results.values()]
        bars2 = ax2.bar(grid_pos, mae_vals, alpha=0.8, color='steelblue')
        ax2.set_xlabel('Starting Grid Position', fontsize=12)
        ax2.set_ylabel('MAE (positions)', fontsize=12)
        ax2.set_title('Prediction Accuracy by Grid', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, mae_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax3 = axes[1, 0]
        all_strats = set()
        for data in val_results.values():
            all_strats.update(data['strat_acc'].keys())
        
        strat_mae = {s: [] for s in all_strats}
        for data in val_results.values():
            for s in all_strats:
                if s in data['strat_acc']:
                    strat_mae[s].append(data['strat_acc'][s]['pos_mae'])
                else:
                    strat_mae[s].append(np.nan)
        
        strat_names = list(strat_mae.keys())
        avg_mae = [np.nanmean(strat_mae[s]) for s in strat_names]
        
        short_names = []
        for name in strat_names:
            if len(name) > 15:
                short_names.append(name[:12] + '...')
            else:
                short_names.append(name)
        
        bars3 = ax3.bar(range(len(short_names)), avg_mae, alpha=0.8, color='lightcoral')
        ax3.set_xlabel('Strategy', fontsize=12)
        ax3.set_ylabel('Avg MAE', fontsize=12)
        ax3.set_title('Strategy Accuracy', fontsize=14, pad=20)
        ax3.set_xticks(range(len(short_names)))
        ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        pos_changes = [data['actual_pos'] - gp for gp, data in val_results.items()]
        colors = ['green' if x < 0 else 'red' for x in pos_changes]
        bars4 = ax4.bar(grid_pos, pos_changes, alpha=0.8, color=colors)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax4.set_xlabel('Starting Grid', fontsize=12)
        ax4.set_ylabel('Position Change', fontsize=12)
        ax4.set_title('Actual Position Changes', fontsize=14, pad=20)
        ax4.grid(True, alpha=0.3)
        
        for bar, val in zip(bars4, pos_changes):
            y_pos = bar.get_height() + (0.2 if val >= 0 else -0.4)
            ax4.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{val:+.0f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
        
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.90, 
                           wspace=0.25, hspace=0.35)
        plt.show()

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python validation.py <year> <gp_name>")
        print("Example: python validation.py 2025 'United States'")
        return
    
    year = int(sys.argv[1])
    gp_key = sys.argv[2]
    
    with open('circuit_config.json', 'r') as f:
        circuits = json.load(f)
    
    if gp_key not in circuits:
        print(f"Circuit '{gp_key}' not found")
        return
    
    circuit = circuits[gp_key]
    
    val = Validator(year, gp_key, circuit)
    
    if not val.load_actual():
        print(f"\n{year} {circuit['gp_name']} data not available")
        return
    
    print("\nLoading simulation predictions...")
    val.sim_results = val.gen_sim_results()
    
    pos_val = val.validate_positions()
    cond_val = val.validate_conditions()
    strat_val = val.validate_strats()
    
    if pos_val:
        val.plot(pos_val)
    
    if pos_val:
        print(f"\nVALIDATION SUMMARY:")
        total_mae = np.mean([data['best_mae'] for data in pos_val.values()])
        print(f"Avg Prediction Error: {total_mae:.2f} positions")
        
        if total_mae <= 2.0:
            print("Excellent accuracy")
        elif total_mae <= 3.0:
            print("Good accuracy")
        elif total_mae <= 4.0:
            print("Moderate accuracy")
        else:
            print("Poor accuracy")

if __name__ == "__main__":
    main()
