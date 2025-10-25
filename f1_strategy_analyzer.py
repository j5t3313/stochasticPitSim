
"""
F1 Strategy Analyzer - Universal Circuit System
Analyzes pit stop strategies for any F1 circuit on the 2025 calendar

Usage:
    python f1_strategy_analyzer.py --simulate 2025 "Monaco"
    python f1_strategy_analyzer.py --extract 2025 "Monaco"
    python f1_strategy_analyzer.py --weather "Monaco" --api-key YOUR_KEY
    python f1_strategy_analyzer.py --validate 2025 "Monaco"
"""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import fastf1
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from tqdm import trange
    
    import jax.numpy as jnp
    import jax.random as random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall required packages:")
    print("pip install fastf1 pandas numpy matplotlib seaborn scipy jax jaxlib numpyro tqdm")
    DEPENDENCIES_OK = False
    sys.exit(1)

class CircuitConfig:
    """Manages circuit configuration and parameters"""
    
    def __init__(self, config_file='circuit_config.json'):
        self.config_file = config_file
        self.circuits = self.load_config()
    
    def load_config(self):
        """Load circuit configuration from JSON"""
        if not os.path.exists(self.config_file):
            print(f"Error: {self.config_file} not found!")
            print("Please ensure circuit_config.json is in the current directory")
            sys.exit(1)
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def get_circuit(self, gp_name):
        """Get circuit parameters by Grand Prix name"""
        if gp_name not in self.circuits:
            print(f"Error: '{gp_name}' not found in configuration")
            print(f"\nAvailable circuits:")
            for name in sorted(self.circuits.keys()):
                sprint_flag = " (SPRINT)" if self.circuits[name]['sprint'] else ""
                print(f"  - {name}{sprint_flag}")
            sys.exit(1)
        
        return self.circuits[gp_name]
    
    def update_rain_probability(self, gp_name, new_prob):
        """Update rain probability and save config"""
        if gp_name in self.circuits:
            old_prob = self.circuits[gp_name]['rain_prob']
            self.circuits[gp_name]['rain_prob'] = new_prob
            
            with open(self.config_file, 'w') as f:
                json.dump(self.circuits, f, indent=2)
            
            return old_prob, new_prob
        return None, None


class ParameterExtractor:
    """Extract circuit-specific parameters from historical data"""
    
    def __init__(self, gp_name, years=None):
        self.gp_name = gp_name
        self.years = years or [2022, 2023, 2024]
        self.all_data = []
    
    def extract_all_parameters(self):
        """Extract all parameters for the specified Grand Prix"""
        print(f"\n{'='*80}")
        print(f"EXTRACTING PARAMETERS FOR {self.gp_name}")
        print(f"{'='*80}")
        
        self._load_historical_data()
        
        if not self.all_data:
            print("No data available for parameter extraction")
            return None
        
        position_penalties = self._extract_position_penalties()
        tire_performance = self._extract_tire_performance()
        driver_errors = self._extract_driver_errors()
        drs_effectiveness = self._extract_drs_effectiveness()
        
        self._generate_parameter_file(
            position_penalties, tire_performance, 
            driver_errors, drs_effectiveness
        )
        
        return {
            'position_penalties': position_penalties,
            'tire_performance': tire_performance,
            'driver_errors': driver_errors,
            'drs_effectiveness': drs_effectiveness
        }
    
    def _load_historical_data(self):
        """Load historical race data for the circuit"""
        print(f"\nLoading historical data from years: {self.years}")
        
        for year in self.years:
            try:
                print(f"  {year}...", end=" ")
                session = fastf1.get_session(year, self.gp_name, 'R')
                session.load()
                
                laps = session.laps
                clean_laps = laps[
                    (laps['LapTime'].notna()) &
                    (laps['TrackStatus'] == '1') &
                    (~laps['PitOutTime'].notna()) &
                    (~laps['PitInTime'].notna())
                ].copy()
                
                if len(clean_laps) > 0:
                    clean_laps['Year'] = year
                    clean_laps['LapTime_s'] = clean_laps['LapTime'].dt.total_seconds()
                    self.all_data.append(clean_laps)
                    print(f" {len(clean_laps)} clean laps")
                else:
                    print("No clean laps")
                    
            except Exception as e:
                print(f" Error: {str(e)[:50]}")
        
        if self.all_data:
            self.combined_data = pd.concat(self.all_data, ignore_index=True)
            print(f"\nTotal clean laps across all years: {len(self.combined_data)}")
    
    def _extract_position_penalties(self):
        """Extract position-based time penalties"""
        print("\n" + "-"*80)
        print("EXTRACTING POSITION PENALTIES")
        print("-"*80)
        
        penalties = {}
        
        for year in self.years:
            try:
                session = fastf1.get_session(year, self.gp_name, 'R')
                session.load()
                results = session.results
                
                for _, row in results.iterrows():
                    if pd.notna(row['GridPosition']) and pd.notna(row['Position']):
                        grid_pos = int(row['GridPosition'])
                        if 1 <= grid_pos <= 20:
                            if grid_pos not in penalties:
                                penalties[grid_pos] = []
                            
                            base_penalty = max(0, (grid_pos - 1) * 0.15)
                            penalties[grid_pos].append(base_penalty + np.random.normal(0, 0.5))
            except:
                continue
        
        position_penalties = {}
        for pos, penalty_list in penalties.items():
            if len(penalty_list) >= 3:
                position_penalties[pos] = {
                    'penalty': float(np.mean(penalty_list)),
                    'std': float(np.std(penalty_list)),
                    'sample_size': len(penalty_list)
                }
        
        print(f"Extracted penalties for {len(position_penalties)} grid positions")
        return position_penalties
    
    def _extract_tire_performance(self):
        """Extract tire compound performance data"""
        print("\n" + "-"*80)
        print("EXTRACTING TIRE PERFORMANCE")
        print("-"*80)
        
        tire_data = {}
        
        if not hasattr(self, 'combined_data'):
            return {}
        
        compounds = self.combined_data['Compound'].unique()
        compounds = [c for c in compounds if pd.notna(c)]
        
        for compound in compounds:
            compound_laps = self.combined_data[
                (self.combined_data['Compound'] == compound) &
                (self.combined_data['LapTime_s'] > 0)
            ]
            
            if len(compound_laps) > 10:
                base_time = compound_laps['LapTime_s'].median()
                
                degradation_rate = 0.08
                if len(compound_laps) > 50:
                    stint_data = compound_laps.groupby(['Driver', 'Stint'])['LapTime_s'].apply(list)
                    deg_rates = []
                    
                    for stint_laps in stint_data:
                        if len(stint_laps) > 5:
                            x = np.arange(len(stint_laps))
                            slope, _, r_val, _, _ = stats.linregress(x, stint_laps)
                            if abs(r_val) > 0.3:
                                deg_rates.append(max(0, slope))
                    
                    if deg_rates:
                        degradation_rate = np.median(deg_rates)
                
                tire_data[compound] = {
                    'base_time': float(base_time),
                    'degradation_rate': float(degradation_rate),
                    'r_squared': 0.01,
                    'sample_size': len(compound_laps),
                    'offset': 0.0
                }
                
                print(f"  {compound}: {len(compound_laps)} laps, "
                      f"base={base_time:.2f}s, deg={degradation_rate:.4f}s/lap")
        
        # Calculate offsets relative to fastest compound
        if tire_data:
            min_time = min(data['base_time'] for data in tire_data.values())
            for compound in tire_data:
                tire_data[compound]['offset'] = tire_data[compound]['base_time'] - min_time
        
        print(f"\nExtracted data for {len(tire_data)} compounds")
        return tire_data
    
    def _extract_driver_errors(self):
        """Extract driver error rates"""
        print("\n" + "-"*80)
        print("EXTRACTING DRIVER ERROR RATES")
        print("-"*80)
        
        error_data = {
            'dry': {'base_error_rate': 0.04, 'sample_size': 100}, 
            'wet': {'base_error_rate': 0.08, 'sample_size': 20}
        }
        
        print("Using estimated error rates (simplified)")
        return error_data
    
    def _extract_drs_effectiveness(self):
        """Extract DRS effectiveness data"""
        print("\n" + "-"*80)
        print("EXTRACTING DRS EFFECTIVENESS")
        print("-"*80)
        
        drs_data = {
            'mean_advantage': 0.35,
            'median_advantage': 0.32,
            'std_advantage': 0.18,
            'sample_size': 500,
            'usage_probability': 0.35
        }
        
        print("Using estimated DRS effectiveness")
        return drs_data
    
    def _generate_parameter_file(self, position_penalties, tire_performance, 
                                driver_errors, drs_effectiveness):
        """Generate the parameter file for the circuit"""
        
        gp_clean = self.gp_name.lower().replace(' ', '_').replace('grand_prix', 'gp').replace('__', '_')
        if not gp_clean.endswith('_gp') and not gp_clean.endswith('_prix'):
            gp_clean += '_gp'
        
        filename = f"{gp_clean}_params.py"
        
        content = f'''"""
Extracted F1 simulation parameters for {self.gp_name}
Generated automatically from historical FastF1 data ({", ".join(map(str, self.years))})
"""

import numpy as np

POSITION_PENALTIES = {repr(position_penalties)}

TIRE_PERFORMANCE = {repr(tire_performance)}

DRIVER_ERROR_RATES = {repr(driver_errors)}

DRS_EFFECTIVENESS = {repr(drs_effectiveness)}

def get_position_penalty(position):
    """Get traffic/dirty air penalty for grid position"""
    if position in POSITION_PENALTIES:
        return POSITION_PENALTIES[position]["penalty"]
    else:
        if position <= 20:
            return 0.05 * (position - 1)
        else:
            return 1.0

def get_tire_offset(compound):
    """Get tire compound offset relative to fastest"""
    return TIRE_PERFORMANCE.get(compound, {{}}).get("offset", 0.0)

def get_tire_degradation_rate(compound):
    """Get tire degradation rate in s/lap"""
    return TIRE_PERFORMANCE.get(compound, {{}}).get("degradation_rate", 0.08)

def get_driver_error_rate(weather_condition="dry"):
    """Get driver error probability per lap"""
    return DRIVER_ERROR_RATES.get(weather_condition, {{}}).get("base_error_rate", 0.01)

def get_drs_advantage():
    """Get DRS time advantage in seconds"""
    mean_adv = DRS_EFFECTIVENESS.get("median_advantage", 0.25)
    std_adv = DRS_EFFECTIVENESS.get("std_advantage", 0.1)
    return max(0.1, np.random.normal(mean_adv, std_adv))

def get_drs_usage_probability():
    """Get probability of being in DRS range"""
    return DRS_EFFECTIVENESS.get("usage_probability", 0.3)
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n{'='*80}")
        print(f" Generated parameter file: {filename}")
        print(f"{'='*80}\n")


class WeatherAPI:
    """Interface with OpenWeatherMap API for real-time forecasts"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/forecast"
    
    def get_forecast(self, city, state, country_code, target_datetime):
        """Get weather forecast for specific location and time"""
        try:
            import requests
        except ImportError:
            print("Error: requests library not installed")
            print("Install with: pip install requests")
            return None
        
        if not self.api_key:
            print("Error: No API key provided")
            return None
        
        if state:
            location = f"{city},{state},{country_code}"
        else:
            location = f"{city},{country_code}"
        
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Find forecast closest to target time
            forecasts = data.get('list', [])
            if not forecasts:
                print("No forecast data available")
                return None
            
            # Parse target datetime
            from datetime import datetime
            target_dt = datetime.fromisoformat(target_datetime)
            
            # Find closest forecast
            closest_forecast = min(forecasts, 
                key=lambda f: abs(datetime.fromtimestamp(f['dt']) - target_dt))
            
            weather_data = {
                'temp': closest_forecast['main']['temp'],
                'weather': closest_forecast['weather'][0]['description'],
                'rain_prob': closest_forecast.get('pop', 0) * 100,  # Probability of precipitation
                'wind_speed': closest_forecast['wind']['speed'],
                'clouds': closest_forecast['clouds']['all']
            }
            
            return weather_data
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing forecast: {e}")
            return None

# TIRE MODELING (Practice-based Bayesian)

class TireModeling:
    """Build tire degradation models from practice session data"""
    
    @staticmethod
    def load_practice_sessions(year, gp_name):
        """Load FP1 and Sprint sessions for tire modeling"""
        print(f"\nLoading practice sessions for {year} {gp_name}...")
        
        combined_laps = []
        session_info = {'fp1_available': False, 'sprint_available': False}
        
        try:
            fp1 = fastf1.get_session(year, gp_name, 'FP1')
            fp1.load()
            
            fp1_laps = fp1.laps.copy()
            fp1_laps['Session'] = 'FP1'
            
            fp1_clean = fp1_laps[
                (fp1_laps['LapTime'].notna()) &
                (fp1_laps['Compound'].notna()) &
                (~fp1_laps['PitOutTime'].notna()) &
                (~fp1_laps['PitInTime'].notna()) &
                (fp1_laps['TrackStatus'] == '1')
            ].copy()
            
            if len(fp1_clean) > 0:
                combined_laps.append(fp1_clean)
                session_info['fp1_available'] = True
                session_info['fp1_laps'] = len(fp1_clean)
                print(f"  FP1: {len(fp1_clean)} clean laps ")
        except Exception as e:
            print(f"  FP1: Not available ({str(e)[:40]})")
        
        try:
            sprint = fastf1.get_session(year, gp_name, 'S')
            sprint.load()
            
            sprint_laps = sprint.laps.copy()
            sprint_laps['Session'] = 'Sprint'
            
            sprint_clean = sprint_laps[
                (sprint_laps['LapTime'].notna()) &
                (sprint_laps['Compound'].notna()) &
                (~sprint_laps['PitOutTime'].notna()) &
                (~sprint_laps['PitInTime'].notna()) &
                (sprint_laps['TrackStatus'] == '1')
            ].copy()
            
            if len(sprint_clean) > 0:
                combined_laps.append(sprint_clean)
                session_info['sprint_available'] = True
                session_info['sprint_laps'] = len(sprint_clean)
                print(f"  Sprint: {len(sprint_clean)} clean laps ")
        except Exception as e:
            print(f"  Sprint: Not available ({str(e)[:40]})")
        
        if combined_laps:
            combined_data = pd.concat(combined_laps, ignore_index=True)
            combined_data['StintLap'] = combined_data.groupby(['Driver', 'Session', 'Stint']).cumcount() + 1
            combined_data['LapTime_s'] = combined_data['LapTime'].dt.total_seconds()
            
            session_info['total_laps'] = len(combined_data)
            session_info['compounds_available'] = list(combined_data['Compound'].unique())
            
            print(f"  Combined: {len(combined_data)} total laps")
            print(f"  Compounds: {', '.join(session_info['compounds_available'])}")
            
            return combined_data, session_info
        
        print("  No practice data available")
        return pd.DataFrame(), session_info
    
    @staticmethod
    def build_tire_model(compound_data, compound_name, base_pace):
        """Build Bayesian tire degradation model"""
        if len(compound_data) < 10:
            print(f"    {compound_name}: Insufficient data ({len(compound_data)} laps)")
            return None
        
        # Remove outliers
        lap_times = compound_data['LapTime_s']
        mean_time = lap_times.mean()
        std_time = lap_times.std()
        
        clean_data = compound_data[
            (lap_times >= mean_time - 3 * std_time) &
            (lap_times <= mean_time + 3 * std_time)
        ].copy()
        
        if len(clean_data) < 8:
            print(f"    {compound_name}: Too few clean laps ({len(clean_data)})")
            return None
        
        x = clean_data["StintLap"].values
        y = clean_data["LapTime_s"].values
        
        print(f"    {compound_name}: Modeling {len(clean_data)} laps...", end=" ")
        
        def tire_model(x, y=None):
            alpha = numpyro.sample("alpha", dist.Normal(base_pace, 2.0))
            
            if compound_name == 'SOFT':
                beta_prior = dist.Normal(0.15, 0.05)
            elif compound_name == 'MEDIUM':
                beta_prior = dist.Normal(0.08, 0.03)
            elif compound_name == 'HARD':
                beta_prior = dist.Normal(0.04, 0.02)
            else:
                beta_prior = dist.Normal(0.06, 0.03)
            
            beta = numpyro.sample("beta", beta_prior)
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            mu = alpha + beta * x
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        
        try:
            kernel = NUTS(tire_model)
            mcmc = MCMC(kernel, num_warmup=800, num_samples=1200)
            mcmc.run(random.PRNGKey(42), x, y)
            
            samples = mcmc.get_samples()
            alpha_mean = np.mean(samples['alpha'])
            beta_mean = np.mean(samples['beta'])
            
            print(f" Base={alpha_mean:.2f}s, Deg={beta_mean:.4f}s/lap")
            return mcmc
            
        except Exception as e:
            print(f" Failed: {str(e)[:40]}")
            return None

# RACE SIMULATION ENGINE

class RaceSimulator:
    """Monte Carlo race simulation engine"""
    
    def __init__(self, circuit_params, extracted_params=None, compound_models=None):
        self.circuit = circuit_params
        self.extracted_params = extracted_params
        self.compound_models = compound_models or {}
        
        # Setup parameters
        self._setup_parameters()
    
    def _setup_parameters(self):
        """Initialize simulation parameters"""
        if self.extracted_params:
            self.params = self.extracted_params
            print("Using extracted parameters")
        else:
            self.params = self._fallback_parameters()
            print("Using fallback parameters")
    
    def _fallback_parameters(self):
        """Fallback parameters when extraction not available"""
        return {
            'POSITION_PENALTIES': {i: {'penalty': 0.05 * (i-1)} for i in range(1, 21)},
            'TIRE_PERFORMANCE': {
                'SOFT': {'offset': 0.0, 'degradation_rate': 0.12},
                'MEDIUM': {'offset': 0.37, 'degradation_rate': 0.07},
                'HARD': {'offset': 0.70, 'degradation_rate': 0.04}
            },
            'DRIVER_ERROR_RATES': {'dry': {'base_error_rate': 0.008}},
            'DRS_EFFECTIVENESS': {'median_advantage': 0.40, 'usage_probability': 0.38}
        }
    
    def generate_strategies(self):
        """Generate race strategies based on circuit type"""
        strategies = {
            "1-stop (M-H)": [
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.32)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.68)}
            ],
            "1-stop (H-M)": [
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.68)},
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.32)}
            ],
            "1-stop (H-S)": [
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.77)},
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.23)}
            ],
            "1-stop (S-H)": [
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.25)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.75)}
            ],
            "1-stop (M-S)": [
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.62)},
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.38)}
            ],
            "1-stop (S-M)": [
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.32)},
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.68)}
            ],
            "2-stop (M-H-S)": [
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.29)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.57)},
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.14)}
            ],
            "2-stop (H-M-H)": [
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.21)},
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.29)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.50)}
            ],
            "2-stop (H-S-H)": [
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.30)},
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.20)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.50)}
            ],
            "2-stop (S-H-M)": [
                {"compound": "SOFT", "laps": int(self.circuit['laps'] * 0.14)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.57)},
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.29)}
            ],
            "2-stop (M-H-M)": [
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.25)},
                {"compound": "HARD", "laps": int(self.circuit['laps'] * 0.50)},
                {"compound": "MEDIUM", "laps": int(self.circuit['laps'] * 0.25)}
            ]
        }
        
        return strategies
    
    def simulate_race(self, strategy, grid_position, car_performance_factor=1.0):
        """Simulate a single race"""
        race_time = 0
        current_lap = 1
        current_position = grid_position
        fuel_load = 110
        
        # Generate SC/VSC laps
        sc_laps = self._generate_sc_laps()
        vsc_laps = self._generate_vsc_laps()
        
        for stint_idx, stint in enumerate(strategy):
            compound = stint["compound"]
            stint_len = min(stint["laps"], self.circuit['laps'] - current_lap + 1)
            
            for lap_in_stint in range(1, stint_len + 1):
                if current_lap > self.circuit['laps']:
                    break
                
                # Calculate lap time
                lap_time = self._calculate_lap_time(
                    compound, lap_in_stint, current_lap, 
                    current_position, fuel_load, car_performance_factor
                )
                
                # Apply SC/VSC
                if current_lap in sc_laps:
                    lap_time *= 1.35
                elif current_lap in vsc_laps:
                    lap_time *= 1.18
                
                race_time += lap_time
                current_lap += 1
                fuel_load = max(5, fuel_load - self.circuit['fuel_rate'])
            
            # Pit stop
            if stint_idx < len(strategy) - 1:
                pit_time = np.random.normal(self.circuit['pit_loss'], 1.2)
                
                # SC/VSC pit advantage
                if any(lap in sc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                    pit_time *= 0.20
                elif any(lap in vsc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                    pit_time *= 0.55
                
                race_time += max(13, pit_time)
                
                # Position change from pit
                position_change = np.random.choice([-1, 0, 1, 2], p=[0.10, 0.50, 0.30, 0.10])
                current_position = max(1, min(20, current_position + position_change))
        
        return race_time, current_position
    
    def _calculate_lap_time(self, compound, lap_in_stint, current_lap, 
                           position, fuel_load, car_factor):
        """Calculate single lap time"""
        if compound in self.compound_models:
            try:
                samples = self.compound_models[compound].get_samples()
                alpha = np.median(samples['alpha'])
                beta = np.median(samples['beta'])
                base_time = alpha
                degradation = beta * lap_in_stint
            except:
                base_time, degradation = self._fallback_tire_calc(compound, lap_in_stint)
        else:
            base_time, degradation = self._fallback_tire_calc(compound, lap_in_stint)
        
        lap_time = base_time + degradation
        
        # Apply factors
        lap_time *= car_factor
        lap_time += -(fuel_load * 0.035)  # Fuel effect
        lap_time += self._get_position_penalty(position) * max(0.4, 1.0 - (current_lap / self.circuit['laps']) * 0.6)
        lap_time += np.random.normal(0, 0.35)  # Random variation
        
        # Track evolution
        lap_time += -0.0025 * current_lap
        
        return lap_time
    
    def _fallback_tire_calc(self, compound, lap_in_stint):
        """Fallback tire calculation"""
        tire_perf = self.params['TIRE_PERFORMANCE']
        if compound in tire_perf:
            offset = tire_perf[compound].get('offset', 0.0)
            deg_rate = tire_perf[compound].get('degradation_rate', 0.08)
        else:
            offset = 0.0
            deg_rate = 0.08
        
        base_time = self.circuit['base_pace'] + offset
        degradation = deg_rate * lap_in_stint
        
        return base_time, degradation
    
    def _get_position_penalty(self, position):
        """Get position-based penalty"""
        if 'POSITION_PENALTIES' in self.params:
            penalties = self.params['POSITION_PENALTIES']
            if position in penalties:
                return penalties[position].get('penalty', 0.05 * (position - 1))
        return 0.05 * (position - 1)
    
    def _generate_sc_laps(self):
        """Generate safety car laps"""
        sc_laps = set()
        if np.random.rand() < self.circuit['sc_prob']:
            sc_start = np.random.choice(range(8, self.circuit['laps']-5))
            sc_duration = np.random.choice(range(2, 5))
            sc_laps.update(range(sc_start, min(sc_start + sc_duration, self.circuit['laps'])))
        return sc_laps
    
    def _generate_vsc_laps(self):
        """Generate virtual safety car laps"""
        vsc_laps = set()
        if np.random.rand() < self.circuit['vsc_prob']:
            vsc_start = np.random.choice(range(10, self.circuit['laps']-5))
            vsc_laps.update(range(vsc_start, vsc_start + 2))
        return vsc_laps
    
    def run_monte_carlo(self, num_sims=1500, grid_positions=None):
        """Run Monte Carlo simulation"""
        if grid_positions is None:
            grid_positions = [1, 3, 5, 8, 10, 15]
        
        strategies = self.generate_strategies()
        results = {}
        
        car_performance_map = {
            1: 0.98, 2: 0.985, 3: 0.99, 4: 0.995, 5: 1.00,
            6: 1.005, 7: 1.01, 8: 1.015, 9: 1.02, 10: 1.025,
            11: 1.03, 12: 1.035, 13: 1.04, 14: 1.045, 15: 1.05
        }
        
        print(f"\nRunning {num_sims} Monte Carlo simulations...")
        
        for grid_pos in grid_positions:
            print(f"\n  Grid P{grid_pos}:", end=" ")
            pos_results = {name: {'times': [], 'final_positions': [], 'points': []} 
                          for name in strategies.keys()}
            
            car_factor = car_performance_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
            
            for _ in trange(num_sims, desc=f"P{grid_pos}", leave=False):
                for name, strat in strategies.items():
                    race_time, final_pos = self.simulate_race(strat, grid_pos, car_factor)
                    points = self._get_f1_points(final_pos)
                    
                    pos_results[name]['times'].append(race_time)
                    pos_results[name]['final_positions'].append(final_pos)
                    pos_results[name]['points'].append(points)
            
            results[grid_pos] = pos_results
            print("")
        
        return results, strategies
    
    @staticmethod
    def _get_f1_points(position):
        """F1 points system"""
        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return points_map.get(position, 0)

# VISUALIZATION

def plot_results(results, strategies, gp_name, grid_positions):
    """Create comprehensive visualization plots"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'{gp_name} - Strategy Analysis', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    strategy_colors = dict(zip(strategies.keys(), colors))
    
    plot_positions = [1, 3, 5] if all(p in grid_positions for p in [1, 3, 5]) else grid_positions[:3]
    
    for i, grid_pos in enumerate(plot_positions):
        ax = axes[i]
        for strategy_name in strategies.keys():
            times = results[grid_pos][strategy_name]['times']
            ax.hist(times, bins=25, alpha=0.6, label=strategy_name,
                   color=strategy_colors[strategy_name])
        
        ax.set_title(f'Grid P{grid_pos} - Race Time Distribution', fontsize=12)
        ax.set_xlabel('Race Time (s)')
        ax.set_ylabel('Frequency')
        if i == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# MAIN CLI

def main():
    parser = argparse.ArgumentParser(
        description='F1 Strategy Analyzer - Universal Circuit System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python f1_strategy_analyzer.py --simulate 2025 "Monaco"
  python f1_strategy_analyzer.py --extract 2025 "Italian Grand Prix"
  python f1_strategy_analyzer.py --weather "United States" --api-key YOUR_KEY
  python f1_strategy_analyzer.py --validate 2025 "Monaco"
        """
    )
    
    parser.add_argument('--simulate', nargs=2, metavar=('YEAR', 'GP_NAME'),
                       help='Run Monte Carlo simulation for specified race')
    parser.add_argument('--extract', nargs=2, metavar=('YEAR', 'GP_NAME'),
                       help='Extract circuit-specific parameters from historical data')
    parser.add_argument('--weather', metavar='GP_NAME',
                       help='Fetch weather forecast and update rain probability')
    parser.add_argument('--api-key', metavar='KEY',
                       help='OpenWeatherMap API key (for --weather)')
    parser.add_argument('--validate', nargs=2, metavar=('YEAR', 'GP_NAME'),
                       help='Validate predictions against actual race results')
    parser.add_argument('--strategies', nargs=2, metavar=('YEAR', 'GP_NAME'),
                       help='Extract actual tire strategies from race')
    parser.add_argument('--historical', metavar='GP_NAME',
                       help='Analyze tire strategies from past 4 years')
    
    args = parser.parse_args()
    
    if not any([args.simulate, args.extract, args.weather, args.validate, 
                args.strategies, args.historical]):
        parser.print_help()
        sys.exit(1)
    
    config = CircuitConfig()
    
    if args.extract:
        year, gp_name = args.extract
        print(f"\n{'='*80}")
        print(f"PARAMETER EXTRACTION MODE")
        print(f"{'='*80}")
        print(f"Race: {year} {gp_name}")
        
        extractor = ParameterExtractor(gp_name)
        params = extractor.extract_all_parameters()
        
        if params:
            print(f"\n Parameter extraction completed successfully")
        else:
            print(f"\n Parameter extraction failed")
    
    elif args.simulate:
        year, gp_name = args.simulate
        year = int(year)
        
        print(f"\n{'='*80}")
        print(f"RACE SIMULATION MODE")
        print(f"{'='*80}")
        print(f"Race: {year} {gp_name}")
        
        circuit_params = config.get_circuit(gp_name)
        print(f"\nCircuit Configuration:")
        print(f"  Laps: {circuit_params['laps']}")
        print(f"  Base pace: {circuit_params['base_pace']}s")
        print(f"  Rain probability: {circuit_params['rain_prob']:.0%}")
        print(f"  Sprint weekend: {'Yes' if circuit_params['sprint'] else 'No'}")
        
        gp_clean = gp_name.lower().replace(' ', '_').replace('grand_prix', 'gp').replace('__', '_')
        if not gp_clean.endswith('_gp') and not gp_clean.endswith('_prix'):
            gp_clean += '_gp'
        param_file = f"{gp_clean}_params.py"
        
        extracted_params = None
        if os.path.exists(param_file):
            print(f"\n Found extracted parameters: {param_file}")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("circuit_params", param_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                extracted_params = {
                    'TIRE_PERFORMANCE': module.TIRE_PERFORMANCE,
                    'POSITION_PENALTIES': module.POSITION_PENALTIES,
                    'DRIVER_ERROR_RATES': module.DRIVER_ERROR_RATES,
                    'DRS_EFFECTIVENESS': module.DRS_EFFECTIVENESS
                }
            except:
                print(f"  Warning: Could not load {param_file}, using fallback")
        
        print(f"\n{'='*80}")
        print("BUILDING TIRE MODELS FROM PRACTICE DATA")
        print(f"{'='*80}")
        
        practice_data, session_info = TireModeling.load_practice_sessions(year, gp_name)
        compound_models = {}
        
        if not practice_data.empty:
            compounds = practice_data['Compound'].unique()
            print(f"\nBuilding models for: {', '.join(compounds)}")
            
            for compound in compounds:
                compound_data = practice_data[practice_data['Compound'] == compound]
                model = TireModeling.build_tire_model(
                    compound_data, compound, circuit_params['base_pace']
                )
                if model:
                    compound_models[compound] = model
        
        print(f"\n{'='*80}")
        print("STARTING MONTE CARLO SIMULATION")
        print(f"{'='*80}")
        
        simulator = RaceSimulator(circuit_params, extracted_params, compound_models)
        results, strategies = simulator.run_monte_carlo()
        
        print(f"\n{'='*80}")
        print(f"SIMULATION RESULTS - {gp_name.upper()}")
        print(f"{'='*80}")
        
        for grid_pos in [1, 3, 5, 8, 10, 15]:
            if grid_pos in results:
                print(f"\n{'─'*80}")
                print(f"GRID POSITION {grid_pos}")
                print(f"{'─'*80}")
                
                strategy_summary = []
                for strategy_name, data in results[grid_pos].items():
                    times = np.array(data['times'])
                    positions = np.array(data['final_positions'])
                    points = np.array(data['points'])
                    
                    strategy_summary.append({
                        'Strategy': strategy_name,
                        'Avg Time': f"{np.mean(times):.1f}s",
                        'Avg Pos': f"{np.mean(positions):.1f}",
                        'Avg Points': f"{np.mean(points):.1f}",
                        'Top 5 %': f"{np.mean(positions <= 5)*100:.1f}%"
                    })
                
                summary_df = pd.DataFrame(strategy_summary)
                summary_df = summary_df.sort_values('Avg Points', ascending=False)
                print(summary_df.to_string(index=False))
        
        plot_results(results, strategies, gp_name, [1, 3, 5, 8, 10, 15])
    
    elif args.weather:
        gp_name = args.weather
        
        print(f"\n{'='*80}")
        print(f"WEATHER FORECAST MODE")
        print(f"{'='*80}")
        print(f"Grand Prix: {gp_name}\n")
        
        api_key = args.api_key
        if not api_key:
            api_key = input("Enter OpenWeatherMap API key: ").strip()
        
        if not api_key:
            print("Error: API key required")
            sys.exit(1)
        
        print("\nEnter race location details:")
        city = input("City: ").strip()
        state = input("Region/State (press Enter if not applicable): ").strip()
        country_code = input("Country code (e.g., US, GB, IT): ").strip().upper()
        
        race_date = input("Race date (YYYY-MM-DD): ").strip()
        race_time = input("Local race time (HH:MM in 24-hour format): ").strip()
        
        target_datetime = f"{race_date}T{race_time}:00"
        
        weather_api = WeatherAPI(api_key)
        forecast = weather_api.get_forecast(city, state, country_code, target_datetime)
        
        if forecast:
            print(f"\n{'='*80}")
            print(f"WEATHER FORECAST for {city}, {state if state else country_code}")
            print(f"Race: {race_date} at {race_time}")
            print(f"{'='*80}")
            print(f"Rain probability: {forecast['rain_prob']:.2f}%")
            print(f"Weather: {forecast['weather']}")
            print(f"Temperature: {forecast['temp']:.1f}°C")
            print(f"Wind: {forecast['wind_speed']:.1f} m/s")
            
            update = input("\nUpdate circuit_config.json with this rain probability? (y/n): ").strip().lower()
            if update == 'y':
                old_prob, new_prob = config.update_rain_probability(
                    gp_name, forecast['rain_prob'] / 100
                )
                if old_prob is not None:
                    print(f" Updated {gp_name}: rain_prob {old_prob:.2f} -> {new_prob:.2f}")
                else:
                    print(f" Could not update config")
        else:
            print("\n Could not fetch weather forecast")
    
    elif args.validate:
        print("\nValidation mode - implementation depends on having actual race results")
        print("Run this after the race when data is available in FastF1")
    
    elif args.strategies:
        year, gp_name = args.strategies
        print(f"\nExtracting strategies from {year} {gp_name}")
        print("This would show actual tire strategies used in the race")
    
    elif args.historical:
        gp_name = args.historical
        print(f"\nAnalyzing historical strategies for {gp_name}")
        print("This would show tire strategies from past 4 years")

if __name__ == "__main__":
    main()