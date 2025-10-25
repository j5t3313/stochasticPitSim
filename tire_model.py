import fastf1
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import warnings

warnings.filterwarnings('ignore')

def load_practice(year, gp_name):
    print(f"Loading practice sessions for {year} {gp_name}...")
    
    combined = []
    info = {'fp1': False, 'sprint': False, 'total': 0}
    
    try:
        fp1 = fastf1.get_session(year, gp_name, 'FP1')
        fp1.load()
        
        fp1_laps = fp1.laps.copy()
        fp1_laps['Session'] = 'FP1'
        fp1_laps['Type'] = 'Practice'
        
        fp1_clean = fp1_laps[
            (fp1_laps['LapTime'].notna()) &
            (fp1_laps['Compound'].notna()) &
            (~fp1_laps['PitOutTime'].notna()) &
            (~fp1_laps['PitInTime'].notna()) &
            (fp1_laps['TrackStatus'] == '1')
        ].copy()
        
        if len(fp1_clean) > 0:
            combined.append(fp1_clean)
            info['fp1'] = True
            info['fp1_laps'] = len(fp1_clean)
            print(f"   FP1: {len(fp1_clean)} laps")
        else:
            print(f"   FP1: No clean laps")
            
    except Exception as e:
        print(f"  FP1: Could not load - {e}")
    
    try:
        sprint = fastf1.get_session(year, gp_name, 'S')
        sprint.load()
        
        sprint_laps = sprint.laps.copy()
        sprint_laps['Session'] = 'Sprint'
        sprint_laps['Type'] = 'Race'
        
        sprint_clean = sprint_laps[
            (sprint_laps['LapTime'].notna()) &
            (sprint_laps['Compound'].notna()) &
            (~sprint_laps['PitOutTime'].notna()) &
            (~sprint_laps['PitInTime'].notna()) &
            (sprint_laps['TrackStatus'] == '1')
        ].copy()
        
        if len(sprint_clean) > 0:
            combined.append(sprint_clean)
            info['sprint'] = True
            info['sprint_laps'] = len(sprint_clean)
            print(f"   Sprint: {len(sprint_clean)} laps")
        else:
            print(f"   Sprint: No clean laps")
            
    except Exception as e:
        print(f"   Sprint: Could not load - {e}")
    
    if combined:
        data = pd.concat(combined, ignore_index=True)
        
        data['StintLap'] = data.groupby(['Driver', 'Session', 'Stint']).cumcount() + 1
        data['LapTime_s'] = data['LapTime'].dt.total_seconds()
        
        info['total'] = len(data)
        info['comps'] = list(data['Compound'].unique())
        
        print(f"   Combined: {len(data)} total laps")
        print(f"   Compounds: {', '.join(info['comps'])}")
        
        return data, info
    else:
        print(f"   No data available")
        return pd.DataFrame(), info

def build_model(comp_data, comp_name, base_pace=96.0):
    if len(comp_data) < 10:
        print(f"     {comp_name}: Only {len(comp_data)} laps - insufficient")
        return None
    
    lap_times = comp_data['LapTime_s']
    mean_time = lap_times.mean()
    std_time = lap_times.std()
    
    clean = comp_data[
        (lap_times >= mean_time - 3 * std_time) &
        (lap_times <= mean_time + 3 * std_time)
    ].copy()
    
    if len(clean) < 8:
        print(f"     {comp_name}: Only {len(clean)} clean laps after outlier removal")
        return None
    
    x = clean["StintLap"].values
    y = clean["LapTime_s"].values
    
    print(f"     {comp_name}: Modeling {len(clean)} laps (stint laps 1-{max(x)})")
    
    def tire_model(x, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(base_pace, 2.0))
        
        if comp_name == 'SOFT':
            beta_prior = dist.Normal(0.15, 0.05)
        elif comp_name == 'MEDIUM':
            beta_prior = dist.Normal(0.08, 0.03)
        elif comp_name == 'HARD':
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
        
        print(f"     {comp_name}: Base={alpha_mean:.2f}s, Deg={beta_mean:.4f}s/lap")
        
        return mcmc
        
    except Exception as e:
        print(f"     {comp_name}: Model fitting failed - {e}")
        return None

def build_models(year, gp_name, base_pace=96.0):
    print(f"\n{'='*60}")
    print(f"BUILDING TIRE MODELS FROM PRACTICE")
    print(f"{'='*60}")
    
    practice, info = load_practice(year, gp_name)
    
    if practice.empty:
        print(" No practice data available")
        return {}, {'error': 'No practice data'}
    
    models = {}
    model_info = {
        'info': info,
        'modeled': [],
        'failed': [],
        'total': len(practice),
        'quality': {}
    }
    
    compounds = practice['Compound'].unique()
    print(f"\n Building models for: {', '.join(compounds)}")
    
    for comp in compounds:
        print(f"\n   Processing {comp}...")
        
        comp_data = practice[practice['Compound'] == comp].copy()
        
        fp1_laps = len(comp_data[comp_data['Session'] == 'FP1'])
        sprint_laps = len(comp_data[comp_data['Session'] == 'Sprint'])
        
        print(f"     Data: FP1={fp1_laps}, Sprint={sprint_laps}, Total={len(comp_data)}")
        
        mcmc = build_model(comp_data, comp, base_pace)
        
        if mcmc is not None:
            models[comp] = mcmc
            model_info['modeled'].append(comp)
            
            samples = mcmc.get_samples()
            alpha_std = np.std(samples['alpha'])
            beta_std = np.std(samples['beta'])
            
            if alpha_std < 0.5 and beta_std < 0.01:
                quality = 'High'
            elif alpha_std < 1.0 and beta_std < 0.02:
                quality = 'Good'
            else:
                quality = 'Moderate'
            
            model_info['quality'][comp] = {
                'quality': quality,
                'alpha_unc': alpha_std,
                'beta_unc': beta_std,
                'n': len(comp_data)
            }
            
        else:
            model_info['failed'].append(comp)
    
    print(f"\n{'='*40}")
    print(f"TIRE MODEL SUMMARY")
    print(f"{'='*40}")
    
    print(f" Modeled: {', '.join(model_info['modeled'])}")
    if model_info['failed']:
        print(f" Failed: {', '.join(model_info['failed'])}")
    
    print(f"\n Model Quality:")
    for comp, q_info in model_info['quality'].items():
        quality = q_info['quality']
        n = q_info['n']
        print(f"  {comp}: {quality} ({n} points)")
    
    print(f"\n Data Sources:")
    if info['fp1']:
        print(f"  FP1: {info.get('fp1_laps', 0)} laps")
    if info['sprint']:
        print(f"  Sprint: {info.get('sprint_laps', 0)} laps")
    print(f"  Total: {info['total']} laps")
    
    return models, model_info

def get_tire_perf(comp, lap, models, base_pace=96.0, weather='dry', 
                  track_evo=0, params=None):
    if weather == 'wet':
        if comp == 'INTERMEDIATE':
            base_time = base_pace + 9.0
            deg = 0.03 * lap
        else:
            base_time = base_pace + 15.0
            deg = 0.1 * lap
            
        return base_time + deg + track_evo
    
    if comp in models:
        try:
            samples = models[comp].get_samples()
            
            alpha = np.median(samples['alpha'])
            beta = np.median(samples['beta'])
            
            base_time = alpha
            deg = beta * lap
            
        except Exception as e:
            print(f"Warning: Could not use model for {comp}: {e}")
            base_time, deg = _fallback_tire(comp, lap, base_pace, params)
    else:
        base_time, deg = _fallback_tire(comp, lap, base_pace, params)
    
    base_time += track_evo
    
    if lap > 25:
        deg += 0.04 * (lap - 25) ** 1.3
    
    return base_time + deg

def _fallback_tire(comp, lap, base_pace, params=None):
    if params and 'TIRE_PERF' in params:
        tire_perf = params['TIRE_PERF']
        if comp in tire_perf:
            offset = tire_perf[comp].get('offset', 0.0)
            deg_rate = tire_perf[comp].get('deg', 0.08)
            base_time = base_pace + offset
            deg = deg_rate * lap
            return base_time, deg
    
    offsets = {'SOFT': 0.0, 'MEDIUM': 0.35, 'HARD': 0.65}
    deg_rates = {'SOFT': 0.15, 'MEDIUM': 0.08, 'HARD': 0.04}
    base_time = base_pace + offsets.get(comp, 0.0)
    deg = deg_rates.get(comp, 0.08) * lap
    
    return base_time, deg

def main():
    year = 2025
    gp_name = 'United States Grand Prix'
    base_pace = 96
    
    models, info = build_models(year, gp_name, base_pace)
    
    if models:
        print(f"\n{'='*50}")
        print("TESTING PRACTICE-BASED MODELS")
        print('='*50)
        
        test_comps = ['SOFT', 'MEDIUM', 'HARD']
        test_laps = [1, 10, 20, 30]
        
        for comp in test_comps:
            if comp in models:
                print(f"\n{comp} Predictions:")
                for lap in test_laps:
                    time = get_tire_perf(comp, lap, models, base_pace)
                    print(f"  Lap {lap}: {time:.3f}s")
            else:
                print(f"\n{comp}: No model available")
    
    else:
        print(" No tire models built")

if __name__ == "__main__":
    main()
