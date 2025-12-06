F1 STRATEGY SIMULATOR - UNIVERSAL CIRCUIT SYSTEM
================================================================================

OVERVIEW
--------
F1 pit strategy simulation system that works for any circuit on the
2025 F1 calendar. 

KEY FEATURES
------------
✓ Works for all 24 circuits on 2025 F1 calendar
✓ Automatic sprint weekend detection
✓ Real-time weather API integration (OpenWeather)
✓ Historical parameter extraction from FastF1 data
✓ Practice session (FP1/FP2) tire modeling with Bayesian inference
✓ Monte Carlo simulation with 1500+ iterations
✓ Laptime target thresholds for optimal strategy selection
✓ Comprehensive strategy analysis and visualization

QUICK START
================================================================================

BASIC USAGE (No Weather, Fallback Parameters)
----------------------------------------------
python pit_sim.py 2025 "Monaco"
python pit_sim.py 2025 "United States"
python f1_strategy_analyzer.py --simulate 2025 "Monaco"

WITH EXTRACTED PARAMETERS (More Accurate)
------------------------------------------
# Step 1: Extract circuit-specific parameters from historical data
python f1_strategy_analyzer.py --extract 2025 "Monaco"

# Step 2: Run simulation with extracted parameters
python pit_sim.py 2025 "Monaco"

WITH WEATHER UPDATES
---------------------
python f1_strategy_analyzer.py --weather "Monaco" --api-key YOUR_API_KEY

WITH LAPTIME TARGETS
---------------------
python pit_sim.py 2025 "Monaco" --targets
python laptime_targets.py 2025 "Monaco" --compound SOFT --position 3

PROJECT STRUCTURE
================================================================================

├── circuit_config.json      # Circuit parameters (laps, pit loss, SC probability)
├── pit_sim.py               # Main simulation engine
├── tire_model.py            # Bayesian tire degradation modeling
├── laptime_targets.py       # Laptime threshold calculator
├── param_extractor.py       # Extract parameters from historical data
├── validation.py            # Compare predictions to actual results
├── f1_strategy_analyzer.py  # Full analysis pipeline
└── weather_rain_probability.py  # Weather forecast integration

COMMAND REFERENCE
================================================================================

pit_sim.py
----------
Primary simulation tool with strategy analysis.

python pit_sim.py <year> <gp_name> [options]

Options:
  --sensitivity <strategy>  Analyze pace thresholds for a strategy
  --targets [compound]      Calculate laptime targets for optimal strategy
  --position <1-20>         Grid position for targets (default: 1)

Examples:
  python pit_sim.py 2025 "Monaco"
  python pit_sim.py 2025 "United States" --sensitivity "1-stop (S-H)"
  python pit_sim.py 2025 "Monaco" --targets
  python pit_sim.py 2025 "Monaco" --targets SOFT --position 3

laptime_targets.py
------------------
Standalone laptime threshold analysis.

python laptime_targets.py <year> <gp_name> [options]

Options:
  --compound <SOFT|MEDIUM|HARD>  Analyze specific compound
  --position <1-20>              Grid position (default: 1)
  --sims <n>                     Simulations per comparison (default: 100)

Examples:
  python laptime_targets.py 2025 "United States"
  python laptime_targets.py 2025 "Monaco" --compound SOFT --position 3 --sims 150

f1_strategy_analyzer.py
-----------------------

--extract YEAR "GP_NAME"
    Extract circuit-specific parameters from 3 years of historical data.
    Creates a {circuit}_params.py file with:
    - Position penalties
    - Tire performance data
    - Driver error rates
    - DRS effectiveness
    
    Example:
    python f1_strategy_analyzer.py --extract 2025 "Monaco"

--simulate YEAR "GP_NAME"
    Run Monte Carlo simulation (1500 iterations) for the specified race.
    Automatically uses extracted parameters if available, otherwise uses
    fallback parameters.
    
    Example:
    python f1_strategy_analyzer.py --simulate 2025 "Monaco"

--weather "GP_NAME" [--api-key KEY]
    Fetch real-time weather forecast and update rain probability.
    Requires free OpenWeatherMap API key.
    
    Example:
    python f1_strategy_analyzer.py --weather "Monaco" --api-key abc123
    
    If --api-key not provided, will prompt for it interactively.

--validate YEAR "GP_NAME"
    Validate model predictions against actual race results.
    Only works after the race has occurred and data is available.
    
    Example:
    python f1_strategy_analyzer.py --validate 2025 "Monaco"

--strategies YEAR "GP_NAME"
    Extract and display actual tire strategies used in the race.
    Requires race data to be available in FastF1.
    
    Example:
    python f1_strategy_analyzer.py --strategies 2025 "Monaco"

--historical "GP_NAME"
    Analyze tire strategies from past 4 years (2021-2024).
    
    Example:
    python f1_strategy_analyzer.py --historical "Monaco"

LAPTIME TARGET ANALYSIS
================================================================================

OVERVIEW
--------
The laptime target system calculates the threshold laptimes above which
(slower than) a strategy or compound becomes suboptimal. This answers the
question: "How slow can my tires be before I should switch strategies?"

HOW IT WORKS
------------
1. Takes current tire model pace as baseline
2. Simulates pairwise strategy comparisons
3. Uses binary search to find crossover points
4. Reports threshold laptimes where strategy preference changes

OUTPUT EXAMPLE
--------------
SOFT COMPOUND
------------------------------------------------------------
Current pace (lap 10): 73.45s

Current best strategy: 1-stop (S-H)
  Avg time: 5765.2s, Avg pos: 2.3

Switch from 1-stop (S-H) when SOFT is slower than:
  73.89s/lap: switch to 1-stop (M-H)
  74.12s/lap: switch to 2-stop (S-H-M)
  74.45s/lap: switch to 1-stop (H-M)

INTERPRETATION
--------------
- If SOFT is running at 73.45s, stay with 1-stop (S-H)
- If SOFT degrades to 73.89s or slower, switch to 1-stop (M-H)
- Thresholds account for full race simulation including pit stops

SENSITIVITY TABLE
-----------------
The --targets option also produces a sensitivity table showing which
strategy is optimal at each pace level:

Laptime      Best Strategy         Avg Pos    Avg Pts
----------------------------------------------------
72.45s/lap   1-stop (S-H)          2.1        15.2
72.95s/lap   1-stop (S-H)          2.3        14.8
73.45s/lap   1-stop (S-H)          2.5        14.1
73.95s/lap   1-stop (M-H)          2.8        13.2
74.45s/lap   1-stop (M-H)          3.1        12.1

PROGRAMMATIC USAGE
------------------
from laptime_targets import LaptimeTargetCalculator

calculator = LaptimeTargetCalculator(circuit, models, params)

# Get all crossover thresholds
thresholds = calculator.calculate_all_thresholds(grid_pos=1, n_sims=100)

# Generate sensitivity table for a compound
sensitivity = calculator.generate_pace_sensitivity_table(
    compound='SOFT', 
    grid_pos=1, 
    pace_range=(-1.5, 1.5), 
    steps=7
)

# Find best strategy given specific compound paces
best, all_results = calculator.find_best_strategy_for_pace(
    compound_paces={'SOFT': 74.0, 'MEDIUM': 74.5, 'HARD': 75.2},
    grid_pos=1
)

CIRCUIT CONFIGURATION
================================================================================

All circuit parameters are stored in circuit_config.json:

- laps: Race distance in laps
- base_pace: Baseline lap time in seconds (based on typical race pace)
- pit_loss: Pit stop time loss in seconds
- rain_prob: Historical or forecast rain probability (0.0 - 1.0)
- sc_prob: Safety car probability (0.0 - 1.0)
- vsc_prob: Virtual safety car probability (0.0 - 1.0)
- fuel_rate: Fuel consumption per lap in kg
- fuel_effect: Lap time gain per kg of fuel burned
- sprint: Boolean flag for sprint weekend format

SUPPORTED CIRCUITS (2025)
--------------------------
Standard Weekends (18):
  • Australian Grand Prix (Melbourne)
  • Bahrain Grand Prix (Sakhir)
  • Saudi Arabian Grand Prix (Jeddah)
  • Japanese Grand Prix (Suzuka)
  • Emilia Romagna Grand Prix (Imola)
  • Monaco Grand Prix
  • Canadian Grand Prix (Montreal)
  • Spanish Grand Prix (Barcelona)
  • Austrian Grand Prix (Red Bull Ring)
  • British Grand Prix (Silverstone)
  • Hungarian Grand Prix (Hungaroring)
  • Dutch Grand Prix (Zandvoort)
  • Italian Grand Prix (Monza)
  • Azerbaijan Grand Prix (Baku)
  • Singapore Grand Prix
  • Mexican Grand Prix (Mexico City)
  • Las Vegas Grand Prix
  • Abu Dhabi Grand Prix (Yas Marina)

Sprint Weekends (6):
  • Chinese Grand Prix (Shanghai) 🏁
  • Miami Grand Prix 🏁
  • Belgian Grand Prix (Spa) 🏁
  • United States Grand Prix (Austin) 🏁
  • São Paulo Grand Prix (Interlagos) 🏁
  • Qatar Grand Prix (Lusail) 🏁

WEATHER API INTEGRATION
================================================================================

SETUP
-----
Get FREE API key from: https://openweathermap.org/api

USAGE
-----
python f1_strategy_analyzer.py --weather "United States" --api-key YOUR_KEY

The script will prompt for:
  • City name (e.g., "Austin")
  • Region/State (e.g., "Texas")
  • Country code (e.g., "US")
  • Race date (YYYY-MM-DD)
  • Local race time (HH:MM in 24-hour format)

OUTPUT
------
The script will display:
  • Rain probability percentage
  • Weather conditions (clear, clouds, rain, etc.)
  • Temperature
  • Option to automatically update circuit_config.json

EXAMPLE SESSION
---------------
$ python f1_strategy_analyzer.py --weather "United States" --api-key abc123

Enter race location details:
City: Austin
Region/State: Texas
Country code: US
Race date (YYYY-MM-DD): 2025-10-19
Local race time (HH:MM): 14:00

Fetching weather forecast...

WEATHER FORECAST for Austin, Texas, US
Race: October 19, 2025 at 14:00
========================================
Rain probability: 12.00%
Weather: scattered clouds
Temperature: 24.5°C
Wind: 4.2 m/s

Update circuit_config.json? (y/n): y
✓ Updated United States Grand Prix: rain_prob 0.10 → 0.12

COMMON COUNTRY CODES
--------------------
US (United States)    MC (Monaco)         IT (Italy)
GB (Great Britain)    BE (Belgium)        NL (Netherlands)
ES (Spain)            AT (Austria)        JP (Japan)
CN (China)            SG (Singapore)      AE (UAE)
BH (Bahrain)          SA (Saudi Arabia)   AU (Australia)
BR (Brazil)           MX (Mexico)         CA (Canada)
AZ (Azerbaijan)       QA (Qatar)

RECOMMENDED WORKFLOW
--------------------
1. Run weather update 2-3 days before race weekend
2. System automatically updates circuit_config.json
3. Run simulation with updated rain probability
4. Optionally repeat closer to race for better accuracy

PARAMETER EXTRACTION
================================================================================

The --extract command analyzes 3 years of historical data (2022-2024) to
determine circuit-specific parameters:

EXTRACTED DATA
--------------
1. Position Penalties
   - Traffic/dirty air effects for each grid position
   - Based on actual finishing position changes

2. Tire Performance
   - Base lap times for each compound (SOFT, MEDIUM, HARD)
   - Degradation rates in seconds per lap
   - Relative compound offsets

3. Driver Error Rates
   - Dry and wet condition error probabilities
   - Incident likelihood per lap

4. DRS Effectiveness
   - Time advantage in seconds
   - Usage probability based on track characteristics

WHEN TO USE
-----------
✓ Before important race weekends (more accurate predictions)
✓ After major regulation changes
✓ For detailed analysis requiring historical accuracy

When NOT needed:
- Quick strategy overview (fallback parameters work fine)
- Circuit with limited historical data
- Time-sensitive analysis

OUTPUT FILE
-----------
Creates: {circuit_name}_params.py

Example: monaco_params.py, united_states_gp_params.py

This file is automatically loaded by --simulate if it exists.

TIRE MODELING SYSTEM
================================================================================

PRACTICE-BASED BAYESIAN MODELS
-------------------------------
During simulation, the system automatically:

1. Loads FP1, FP2 and/or Sprint session data (if available)
2. Builds Bayesian tire degradation models using MCMC (NumPyro)
3. Uses actual practice tire performance in race simulation
4. Falls back to extracted/default parameters if practice data unavailable

MODEL QUALITY INDICATORS
-------------------------
HIGH: 500+ clean laps, narrow uncertainty intervals
GOOD: 200-500 clean laps, moderate uncertainty
MODERATE: 50-200 clean laps, wider uncertainty
LIMITED: <50 clean laps, fallback parameters used

COMPOUNDS MODELED
-----------------
• SOFT: Fastest, highest degradation
• MEDIUM: Balanced performance
• HARD: Slowest, lowest degradation

Note: WET and INTERMEDIATE compounds not modeled as usage is dictated by
track conditions rather than degradation

DEGRADATION MODELING
--------------------
The Bayesian model fits: LapTime = α + β × StintLap

Where:
- α (alpha) = Base lap time for compound
- β (beta) = Degradation rate per lap
- Both parameters have uncertainty distributions

SIMULATION MECHANICS
================================================================================

MONTE CARLO APPROACH
--------------------
- 1,500 iterations per strategy per grid position
- Stochastic elements: SC/VSC timing, rain onset, driver errors, DRS usage
- Parallel simulation of all strategies for fair comparison

KEY POSITIONS ANALYZED
----------------------
Grid P1, P3, P5, P8, P10, P15

(Represents podium contenders, midfield, and back markers)

SIMULATED STRATEGIES
--------------------

1-Stop Strategies:
  1. 1-stop (M-H): Medium → Hard
  2. 1-stop (H-M): Hard → Medium
  3. 1-stop (H-S): Hard → Soft
  4. 1-stop (S-H): Soft → Hard
  5. 1-stop (S-M): Soft → Medium
  6. 1-stop (M-S): Medium → Soft

2-Stop Strategies:
  7. 2-stop (M-H-S): Medium → Hard → Soft
  8. 2-stop (H-M-H): Hard → Medium → Hard
  9. 2-stop (S-H-S): Soft → Hard → Soft
  10. 2-stop (S-H-M): Soft → Hard → Medium
  11. 2-stop (S-M-S): Soft → Medium → Soft

RACE SIMULATION FACTORS
------------------------
✓ Tire degradation (practice-based or extracted models)
✓ Fuel load effect (decreasing weight over race)
✓ Track evolution (track gets faster throughout race)
✓ Position penalties (traffic, dirty air)
✓ DRS effectiveness (position-dependent probability)
✓ Safety car periods (random timing, compound effects)
✓ Virtual safety car (separate probability)
✓ Driver errors (lap-dependent probability)
✓ Weather changes (rain probability)
✓ Pit stop execution variability
✓ Undercut effects (strategic timing bonus)
✓ Car performance (grid position proxy)

OUTPUT METRICS
--------------
For each strategy:
- Average race time
- Average final position
- Average points scored
- Points finish percentage
- Top 5 finish percentage
- Podium percentage (for relevant grid positions)
- Win percentage (for front-runners)

VALIDATION SYSTEM
================================================================================

POST-RACE VALIDATION (--validate)
----------------------------------
After a race, validate model predictions against actual results:

python validation.py 2024 "Monaco"
python f1_strategy_analyzer.py --validate 2025 "Monaco"

VALIDATION METRICS
------------------
1. Position Prediction Accuracy
   - Mean Absolute Error (MAE) in finishing positions
   - Predicted vs Actual scatter plots
   - Strategy-specific accuracy

2. Race Conditions
   - Rain occurrence prediction
   - Safety car occurrence prediction
   - Parameter quality assessment

3. Strategy Effectiveness
   - Actual strategy distribution
   - Strategy-to-results correlation
   - Position gains/losses analysis

INTERPRETATION
--------------
MAE ≤ 2.0 positions: Excellent accuracy
MAE 2.0-3.0: Good accuracy
MAE 3.0-4.0: Moderate accuracy
MAE > 4.0: Poor accuracy (parameter refinement needed)

OUTPUT FILES
================================================================================

GENERATED FILES
---------------
{circuit}_params.py
    Extracted circuit-specific parameters (if --extract used)

circuit_config.json
    Updated with weather data (if --weather used)

VISUALIZATION PLOTS
-------------------
1. Race Time Distributions
   - Histograms for each strategy
   - Separate plots for different grid positions

2. Tire Degradation Models
   - Practice-based tire performance curves
   - Model quality indicators
   - Compound comparison

3. Validation Plots (if --validate used)
   - Predicted vs Actual positions
   - Prediction accuracy by grid position
   - Strategy effectiveness
   - Position changes analysis

TYPICAL WORKFLOWS
================================================================================

RACE WEEKEND PREPARATION
------------------------
# Monday-Tuesday: Initial analysis with historical data
python f1_strategy_analyzer.py --extract 2025 "Monaco"
python pit_sim.py 2025 "Monaco"

# Wednesday-Thursday: Update with weather forecast
python f1_strategy_analyzer.py --weather "Monaco" --api-key YOUR_KEY

# Friday: Re-run simulation with updated weather
python pit_sim.py 2025 "Monaco"

# Post-FP1/FP2: Simulation now uses practice tire data automatically
python pit_sim.py 2025 "Monaco"

# Post-FP2: Calculate laptime targets for strategy decisions
python pit_sim.py 2025 "Monaco" --targets

# Sunday: Post-race validation
python validation.py 2025 "Monaco"

QUICK ANALYSIS (NO PREP)
-------------------------
# Just run simulation with fallback parameters
python pit_sim.py 2025 "Monaco"

LAPTIME TARGET ANALYSIS
------------------------
# Full analysis for all compounds
python laptime_targets.py 2025 "Monaco"

# Specific compound analysis from P3
python laptime_targets.py 2025 "Monaco" --compound SOFT --position 3

# Via pit_sim.py
python pit_sim.py 2025 "Monaco" --targets MEDIUM --position 5

HISTORICAL RESEARCH
-------------------
# Look at past strategies for a circuit
python f1_strategy_analyzer.py --historical "Monaco"

# Extract actual strategies from most recent race
python f1_strategy_analyzer.py --strategies 2025 "Monaco"

PROGRAMMATIC USAGE
================================================================================

BASIC SIMULATION
----------------
import json
from tire_model import build_models
from pit_sim import load_params, get_strats, run_mc

with open('circuit_config.json') as f:
    circuits = json.load(f)

circuit = circuits['Monaco']
params, _ = load_params(circuit['gp_name'])
models, model_info = build_models(2025, 'Monaco', circuit['base_pace'])

strategies = get_strats(circuit)
results = run_mc(strategies, models, circuit, 
                 grid_pos=[1, 5, 10], params=params, n_sims=1000)

for pos, data in results.items():
    print(f"Grid P{pos}:")
    for strat, res in data.items():
        avg_pts = sum(res['pts']) / len(res['pts'])
        print(f"  {strat}: {avg_pts:.1f} avg points")

LAPTIME TARGET CALCULATOR
--------------------------
from laptime_targets import LaptimeTargetCalculator

calculator = LaptimeTargetCalculator(circuit, models, params)

# Get current compound pace
soft_pace = calculator.get_current_compound_pace('SOFT', stint_lap=10)

# Calculate all crossover thresholds
thresholds = calculator.calculate_all_thresholds(grid_pos=1, n_sims=100)

# Generate pace sensitivity table
sensitivity = calculator.generate_pace_sensitivity_table(
    compound='SOFT', 
    grid_pos=1, 
    pace_range=(-1.5, 1.5), 
    steps=7
)
print(sensitivity)

# Find best strategy for specific compound paces
best_strat, all_results = calculator.find_best_strategy_for_pace(
    compound_paces={'SOFT': 74.0, 'MEDIUM': 74.5, 'HARD': 75.2},
    grid_pos=1,
    n_sims=100
)
print(f"Best strategy: {best_strat}")

TECHNICAL REQUIREMENTS
================================================================================

DEPENDENCIES
------------
fastf1              # F1 data access
pandas              # Data manipulation
numpy               # Numerical computing
jax                 # Accelerated computing
numpyro             # Bayesian modeling
matplotlib          # Plotting
seaborn             # Statistical visualization
scipy               # Scientific computing
requests            # HTTP requests (for weather API)
tqdm                # Progress bars

Install all:
pip install fastf1 pandas numpy jax jaxlib numpyro matplotlib seaborn scipy requests tqdm

PYTHON VERSION
--------------
Python 3.8 or higher required
Python 3.9+ recommended

DISK SPACE
----------
~500 MB for FastF1 cache
(automatically downloaded on first use)

INTERNET CONNECTION
-------------------
Required for:
- FastF1 data downloads
- Weather API requests
- Initial setup

Can run offline after data is cached.

TROUBLESHOOTING
================================================================================

ISSUE: "No data available for X Grand Prix"
SOLUTION: Race hasn't occurred yet or FastF1 data not released.
          Wait 24-48 hours after race for data availability.

ISSUE: "Could not load practice data"
SOLUTION: Practice sessions may not have occurred yet.
          System will use fallback tire modeling automatically.

ISSUE: "Parameter extraction failed"
SOLUTION: Insufficient historical data (new circuit, regulation changes).
          Use fallback parameters or try fewer years.

ISSUE: Weather API not working
SOLUTION: 
    - Check API key is valid
    - Verify internet connection
    - Ensure free tier limit (1000 calls/day) not exceeded
    - Check city/country code spelling

ISSUE: Import errors (numpyro, jax)
SOLUTION: 
    pip install --upgrade jax jaxlib numpyro
    
    For Mac M1/M2:
    pip install --upgrade jax jaxlib numpyro

ISSUE: Simulation very slow
SOLUTION: 
    - Reduce n_sims parameter (default 1500)
    - Check CPU usage (should use multiple cores)
    - Close other applications

ISSUE: "Circuit not found in config"
SOLUTION: 
    - Check exact spelling of Grand Prix name
    - Use quotes: "Monaco" not Monaco
    - Check circuit_config.json for exact name format

ISSUE: Laptime targets taking too long
SOLUTION:
    - Reduce --sims parameter (default 100)
    - Analyze single compound with --compound option
    - Use fewer grid positions

LIMITATIONS
================================================================================

DATA CONSTRAINTS
----------------
- Public timing data does not include fuel loads, tire temperatures, or telemetry
- Practice stints are short, limiting degradation curve accuracy
- Unknown driver run plans make lap filtering imprecise

MODEL ASSUMPTIONS
-----------------
- Linear tire degradation (does not capture tire cliff)
- Position changes at pit stops are probabilistic estimates
- Safety car timing is randomized based on historical probability
- Fuel correction uses fixed rate per circuit

LAPTIME TARGETS
---------------
- Thresholds assume other compounds remain at modeled pace
- Crossover points have uncertainty from Monte Carlo variance
- Analysis at boundary values (>5s adjustment) less reliable

CUSTOMIZATION
================================================================================

MODIFYING CIRCUIT PARAMETERS
-----------------------------
Edit circuit_config.json manually:

{
  "Monaco": {
    "laps": 78,
    "base_pace": 73,
    "pit_loss": 18,
    "rain_prob": 0.10,
    "sc_prob": 0.40,
    "vsc_prob": 0.30,
    "fuel_rate": 1.4,
    "fuel_effect": 0.035,
    "sprint": false
  }
}

ADJUSTING SIMULATION PARAMETERS
--------------------------------
In pit_sim.py, modify:

- n_sims in run_mc(): Number of Monte Carlo iterations (default 1500)
- key_pos: Grid positions to analyze
- get_strats(): Add/remove strategies

ADDING CUSTOM STRATEGIES
-------------------------
Modify get_strats() function in pit_sim.py:

def get_strats(circuit):
    total_laps = circuit['laps']
    
    dry = {
        "3-stop (S-S-M-H)": [
            {"comp": "SOFT", "laps": int(total_laps * 0.20)},
            {"comp": "SOFT", "laps": int(total_laps * 0.20)},
            {"comp": "MEDIUM", "laps": int(total_laps * 0.25)},
            {"comp": "HARD", "laps": int(total_laps * 0.35)}
        ],
        # ... existing strategies
    }
    return dry

CREDITS & REFERENCES
================================================================================

FASTF1 PROJECT
--------------
FastF1 provides all F1 data: https://github.com/theOehrly/Fast-F1
Formula 1 timing data: © Formula 1

OPENWEATHERMAP
--------------
Weather forecasts: https://openweathermap.org/

MODELING APPROACH
-----------------
- Monte Carlo simulation
- Bayesian tire modeling (MCMC with NumPyro)
- Stochastic race event generation
- Historical parameter extraction
- Binary search crossover detection

================================================================================
