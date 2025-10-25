
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
✓ Comprehensive strategy analysis and visualization

================================================================================
QUICK START
================================================================================

BASIC USAGE (No Weather, Fallback Parameters)
----------------------------------------------
python f1_strategy_analyzer.py --simulate 2025 "Monaco"
python f1_strategy_analyzer.py --simulate 2025 "United States"
python f1_strategy_analyzer.py --simulate 2025 "Italian Grand Prix"

WITH EXTRACTED PARAMETERS (More Accurate)
------------------------------------------
# Step 1: Extract circuit-specific parameters from historical data
python f1_strategy_analyzer.py --extract 2025 "Monaco"

# Step 2: Run simulation with extracted parameters
python f1_strategy_analyzer.py --simulate 2025 "Monaco"

WITH WEATHER UPDATES
---------------------
python f1_strategy_analyzer.py --weather "Monaco" --api-key YOUR_API_KEY

================================================================================
COMMAND REFERENCE
================================================================================

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

================================================================================
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

================================================================================
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

================================================================================
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

================================================================================
TIRE MODELING SYSTEM
================================================================================

PRACTICE-BASED BAYESIAN MODELS
-------------------------------
During --simulate, the system automatically:

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

Note: WET and INTERMEDIATE compounds not modeled as usage is dictated by track conditions rather than degradation

DEGRADATION MODELING
--------------------
The Bayesian model fits: LapTime = α + β × StintLap

Where:
- α (alpha) = Base lap time for compound
- β (beta) = Degradation rate per lap
- Both parameters have uncertainty distributions

================================================================================
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

1. 1-stop (M-H): Medium → Hard
2. 1-stop (H-M): Hard → Medium
3. 1-stop (H-S): Hard → Soft
4. 1-stop (S-H): Soft → Hard
5. 1-stop (S-M): Soft → Medium
6. 1-stop (M-S): Medium → Soft
7. 2-stop (M-H-S): Medium → Hard → Soft
8. 2-stop (H-M-H): Hard → Medium → Hard
9. 2-stop (H-S-H): Hard → Soft → Hard
10. 2-stop (M-H-M): Medium → Hard → Medium
11. 2-stop (S-H-M): Soft → Hard → Medium

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

================================================================================
VALIDATION SYSTEM
================================================================================

POST-RACE VALIDATION (--validate)
----------------------------------
After a race, validate model predictions against actual results:

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

================================================================================
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

================================================================================
TYPICAL WORKFLOWS
================================================================================

RACE WEEKEND PREPARATION
------------------------
# Monday-Tuesday: Initial analysis with historical data
python f1_strategy_analyzer.py --extract 2025 "Monaco"
python f1_strategy_analyzer.py --simulate 2025 "Monaco"

# Wednesday-Thursday: Update with weather forecast
python f1_strategy_analyzer.py --weather "Monaco" --api-key YOUR_KEY

# Friday: Re-run simulation with updated weather
python f1_strategy_analyzer.py --simulate 2025 "Monaco"

# Post-FP1/Sprint: Simulation now uses practice tire data automatically
python f1_strategy_analyzer.py --simulate 2025 "Monaco"

# Sunday: Post-race validation
python f1_strategy_analyzer.py --validate 2025 "Monaco"

QUICK ANALYSIS (NO PREP)
-------------------------
# Just run simulation with fallback parameters
python f1_strategy_analyzer.py --simulate 2025 "Monaco"

HISTORICAL RESEARCH
-------------------
# Look at past strategies for a circuit
python f1_strategy_analyzer.py --historical "Monaco"

# Extract actual strategies from most recent race
python f1_strategy_analyzer.py --strategies 2025 "Monaco"

================================================================================
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

================================================================================
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
    - Reduce num_sims in code (default 1500)
    - Check CPU usage (should use multiple cores)
    - Close other applications

ISSUE: "Circuit not found in config"
SOLUTION: 
    - Check exact spelling of Grand Prix name
    - Use quotes: "Monaco" not Monaco
    - Check circuit_config.json for exact name format

================================================================================
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
    "sprint": false
  }
}

ADJUSTING SIMULATION PARAMETERS
--------------------------------
In f1_strategy_analyzer.py, modify:

- num_sims: Number of Monte Carlo iterations (default 1500)
- key_positions: Grid positions to analyze
- strategies: Add/remove strategies in generate_strategies()

WEATHER API ALTERNATIVES
-------------------------
Code uses OpenWeatherMap but can be adapted for:
- WeatherAPI.com
- Tomorrow.io
- Visual Crossing
- Met Office

================================================================================
ADVANCED FEATURES
================================================================================

CUSTOM STRATEGY TESTING
------------------------
Modify generate_strategies() function to test custom strategies:

def generate_strategies(rain_prob, sprint):
    strategies = {
        "3-stop (S-S-M-H)": [
            {"compound": "SOFT", "laps": 12},
            {"compound": "SOFT", "laps": 12},
            {"compound": "MEDIUM", "laps": 15},
            {"compound": "HARD", "laps": 17}
        ]
    }
    return strategies

SENSITIVITY ANALYSIS
--------------------
Test how different parameters affect outcomes:

for rain_prob in [0.0, 0.1, 0.2, 0.3]:
    # Update circuit config
    # Run simulation
    # Compare results

TEAM-SPECIFIC MODELING
----------------------
Add car_performance_factor adjustments for different teams:

car_performance_map = {
    1: 0.98,   # Red Bull
    3: 0.99,   # Ferrari
    5: 1.00,   # McLaren
    8: 1.02,   # Mercedes
    ...
}

================================================================================
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

================================================================================

Created by: Jessica Steele
Last Updated: October 2025

For questions, issues, or improvements, please document your findings.

================================================================================
