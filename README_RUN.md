# LEGOLAS ZnSSe - Execution Flow Guide

## Overview

This document traces the **exact sequence of function calls** when running `python3 demo.py`. Use this to understand control flow and data flow without reverse engineering the code.

**Purpose**: Educational guide to LEGOLAS system architecture and execution

**Last Updated**: November 2025

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Entry Point](#entry-point)
3. [Execution Flow - Complete Sequence](#execution-flow---complete-sequence)
4. [Part 1: Material Properties Demo](#part-1-material-properties-demo)
5. [Part 2: Bayesian Optimization](#part-2-bayesian-optimization)
6. [Part 3: Results Analysis](#part-3-results-analysis)
7. [Part 4: Full Composition Exploration](#part-4-full-composition-exploration)
8. [Function Reference](#function-reference)
9. [Data Flow Diagram](#data-flow-diagram)

---

## Quick Reference

**Command**: `python3 demo.py`

**Total Runtime**: ~15-30 seconds (simulated mode)

**Function Calls**: ~150-200 function calls (depending on optimization iterations)

**Output Files**:
- `results/optimization_results.csv` (14 rows × 7 columns)
- `results/optimization_plot.png` (2-panel figure)
- `results/composition_space_map.png` (2-panel figure)

---

## Entry Point

### File: `demo.py`

```python
if __name__ == "__main__":
    main()
```

**Execution starts here** when you run `python3 demo.py`

---

## Execution Flow - Complete Sequence

### Level 0: Script Execution

```
python3 demo.py
  ↓
if __name__ == "__main__":
  ↓
main()
```

### Level 1: Main Function (`demo.py:236-294`)

```python
def main():
    """Run complete LEGOLAS ZnSSe demonstration."""

    # Call sequence:
    demo_interface()                    # Part 1
    ↓
    optimizer, results_df = demo_optimization()  # Part 2
    ↓
    analyze_results(optimizer, results_df)       # Part 3
    ↓
    demo_full_exploration()             # Part 4
```

**Control Flow**: Sequential execution (no parallelization)

**Error Handling**: Try/except wrapper catches KeyboardInterrupt and general exceptions

---

## Part 1: Material Properties Demo

### Function: `demo_interface()` (`demo.py:42-67`)

**Purpose**: Demonstrate ZnSSe material interface capabilities

**Duration**: ~0.5 seconds

#### Call Sequence:

```
1. demo_interface()
   ├─ print_header("PART 1: ZnS(1-x)Se(x) Material Properties")
   │
   ├─ ZnSSeInterface.__init__(mode='hardware_with_fallback', egap_method='aflow_with_fallback')
   │  ├─ AFLOWInterface.__init__()  [if AFLOW_AVAILABLE]
   │  │  └─ Initializes: base_url, timeout=30, cache={}
   │  │
   │  ├─ _init_hardware()  [attempts hardware initialization]
   │  │  ├─ import spidev, RPi.GPIO  [ImportError expected on non-RPi]
   │  │  └─ Returns: False (hardware unavailable)
   │  │
   │  └─ Fallback: self.mode = 'simulated'
   │
   └─ FOR x_Se in [0.0, 0.25, 0.50, 0.75, 1.0]:
      ├─ interface.compute_bandgap(x_Se)
      │  ├─ Validates: 0.0 <= x_Se <= 1.0
      │  ├─ Checks method: 'aflow_with_fallback'
      │  ├─ AFLOWInterface.get_bandgap(x_Se, method='aflow_with_fallback')  [if available]
      │  │  ├─ query_znsse_compounds(x_Se_target=x_Se)
      │  │  │  ├─ requests.get(query_url, timeout=30)
      │  │  │  └─ Returns: [] (no AFLOW data for ZnSSe)
      │  │  │
      │  │  └─ Fallback: _vegard_law(x_Se)
      │  │     └─ Eg = (1-x)*3.68 + x*2.70 - 0.50*x*(1-x)
      │  │
      │  └─ Returns: (Eg, 'vegard')
      │
      ├─ interface.composition_string(x_Se)
      │  └─ Returns: f"ZnS{1-x_Se:.2f}Se{x_Se:.2f}"
      │
      └─ Print: x_Se, formula, Eg, delta_vs_TiO2
```

#### Data Flow:

**Input**: x_Se values [0.0, 0.25, 0.50, 0.75, 1.0]

**Processing**:
- x_Se=0.0  → Eg=3.68 eV (pure ZnS)
- x_Se=0.25 → Eg=3.27 eV (ZnS₀.₇₅Se₀.₂₅)
- x_Se=0.50 → Eg=3.07 eV (ZnS₀.₅₀Se₀.₅₀)
- x_Se=0.75 → Eg=2.84 eV (ZnS₀.₂₅Se₀.₇₅)
- x_Se=1.0  → Eg=2.70 eV (pure ZnSe)

**Output**: Printed table comparing bandgaps to TiO₂ (3.2 eV)

**State Changes**:
- `interface` object created and initialized
- AFLOW interface initialized (attempts connection)
- Hardware fallback to simulated mode

---

## Part 2: Bayesian Optimization

### Function: `demo_optimization()` (`demo.py:69-93`)

**Purpose**: Run 10-iteration GP-guided optimization

**Duration**: ~5-10 seconds

#### Call Sequence:

```
1. demo_optimization()
   ├─ print_header("PART 2: Bayesian Optimization with Gaussian Process")
   │
   ├─ ZnSSeInterface.__init__(mode='hardware_with_fallback', egap_method='aflow_with_fallback')
   │  └─ [Same initialization as Part 1]
   │
   ├─ GPOptimizer.__init__(interface, xi=0.01, random_state=42)
   │  ├─ self.interface = interface
   │  ├─ self.xi = 0.01  [Expected Improvement exploration parameter]
   │  ├─ self.random_state = 42
   │  ├─ np.random.seed(42)
   │  ├─ self.X_bandgaps = []
   │  ├─ self.y_voltages = []
   │  ├─ self.x_compositions = []
   │  ├─ self.results = []
   │  └─ self.gp_model = None
   │
   └─ optimizer.optimize(n_iterations=10, n_initial=4)
      │
      ├─ FOR i in range(10):  [Iterations 0-9]
      │  │
      │  ├─ IF i < 4:  [Initial random exploration]
      │  │  ├─ x_Se = np.random.uniform(0.0, 1.0)
      │  │  └─ Print: "Random sampling: x_Se = ..."
      │  │
      │  ├─ ELSE:  [GP-guided optimization]
      │  │  ├─ IF i == 4:  [Build initial GP model after 4 samples]
      │  │  │  └─ _build_gp_model()
      │  │  │     ├─ Define kernel: ConstantKernel * RBF + WhiteKernel
      │  │  │     │  ├─ ConstantKernel(1.0, bounds=(1e-3, 1e3))
      │  │  │     │  ├─ RBF(length_scale=0.5, bounds=(1e-2, 1e1))
      │  │  │     │  └─ WhiteKernel(noise_level=0.01, bounds=(1e-5, 1e0))
      │  │  │     │
      │  │  │     ├─ GaussianProcessRegressor.__init__(kernel, n_restarts=10, normalize_y=True)
      │  │  │     │
      │  │  │     ├─ X = np.array(self.X_bandgaps).reshape(-1, 1)  [4×1 array]
      │  │  │     ├─ y = np.array(self.y_voltages)  [4 values]
      │  │  │     │
      │  │  │     └─ self.gp_model.fit(X, y)
      │  │  │        ├─ Optimize hyperparameters (length_scale, noise_level)
      │  │  │        └─ Returns: trained GP model
      │  │  │
      │  │  └─ _propose_next_composition(n_candidates=100)
      │  │     ├─ x_Se_candidates = np.linspace(0.0, 1.0, 100)
      │  │     │
      │  │     ├─ Eg_candidates = [interface.compute_bandgap(x)[0] for x in x_Se_candidates]
      │  │     │  └─ Calls: compute_bandgap() 100 times
      │  │     │
      │  │     ├─ _expected_improvement(Eg_candidates)
      │  │     │  ├─ mu, sigma = self.gp_model.predict(Eg_candidates, return_std=True)
      │  │     │  ├─ f_best = np.max(self.y_voltages)
      │  │     │  ├─ Z = (mu - f_best - xi) / sigma
      │  │     │  └─ ei = (mu - f_best - xi)*norm.cdf(Z) + sigma*norm.pdf(Z)
      │  │     │
      │  │     ├─ best_idx = np.argmax(ei_values)
      │  │     └─ Returns: x_Se_candidates[best_idx]
      │  │
      │  ├─ formula = interface.composition_string(x_Se)
      │  │
      │  ├─ _measure_composition(x_Se, iteration=i)
      │  │  ├─ interface.measure_voltage(x_Se)
      │  │  │  ├─ compute_bandgap(x_Se)  → Returns: (Eg, 'vegard')
      │  │  │  │
      │  │  │  ├─ IF mode == 'hardware':
      │  │  │  │  └─ _read_adc_voltage(channel=1)
      │  │  │  │     ├─ spi.xfer2([1, (8+channel)<<4, 0])
      │  │  │  │     └─ voltage = (data/1023.0) * 3.3
      │  │  │  │
      │  │  │  └─ ELSE (simulated):
      │  │  │     ├─ Voc_base = 0.3 + 0.15 * Eg
      │  │  │     ├─ Voc += 0.025 * np.log(light_intensity + 0.01)
      │  │  │     ├─ Voc += np.random.normal(0, 0.010)  [±10 mV noise]
      │  │  │     └─ Voc = np.clip(Voc, 0.1, 1.0)
      │  │  │
      │  │  ├─ IF gp_model is not None:
      │  │  │  ├─ Eg_test = [[measurement['Eg_eV']]]
      │  │  │  └─ predicted_Voc, uncertainty = gp_model.predict(Eg_test, return_std=True)
      │  │  │
      │  │  ├─ result = OptimizationResult(iteration, x_Se, Eg, Voc, predicted_Voc, uncertainty)
      │  │  │
      │  │  ├─ self.X_bandgaps.append(Eg)
      │  │  ├─ self.y_voltages.append(Voc)
      │  │  ├─ self.x_compositions.append(x_Se)
      │  │  ├─ self.results.append(result)
      │  │  │
      │  │  └─ Returns: result
      │  │
      │  ├─ Print: Bandgap, Voltage, GP prediction (if available)
      │  │
      │  └─ IF i >= 4:  [Re-train GP model after each new measurement]
      │     └─ _build_gp_model()
      │        └─ [Updates GP with all data points]
      │
      ├─ best_idx = np.argmax(self.y_voltages)
      ├─ best_result = self.results[best_idx]
      │
      ├─ results_df = pd.DataFrame([...])
      │  └─ Columns: iteration, x_Se, formula, Eg_eV, Voc_V, predicted_Voc_V, uncertainty_V
      │
      └─ Returns: results_df
```

#### Data Flow:

**Iteration-by-Iteration Execution**:

| Iter | Phase | x_Se | Eg (eV) | Voc (V) | Method |
|------|-------|------|---------|---------|--------|
| 0 | Random | 0.637 | 2.90 | 0.735 | Random sampling |
| 1 | Random | 0.269 | 3.21 | 0.782 | Random sampling |
| 2 | Random | 0.447 | 3.06 | 0.759 | Random sampling |
| 3 | Random | 0.715 | 2.83 | 0.725 | Random sampling |
| 4 | GP-Build | - | - | - | Build initial GP model (4 points) |
| 4 | GP-Guided | 0.123 | 3.42 | 0.813 | Expected Improvement |
| 5 | GP-Guided | 0.089 | 3.49 | 0.824 | Expected Improvement |
| 6 | GP-Guided | 0.034 | 3.59 | 0.838 | Expected Improvement |
| 7 | GP-Guided | 0.012 | 3.64 | 0.847 | Expected Improvement |
| 8 | GP-Guided | 0.005 | 3.66 | 0.849 | Expected Improvement |
| 9 | GP-Guided | 0.000 | 3.68 | 0.852 | Expected Improvement |

**Note**: Actual values vary due to random seed, but trend is consistent

**State Changes**:
- `optimizer.X_bandgaps`: [] → [10 bandgap values]
- `optimizer.y_voltages`: [] → [10 voltage measurements]
- `optimizer.gp_model`: None → Trained GPR model (after iteration 4)
- `optimizer.results`: [] → [10 OptimizationResult objects]

**Output**:
- `results_df` (DataFrame with 10 rows)
- Saved to: `results/optimization_results.csv`
- Plot saved to: `results/optimization_plot.png`

---

## Part 3: Results Analysis

### Function: `analyze_results(optimizer, results_df)` (`demo.py:95-158`)

**Purpose**: Analyze optimization performance and GP model accuracy

**Duration**: ~0.1 seconds

#### Call Sequence:

```
1. analyze_results(optimizer, results_df)
   ├─ print_header("PART 3: Results Analysis")
   │
   ├─ best_idx = results_df['Voc_V'].idxmax()
   ├─ best = results_df.loc[best_idx]
   │
   ├─ Print best composition:
   │  ├─ Iteration
   │  ├─ Formula (composition_string was called earlier)
   │  ├─ x_Se
   │  ├─ Bandgap
   │  └─ Voltage
   │
   ├─ Calculate improvement:
   │  ├─ initial_Voc = results_df.loc[0, 'Voc_V']
   │  ├─ final_Voc = best['Voc_V']
   │  └─ improvement = ((final_Voc - initial_Voc) / initial_Voc) * 100
   │
   ├─ optimizer.interface.compare_to_tio2(best['x_Se'])
   │  ├─ compute_bandgap(x_Se)  → Returns: (Eg_znsse, 'vegard')
   │  ├─ delta_Eg = Eg_znsse - Eg_TiO2
   │  └─ Returns: {composition, Eg_ZnSSe, Eg_source, Eg_TiO2, delta_Eg, recommendation}
   │
   ├─ GP model performance metrics:
   │  ├─ predictions = results_df['predicted_Voc_V'].dropna()
   │  ├─ actuals = results_df.loc[predictions.index, 'Voc_V']
   │  ├─ mae = np.mean(np.abs(predictions - actuals))
   │  └─ rmse = np.sqrt(np.mean((predictions - actuals)**2))
   │
   └─ Print summary statistics
```

#### Data Flow:

**Input**:
- `optimizer` object with trained GP model
- `results_df` DataFrame (10 rows × 7 columns)

**Processing**:
- Finds best result: highest Voc_V
- Calculates improvement: (best_Voc - initial_Voc) / initial_Voc × 100%
- Compares to TiO₂ benchmark
- Evaluates GP model accuracy (MAE, RMSE)

**Output**: Printed analysis report

**Typical Results**:
- Best Voc: ~0.850 V (ZnS-rich composition)
- Improvement: +10-15% over initial random sample
- GP MAE: <0.01 V
- GP RMSE: <0.02 V

---

## Part 4: Full Composition Exploration

### Function: `demo_full_exploration()` (`demo.py:160-234`)

**Purpose**: Map entire composition space (0.0 to 1.0)

**Duration**: ~2-5 seconds

#### Call Sequence:

```
1. demo_full_exploration()
   ├─ print_header("PART 4: Full Composition Space Exploration")
   │
   ├─ ZnSSeInterface.__init__(mode='hardware_with_fallback', egap_method='aflow_with_fallback')
   │  └─ [Same initialization as before]
   │
   ├─ x_Se_range = np.linspace(0.0, 1.0, 50)  [50 compositions]
   │
   ├─ FOR x_Se in x_Se_range:  [Loop executes 50 times]
   │  ├─ interface.measure_voltage(x_Se)
   │  │  ├─ compute_bandgap(x_Se)  → (Eg, 'vegard')
   │  │  ├─ Voc_base = 0.3 + 0.15 * Eg
   │  │  ├─ Voc += noise
   │  │  └─ Returns: {x_Se, formula, Eg_eV, Eg_source, Voc_V, light_intensity, measurement_mode, egap_method}
   │  │
   │  └─ data.append({x_Se, Eg_eV, Voc_V})
   │
   ├─ df = pd.DataFrame(data)  [50 rows × 3 columns]
   │
   ├─ Create visualization:
   │  ├─ fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
   │  │
   │  ├─ Plot 1: Bandgap vs Composition
   │  │  ├─ ax1.plot(df['x_Se'], df['Eg_eV'])
   │  │  ├─ ax1.axhline(interface.Eg_TiO2, color='green')
   │  │  └─ Add ZnS/ZnSe endpoint labels
   │  │
   │  └─ Plot 2: Voltage vs Composition
   │     ├─ ax2.plot(df['x_Se'], df['Voc_V'])
   │     ├─ ax2.fill_between(±0.01 V noise band)
   │     ├─ optimal_idx = df['Voc_V'].idxmax()
   │     └─ ax2.plot(optimal_x, optimal_V, 'g*')  [Mark optimum]
   │
   ├─ plt.savefig('results/composition_space_map.png', dpi=150)
   │
   └─ Print optimal composition from full scan
```

#### Data Flow:

**Input**: 50 uniformly spaced compositions (x_Se = 0.00, 0.02, 0.04, ..., 1.00)

**Processing**:
- For each composition:
  - Calculate bandgap via Vegard's law
  - Simulate voltage measurement
  - Store (x_Se, Eg, Voc) tuple

**Output**:
- DataFrame (50 rows × 3 columns)
- 2-panel figure saved to `results/composition_space_map.png`
- Optimal composition identified

**Typical Optimal Composition**: x_Se ≈ 0.00-0.10 (ZnS-rich)

---

## Function Reference

### Module: `znsse_interface.py`

#### Class: `ZnSSeInterface`

```python
__init__(mode='simulated', egap_method='vegard')
    │
    ├─ Parameters:
    │  ├─ mode: 'simulated' | 'hardware' | 'hardware_with_fallback'
    │  └─ egap_method: 'vegard' | 'aflow' | 'aflow_with_fallback'
    │
    ├─ Initializes:
    │  ├─ self.Eg_ZnS = 3.68 eV
    │  ├─ self.Eg_ZnSe = 2.70 eV
    │  ├─ self.bowing = 0.50 eV
    │  ├─ self.Eg_TiO2 = 3.2 eV
    │  ├─ self.aflow_interface = AFLOWInterface() [if AFLOW_AVAILABLE]
    │  └─ self.spi = spidev.SpiDev() [if hardware mode]
    │
    └─ Calls:
       ├─ AFLOWInterface.__init__() [if egap_method includes 'aflow']
       └─ _init_hardware() [if mode includes 'hardware']
```

```python
compute_bandgap(x_Se, method=None)
    │
    ├─ Input: x_Se (0.0 to 1.0)
    ├─ Validates: 0.0 <= x_Se <= 1.0
    │
    ├─ IF method in ['aflow', 'aflow_with_fallback']:
    │  └─ aflow_interface.get_bandgap(x_Se, method=method)
    │     ├─ query_znsse_compounds(x_Se_target=x_Se)
    │     └─ Fallback: _vegard_law(x_Se) if no data
    │
    ├─ ELSE (vegard):
    │  └─ Eg = (1-x)*Eg_ZnS + x*Eg_ZnSe - b*x*(1-x)
    │
    └─ Returns: (Eg, source)
       ├─ Eg: float (eV)
       └─ source: 'vegard' | 'aflow' | 'aflow_fallback'
```

```python
measure_voltage(x_Se, light_intensity=1.0)
    │
    ├─ Calls: compute_bandgap(x_Se)  → (Eg, source)
    │
    ├─ IF mode == 'hardware':
    │  └─ _read_adc_voltage(channel=1)
    │     └─ Returns: voltage from MCP3008 ADC
    │
    ├─ ELSE (simulated):
    │  ├─ Voc_base = 0.3 + 0.15 * Eg
    │  ├─ Voc += 0.025 * ln(light_intensity + 0.01)
    │  ├─ Voc += normal(0, 0.010)  [noise]
    │  └─ Voc = clip(Voc, 0.1, 1.0)
    │
    └─ Returns: dict {x_Se, formula, Eg_eV, Eg_source, Voc_V, light_intensity, measurement_mode, egap_method}
```

```python
composition_string(x_Se)
    │
    ├─ x_S = 1.0 - x_Se
    └─ Returns: f"ZnS{x_S:.2f}Se{x_Se:.2f}"
```

```python
compare_to_tio2(x_Se)
    │
    ├─ Calls: compute_bandgap(x_Se)  → (Eg_znsse, source)
    ├─ delta_Eg = Eg_znsse - Eg_TiO2
    ├─ Generates recommendation based on delta_Eg
    │
    └─ Returns: dict {composition, Eg_ZnSSe, Eg_source, Eg_TiO2, delta_Eg, recommendation}
```

---

### Module: `gp_optimizer.py`

#### Class: `GPOptimizer`

```python
__init__(interface, xi=0.01, random_state=42)
    │
    ├─ self.interface = interface
    ├─ self.xi = 0.01  [Expected Improvement exploration parameter]
    ├─ np.random.seed(random_state)
    │
    └─ Initializes empty lists:
       ├─ self.X_bandgaps = []
       ├─ self.y_voltages = []
       ├─ self.x_compositions = []
       └─ self.results = []
```

```python
optimize(n_iterations=10, n_initial=4)
    │
    ├─ FOR i in range(n_iterations):
    │  │
    │  ├─ IF i < n_initial:
    │  │  └─ x_Se = random.uniform(0, 1)  [Random exploration]
    │  │
    │  ├─ ELSE:
    │  │  ├─ IF i == n_initial:
    │  │  │  └─ _build_gp_model()  [First GP build]
    │  │  │
    │  │  └─ x_Se = _propose_next_composition()  [Expected Improvement]
    │  │
    │  ├─ result = _measure_composition(x_Se, i)
    │  │  ├─ measurement = interface.measure_voltage(x_Se)
    │  │  ├─ IF gp_model is not None:
    │  │  │  └─ predicted_Voc, uncertainty = gp_model.predict(Eg)
    │  │  └─ Appends to history lists
    │  │
    │  └─ IF i >= n_initial:
    │     └─ _build_gp_model()  [Re-train GP with new data]
    │
    ├─ best_idx = argmax(y_voltages)
    ├─ best_result = results[best_idx]
    │
    └─ Returns: pd.DataFrame(results)
```

```python
_build_gp_model()
    │
    ├─ kernel = ConstantKernel * RBF + WhiteKernel
    │  ├─ ConstantKernel(1.0, bounds=(1e-3, 1e3))
    │  ├─ RBF(length_scale=0.5, bounds=(1e-2, 1e1))
    │  └─ WhiteKernel(noise_level=0.01, bounds=(1e-5, 1e0))
    │
    ├─ gp_model = GaussianProcessRegressor(kernel, n_restarts=10, normalize_y=True)
    │
    ├─ X = np.array(X_bandgaps).reshape(-1, 1)
    ├─ y = np.array(y_voltages)
    │
    └─ gp_model.fit(X, y)
       └─ Optimizes hyperparameters (length_scale, noise_level)
```

```python
_propose_next_composition(n_candidates=100)
    │
    ├─ x_Se_candidates = linspace(0, 1, 100)
    │
    ├─ Eg_candidates = [interface.compute_bandgap(x)[0] for x in x_Se_candidates]
    │  └─ Calls compute_bandgap() 100 times
    │
    ├─ ei_values = _expected_improvement(Eg_candidates)
    │  ├─ mu, sigma = gp_model.predict(Eg_candidates, return_std=True)
    │  ├─ f_best = max(y_voltages)
    │  ├─ Z = (mu - f_best - xi) / sigma
    │  └─ ei = (mu - f_best - xi)*Φ(Z) + sigma*φ(Z)
    │     ├─ Φ(Z) = norm.cdf(Z)  [cumulative distribution]
    │     └─ φ(Z) = norm.pdf(Z)  [probability density]
    │
    ├─ best_idx = argmax(ei_values)
    │
    └─ Returns: x_Se_candidates[best_idx]
```

```python
_expected_improvement(X_candidates)
    │
    ├─ Input: X_candidates (n×1 array of bandgaps)
    │
    ├─ mu, sigma = gp_model.predict(X_candidates, return_std=True)
    ├─ f_best = max(y_voltages)  [Best observed voltage]
    ├─ Z = (mu - f_best - xi) / sigma
    │
    └─ EI = (mu - f_best - xi) * Φ(Z) + sigma * φ(Z)
       │
       └─ Returns: ei_values (n-length array)
```

```python
plot_results(save_path=None)
    │
    ├─ fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    │
    ├─ Plot 1: Voltage vs Bandgap
    │  ├─ X_plot = linspace(2.6, 3.8, 100)
    │  ├─ mu, sigma = gp_model.predict(X_plot, return_std=True)
    │  ├─ ax1.plot(X_plot, mu, label='GP mean')
    │  ├─ ax1.fill_between(X_plot, mu-1.96*sigma, mu+1.96*sigma, label='95% CI')
    │  └─ ax1.scatter(X_bandgaps, y_voltages, label='Measurements')
    │
    ├─ Plot 2: Voltage vs Composition
    │  ├─ ax2.scatter(x_compositions, y_voltages, colored by iteration)
    │  └─ ax2.annotate(iteration numbers)
    │
    └─ plt.savefig(save_path, dpi=150)
```

---

### Module: `aflow_api.py`

#### Class: `AFLOWInterface`

```python
__init__(timeout=30, cache=True)
    │
    ├─ self.base_url = "http://aflowlib.org/API/search/"
    ├─ self.timeout = 30
    ├─ self.cache = {}  [Composition → bandgap cache]
    │
    └─ Fallback parameters:
       ├─ self.Eg_ZnS = 3.68 eV
       ├─ self.Eg_ZnSe = 2.70 eV
       └─ self.bowing = 0.50 eV
```

```python
get_bandgap(x_Se, method='aflow_with_fallback')
    │
    ├─ IF x_Se in cache:
    │  └─ Returns: (cached_Eg, 'cached')
    │
    ├─ IF method == 'vegard':
    │  └─ Returns: (_vegard_law(x_Se), 'vegard')
    │
    ├─ IF method in ['aflow', 'aflow_with_fallback']:
    │  ├─ entries = query_znsse_compounds(x_Se_target=x_Se)
    │  │
    │  ├─ IF entries found:
    │  │  ├─ Eg = mean([e.Egap for e in entries])
    │  │  └─ Returns: (Eg, 'aflow')
    │  │
    │  └─ ELSE (no AFLOW data):
    │     ├─ IF method == 'aflow':
    │     │  └─ Raise ValueError
    │     │
    │     └─ ELSE (fallback):
    │        ├─ Eg = _vegard_law(x_Se)
    │        └─ Returns: (Eg, 'vegard')
    │
    └─ cache[x_Se] = Eg  [Cache result]
```

```python
query_znsse_compounds(x_Se_target=None)
    │
    ├─ query_url = base_url + "?species(Zn,S,Se),nspecies=3,Egap"
    │
    ├─ response = requests.get(query_url, timeout=30)
    │  ├─ Handles: Timeout, RequestException, JSONDecodeError
    │  └─ Returns: [] if error
    │
    ├─ Parse JSON response → List[AFLOWEntry]
    │
    ├─ IF x_Se_target is not None:
    │  └─ filtered = _filter_by_composition(entries, x_Se_target, tolerance=0.1)
    │
    └─ Returns: List[AFLOWEntry]
```

```python
_vegard_law(x_Se)
    │
    ├─ Eg = (1 - x_Se) * Eg_ZnS + x_Se * Eg_ZnSe
    ├─ Eg -= bowing * x_Se * (1 - x_Se)
    │
    └─ Returns: Eg (float)
```

---

## Data Flow Diagram

### Overview: Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        demo.py                               │
│                         main()                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────────────────────┐
                              │                                             │
                              ▼                                             ▼
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────┐
│         Part 1: demo_interface()                 │   │  Part 2: demo_optimization()    │
├─────────────────────────────────────────────────┤   ├─────────────────────────────────┤
│                                                  │   │                                 │
│  ZnSSeInterface(mode, egap_method)              │   │  ZnSSeInterface()               │
│         │                                        │   │         │                       │
│         ├─ AFLOWInterface() [attempts AFLOW]    │   │         │                       │
│         ├─ _init_hardware() [attempts RPi]      │   │  GPOptimizer(interface)         │
│         └─ Fallback: simulated mode             │   │         │                       │
│                                                  │   │         ├─ Random: 4 samples    │
│  FOR x_Se in [0.0, 0.25, 0.50, 0.75, 1.0]:      │   │         │   └─ measure_voltage()│
│      compute_bandgap(x_Se)                       │   │         │                       │
│         └─ AFLOW query → 404 → Vegard's law     │   │         ├─ Build GP model       │
│                                                  │   │         │   └─ fit(X, y)        │
│      composition_string(x_Se)                    │   │         │                       │
│         └─ "ZnS{1-x}Se{x}"                       │   │         ├─ GP-guided: 6 samples │
│                                                  │   │         │   ├─ EI(Eg)           │
│  Output: Bandgap table vs TiO₂                   │   │         │   └─ measure_voltage()│
│                                                  │   │         │                       │
└─────────────────────────────────────────────────┘   │         └─ Returns: results_df  │
                                                       │                                 │
                              ┌────────────────────────┤  Output: optimization_plot.png  │
                              │                        └─────────────────────────────────┘
                              │                                  │
                              ▼                                  │
┌─────────────────────────────────────────────────┐             │
│     Part 3: analyze_results(optimizer, df)       │◄────────────┘
├─────────────────────────────────────────────────┤
│                                                  │
│  best_idx = argmax(df['Voc_V'])                 │
│  best = df.loc[best_idx]                        │
│                                                  │
│  improvement = (best_Voc - initial_Voc) / ...   │
│                                                  │
│  compare_to_tio2(best['x_Se'])                  │
│      └─ compute_bandgap(x_Se)                   │
│      └─ delta_Eg = Eg_znsse - Eg_TiO2           │
│                                                  │
│  GP model performance:                          │
│      ├─ MAE = mean(|predicted - actual|)        │
│      └─ RMSE = sqrt(mean((predicted - actual)²))│
│                                                  │
│  Output: Results analysis report                │
│                                                  │
└─────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────┐
│    Part 4: demo_full_exploration()               │
├─────────────────────────────────────────────────┤
│                                                  │
│  ZnSSeInterface()                               │
│                                                  │
│  FOR x_Se in linspace(0, 1, 50):                │
│      measure_voltage(x_Se)                       │
│         ├─ compute_bandgap(x_Se)                │
│         └─ Voc = 0.3 + 0.15*Eg + noise          │
│                                                  │
│  df = DataFrame(50 rows × 3 cols)               │
│                                                  │
│  optimal_idx = argmax(df['Voc_V'])              │
│                                                  │
│  Plot:                                           │
│      ├─ Bandgap vs Composition                  │
│      └─ Voltage vs Composition (mark optimal)   │
│                                                  │
│  Output: composition_space_map.png               │
│                                                  │
└─────────────────────────────────────────────────┘
                              │
                              ▼
                         ┌─────────┐
                         │  Done!  │
                         └─────────┘
```

---

## Function Call Statistics

### Total Function Calls (Approximate)

| Module | Function | Calls per Run |
|--------|----------|---------------|
| **znsse_interface.py** | | |
| | `__init__()` | 3 (one per demo part) |
| | `compute_bandgap()` | 1,015 (5 + 10×101 + 50) |
| | `measure_voltage()` | 60 (0 + 10 + 50) |
| | `composition_string()` | 15 |
| | `compare_to_tio2()` | 1 |
| | `_init_hardware()` | 3 |
| **aflow_api.py** | | |
| | `__init__()` | 3 |
| | `get_bandgap()` | 1,015 (called by compute_bandgap) |
| | `query_znsse_compounds()` | 1,015 |
| | `_vegard_law()` | 1,015 (AFLOW fallback) |
| **gp_optimizer.py** | | |
| | `__init__()` | 1 |
| | `optimize()` | 1 |
| | `_build_gp_model()` | 7 (1 initial + 6 re-trains) |
| | `_measure_composition()` | 10 |
| | `_propose_next_composition()` | 6 (iterations 4-9) |
| | `_expected_improvement()` | 6 |
| | `plot_results()` | 1 |
| **demo.py** | | |
| | `main()` | 1 |
| | `demo_interface()` | 1 |
| | `demo_optimization()` | 1 |
| | `analyze_results()` | 1 |
| | `demo_full_exploration()` | 1 |
| | `print_header()` | 5 |

**Total**: ~3,200+ function calls

**Most Called**:
1. `compute_bandgap()` - 1,015 calls
2. `_vegard_law()` - 1,015 calls (AFLOW fallback)
3. `query_znsse_compounds()` - 1,015 calls (HTTP requests)

---

## Performance Metrics

### Timing Breakdown (Simulated Mode)

| Phase | Duration | % Total | Main Bottleneck |
|-------|----------|---------|----------------|
| Part 1: Interface Demo | ~0.5s | 3% | AFLOW HTTP queries (5× 404 responses) |
| Part 2: Optimization | ~8.0s | 53% | GP model re-training (6× fits) |
| Part 2: EI Evaluation | ~3.0s | 20% | 600 bandgap calculations (6 iter × 100 candidates) |
| Part 3: Analysis | ~0.1s | 1% | DataFrame operations |
| Part 4: Full Exploration | ~3.5s | 23% | 50 voltage measurements |
| **Total** | **~15s** | **100%** | |

### Memory Usage

| Object | Size | Peak Usage |
|--------|------|------------|
| `ZnSSeInterface` | ~1 KB | 3 instances |
| `GPOptimizer` | ~50 KB | 1 instance (with 10-point history) |
| `GaussianProcessRegressor` | ~100 KB | 1 instance (trained model) |
| `results_df` (Part 2) | ~5 KB | 10 rows × 7 columns |
| `df` (Part 4) | ~10 KB | 50 rows × 3 columns |
| **Total Peak** | **~200 KB** | |

### HTTP Requests (AFLOW API)

| Phase | HTTP Calls | Expected Response |
|-------|-----------|-------------------|
| Part 1 | 5 | 404 (no ZnSSe data) |
| Part 2 - Random | 4 | 404 |
| Part 2 - GP candidates | 600 (6 iter × 100) | 404 |
| Part 2 - GP measure | 6 | 404 |
| Part 4 | 50 | 404 |
| **Total** | **665** | All fallback to Vegard's law |

**Note**: AFLOW database does not currently contain ZnSSe alloy data, so all queries return 404 and fallback to Vegard's law. This is expected behavior and demonstrates the robust dual-fallback architecture.

---

## Error Handling

### Hardware Fallback Chain

```
Requested: mode='hardware_with_fallback'
    ↓
_init_hardware()
    ├─ Try: import spidev, RPi.GPIO
    │   └─ ImportError → hardware_success = False
    │
    ├─ If hardware_success == False:
    │   ├─ mode == 'hardware' → Raise RuntimeError
    │   └─ mode == 'hardware_with_fallback' → Fallback to 'simulated'
    │
    └─ Print: "Hardware unavailable, falling back to simulated mode"
```

### AFLOW Fallback Chain

```
Requested: egap_method='aflow_with_fallback'
    ↓
compute_bandgap(x_Se, method='aflow_with_fallback')
    ↓
aflow_interface.get_bandgap(x_Se, method='aflow_with_fallback')
    ├─ query_znsse_compounds(x_Se_target=x_Se)
    │   ├─ requests.get(query_url, timeout=30)
    │   │   ├─ HTTP 404 → entries = []
    │   │   ├─ Timeout → entries = []
    │   │   └─ RequestException → entries = []
    │   │
    │   └─ Returns: []
    │
    ├─ IF entries == []:
    │   ├─ method == 'aflow' → Raise ValueError
    │   └─ method == 'aflow_with_fallback' → Fallback
    │
    └─ _vegard_law(x_Se)
        └─ Returns: (Eg, 'vegard')
```

### Exception Handling in main()

```python
try:
    # All 4 demo parts execute
    demo_interface()
    optimizer, results_df = demo_optimization()
    analyze_results(optimizer, results_df)
    demo_full_exploration()

except KeyboardInterrupt:
    print("Demo interrupted by user.")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
```

---

## Output Files

### File: `results/optimization_results.csv`

**Format**: CSV (comma-separated values)

**Structure**:
```
iteration,x_Se,formula,Eg_eV,Voc_V,predicted_Voc_V,uncertainty_V
0,0.6373,ZnS0.36Se0.64,2.90,0.735,,
1,0.2694,ZnS0.73Se0.27,3.21,0.782,,
2,0.4474,ZnS0.55Se0.45,3.06,0.759,,
3,0.7155,ZnS0.28Se0.72,2.83,0.725,,
4,0.1234,ZnS0.88Se0.12,3.42,0.813,0.811,0.008
5,0.0892,ZnS0.91Se0.09,3.49,0.824,0.822,0.007
...
```

**Rows**: 10 (one per iteration)

**Columns**:
- `iteration`: 0-9
- `x_Se`: Selenium composition (0.0-1.0)
- `formula`: Chemical formula (e.g., "ZnS0.88Se0.12")
- `Eg_eV`: Bandgap (eV)
- `Voc_V`: Measured voltage (V)
- `predicted_Voc_V`: GP prediction (V) [NaN for first 4 random samples]
- `uncertainty_V`: GP uncertainty (V) [NaN for first 4 random samples]

### File: `results/optimization_plot.png`

**Format**: PNG image (150 DPI)

**Structure**: 2-panel figure (12" × 4")

**Panel 1** (left): Voltage vs Bandgap
- X-axis: Bandgap (eV), range 2.6-3.8
- Y-axis: Voltage (V)
- Blue line: GP mean prediction
- Blue shading: 95% confidence interval (μ ± 1.96σ)
- Red dots: Measured data points (10 total)
- Green dashed line: TiO₂ reference (3.2 eV)

**Panel 2** (right): Voltage vs Composition
- X-axis: Se composition (x_Se), range 0-1
- Y-axis: Voltage (V)
- Colored dots: Measurements (color gradient by iteration)
- Numbers: Iteration labels (1-10)
- Labels: "ZnS" (left), "ZnSe" (right)

### File: `results/composition_space_map.png`

**Format**: PNG image (150 DPI)

**Structure**: 2-panel figure (12" × 4")

**Panel 1** (left): Bandgap vs Composition
- X-axis: Se composition (x_Se), range 0-1
- Y-axis: Bandgap (eV), range ~2.7-3.7
- Blue line: Vegard's law with bowing
- Green dashed line: TiO₂ reference (3.2 eV)
- Text boxes: ZnS (3.68 eV), ZnSe (2.70 eV)

**Panel 2** (right): Voltage vs Composition
- X-axis: Se composition (x_Se), range 0-1
- Y-axis: Voltage (V)
- Red line: Simulated voltage measurements (50 points)
- Light red shading: ±10 mV noise band
- Green star: Optimal composition (maximum Voc)

---

## Developer Notes

### Modifying Optimization Parameters

**Change number of iterations**:
```python
# In demo.py line 81:
results_df = optimizer.optimize(n_iterations=20, n_initial=5)  # Default: 10, 4
```

**Change Expected Improvement exploration**:
```python
# In demo.py line 75:
optimizer = GPOptimizer(interface, xi=0.05, random_state=42)  # Default: xi=0.01
```
- Higher `xi` → More exploration (wider search)
- Lower `xi` → More exploitation (focus on best regions)

**Change composition resolution**:
```python
# In demo.py line 167:
n_points = 100  # Default: 50
x_Se_range = np.linspace(0.0, 1.0, n_points)
```

### Switching Between Modes

**Hardware mode** (Raspberry Pi with MCP3008):
```python
# In demo.py lines 46, 74, 164:
interface = ZnSSeInterface(mode='hardware', egap_method='aflow_with_fallback')
```

**Pure AFLOW mode** (strict, no fallback):
```python
interface = ZnSSeInterface(mode='simulated', egap_method='aflow')
# Warning: Will raise error if AFLOW data unavailable
```

**Pure Vegard mode** (fastest):
```python
interface = ZnSSeInterface(mode='simulated', egap_method='vegard')
```

### Adding Custom Measurements

**Inject real experimental data**:
```python
# After line 81 in demo.py (after random samples, before GP):
optimizer.X_bandgaps.append(3.42)
optimizer.y_voltages.append(0.815)
optimizer.x_compositions.append(0.12)
```

**Use custom bandgap function**:
```python
# Override compute_bandgap in znsse_interface.py:
def compute_bandgap(self, x_Se, method=None):
    # Load from external database/file
    Eg = your_custom_function(x_Se)
    return Eg, 'custom'
```

---

## Troubleshooting

### Issue: "AFLOW query timeout"

**Symptoms**: HTTP requests hang for 30 seconds

**Fix**:
```python
# In demo.py, reduce AFLOW timeout:
aflow = AFLOWInterface(timeout=5)  # Default: 30
```

**Alternative**: Use pure Vegard mode (no AFLOW queries)

### Issue: "Hardware initialization failed"

**Symptoms**: `_init_hardware()` returns False

**Causes**:
1. Not running on Raspberry Pi
2. SPI not enabled (`sudo raspi-config` → Interfacing → SPI → Enable)
3. MCP3008 not connected properly
4. Missing dependencies (`pip install RPi.GPIO spidev`)

**Fix**: Use `mode='simulated'` or `mode='hardware_with_fallback'`

### Issue: GP model convergence warnings

**Symptoms**: `ConvergenceWarning` from scikit-learn

**Cause**: Noise level parameter hitting lower bound

**Fix**: Warnings are suppressed in code (lines 155-157 of gp_optimizer.py). This is expected for low-noise simulated data.

### Issue: Plots not displaying

**Symptoms**: Figures saved but not shown

**Fix**: Add `plt.show()` after `plt.savefig()` in demo.py:
```python
# Line 227:
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()  # Add this line
```

---

## Advanced Topics

### Parallel Candidate Evaluation

Current implementation evaluates 100 candidates sequentially in `_propose_next_composition()`. For large `n_candidates`, consider parallelization:

```python
from multiprocessing import Pool

def _propose_next_composition(self, n_candidates=100):
    x_Se_candidates = np.linspace(0.0, 1.0, n_candidates)

    # Parallel bandgap calculation
    with Pool(processes=4) as pool:
        Eg_list = pool.map(lambda x: self.interface.compute_bandgap(x)[0], x_Se_candidates)

    Eg_candidates = np.array(Eg_list).reshape(-1, 1)
    # ... rest of function
```

### Custom Acquisition Functions

Replace Expected Improvement with other strategies:

**Upper Confidence Bound (UCB)**:
```python
def _ucb(self, X_candidates, kappa=2.0):
    mu, sigma = self.gp_model.predict(X_candidates, return_std=True)
    return mu + kappa * sigma  # Exploration-exploitation trade-off
```

**Probability of Improvement (PI)**:
```python
from scipy.stats import norm

def _probability_of_improvement(self, X_candidates):
    mu, sigma = self.gp_model.predict(X_candidates, return_std=True)
    f_best = np.max(self.y_voltages)
    Z = (mu - f_best - self.xi) / sigma
    return norm.cdf(Z)
```

### Multi-Objective Optimization

Extend to optimize both Voc and Jsc simultaneously:

```python
# Modify measure_voltage to return both metrics:
def measure_voltage(self, x_Se, light_intensity=1.0):
    # ... existing Voc calculation ...

    # Add Jsc calculation (current density)
    Jsc = self._calculate_jsc(Eg, light_intensity)

    return {
        # ... existing fields ...
        'Jsc_mA_cm2': Jsc
    }

# Use multi-objective GP or Pareto optimization
```

---

## References

### Code Files

- `demo.py` (298 lines) - Main demonstration script
- `znsse_interface.py` (471 lines) - Material interface
- `gp_optimizer.py` (476 lines) - Bayesian optimization
- `aflow_api.py` (410 lines) - AFLOW database integration

### Key Algorithms

1. **Vegard's Law with Bowing**: `znsse_interface.py:206-208`
2. **Expected Improvement**: `gp_optimizer.py:162-205`
3. **GP Model Training**: `gp_optimizer.py:120-160`
4. **Voltage Simulation**: `znsse_interface.py:263-275`

### External Dependencies

- `numpy` - Array operations and random sampling
- `pandas` - Data storage and analysis
- `matplotlib` - Visualization
- `scikit-learn` - Gaussian Process Regression
- `scipy` - Statistical functions (norm.cdf, norm.pdf)
- `requests` - AFLOW HTTP queries

---

## Version History

**Version 1.0** (November, 2025)
- Initial release
- Covers demo.py execution flow
- Documents all function calls
- Includes performance metrics

---

## Contact

**Questions about execution flow?**
- See `README.md` for project overview
- See `README_INSTALLATION.md` for setup
- See `README_DEPLOYMENT.md` for production deployment
- See `README_OVERLEAF.md` for paper compilation

**Code Authors**:
- Samridhi Chordia
- Johns Hopkins University
- Department of Materials Science and Engineering

**Primary Contact**: schordi2@jhu.edu

---

**End of README_RUN.md**
