# LEGOLAS: ZnS₁₋ₓSeₓ DSSC Optimization (AFLOW Integration Version)

**LEGO based Low cost Autonomous Scientist**

Autonomous composition optimization for ZnS₁₋ₓSeₓ dye-sensitized solar cells using Gaussian Process machine learning with **AFLOW Database Integration**.

---

## What's New in This Version

This is an **enhanced version** of LEGOLAS with integrated AFLOW Materials Database support:

- **Dual Bandgap Calculation**: Choose between Vegard's law OR AFLOW API for DFT-calculated bandgaps
- **Dual Measurement Modes**: Simulated physics models OR real hardware (Raspberry Pi + MCP3008)
- **Automatic Fallback**: Seamlessly falls back to Vegard's law if AFLOW data unavailable
- **Future-Ready**: Designed for POCC (Partial Occupation) ensemble-averaged bandgaps

---

## Overview

This project implements the LEGOLAS framework for optimizing ZnS₁₋ₓSeₓ semiconductor alloy composition in dye-sensitized solar cells (DSSCs), based on the methodology from:

> **Chordia, S., Lee, J., & Oses, C. (2025)**
> *"Machine-Learning Based Optimization of ZnS(1-x)Se(x) Composition in Dye-Sensitized Solar Cells Using LEGOLAS Framework"*
> Department of Materials Science and Engineering, Johns Hopkins University

### Key Features

- ✅ **Gaussian Process Optimization** - Minimal measurements needed (< 10 experiments)
- ✅ **AFLOW Database Integration** - DFT-quality bandgap data with automatic fallback
- ✅ **Dual Operation Modes** - Hardware (RPi+ADC) or simulated measurements
- ✅ **Closed-Loop Automation** - Iterative experimental design
- ✅ **Hardware Compatible** - MCP3008 ADC + Raspberry Pi interface
- ✅ **Modular Design** - Easy to extend to other material systems

---

## Quick Start

### Installation

```bash
# Clone or download the project
cd LEGOLAS_ZnSSe

# Install dependencies
pip install -r requirements.txt

# Test installation
python3 znsse_interface.py
```

### Run Complete Demo

```bash
# Basic demo (simulated mode, Vegard's law)
python3 demo.py

# Use AFLOW database for bandgap calculations
python3 demo.py --egap_method aflow

# Use hardware mode with AFLOW bandgaps
python3 demo.py --mode hardware --egap_method aflow

# Use hardware with fallback to simulation (recommended for real hardware)
python3 demo.py --mode hardware_with_fallback --egap_method aflow_with_fallback

# Save figures and data
python3 demo.py --save-figures --save-data --mode simulated --egap_method vegard
```

This will:
1. Demonstrate ZnSSe material properties
2. Run Gaussian Process optimization (10 iterations)
3. Analyze and visualize results
4. Map full composition space

### Command-Line Arguments

**Measurement Mode** (`--mode`):
- `simulated` (default): Physics-based simulation, always available
- `hardware`: MCP3008 ADC + Raspberry Pi, requires hardware setup
- `hardware_with_fallback`: Try hardware first, fallback to simulation if unavailable

**Bandgap Calculation** (`--egap_method`):
- `vegard` (default): Vegard's law with bowing parameter, always available
- `aflow`: AFLOW database DFT-calculated bandgaps, requires API access
- `aflow_with_fallback`: Try AFLOW first, fallback to Vegard's law if unavailable

**Output Options**:
- `--save-figures`: Save visualization plots to `paper/figures/`
- `--save-data`: Export CSV data to `paper/data/`
- `--output-dir DIR`: Custom output directory (default: `paper`)
- `--log FILENAME`: Log file name to save execution results (default: `demo.log`)

---

## Project Structure

```
LEGOLAS_ZnSSe/
├── znsse_interface.py      # Hardware/simulation interface
├── gp_optimizer.py          # Gaussian Process optimizer
├── generate_dataset.py      # Training data generator
├── demo.py                  # Complete demonstration
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── results/                 # Output files
```

---

## Material System: ZnS₁₋ₓSeₓ

### Composition Variable

- **x_Se**: Selenium fraction (0.0 to 1.0)
- **x_Se = 0.0**: Pure ZnS (bandgap = 3.68 eV)
- **x_Se = 1.0**: Pure ZnSe (bandgap = 2.70 eV)
- **Formula**: ZnS₍₁₋ₓ₎Seₓ

### Bandgap Engineering

Uses **Vegard's Law with Bowing Parameter**:

```
Eg(x) = (1-x)·Eg_ZnS + x·Eg_ZnSe - b·x·(1-x)

Where:
- Eg_ZnS = 3.68 eV
- Eg_ZnSe = 2.70 eV
- b = 0.50 eV (bowing parameter)
```

### Example Compositions

| x_Se | Formula | Bandgap (eV) | Application |
|------|---------|--------------|-------------|
| 0.00 | ZnS | 3.68 | High Voc, UV absorption |
| 0.25 | ZnS₀.₇₅Se₀.₂₅ | 3.31 | Balanced performance |
| 0.50 | ZnS₀.₅₀Se₀.₅₀ | 3.07 | Visible light |
| 0.75 | ZnS₀.₂₅Se₀.₇₅ | 2.83 | High Jsc potential |
| 1.00 | ZnSe | 2.70 | Maximum visible absorption |

---

## Optimization Methodology

### 1. Gaussian Process Regression

**Model**: Predicts open-circuit voltage (Voc) as a function of bandgap energy (Eg)

```
μ(E*_g) = k*ᵀ [K + σ²_n·I]⁻¹ y

Where:
- k* = kernel values (RBF + White kernel)
- K = covariance matrix
- σ²_n = noise variance
- y = observed voltages
```

### 2. Expected Improvement Acquisition

**Proposes next composition to test**:

```
EI(x) = (μ(x) - f_best - ξ)·Φ(Z) + σ(x)·φ(Z)

Where:
- Z = (μ(x) - f_best - ξ) / σ(x)
- ξ = exploration parameter (default: 0.01)
- Φ, φ = normal CDF and PDF
```

### 3. Closed-Loop Workflow

```
Start → Initial Random Sampling (4 points)
    ↓
Train GP Model on (Eg, Voc) pairs
    ↓
Predict Optimal Bandgap (argmax EI)
    ↓
Map Bandgap → Composition
    ↓
Measure Voltage
    ↓
Update GP Model
    ↓
Converged? → Yes → Optimal Composition Found
    ↓ No
Return to GP Model Training
```

---

## Usage Examples

### Basic Interface Usage

```python
from znsse_interface import ZnSSeInterface

# Initialize interface - simulated mode with Vegard's law
interface = ZnSSeInterface(mode='simulated', egap_method='vegard')

# Or use AFLOW database for bandgaps
interface = ZnSSeInterface(mode='simulated', egap_method='aflow_with_fallback')

# Or use hardware measurements
interface = ZnSSeInterface(mode='hardware_with_fallback', egap_method='vegard')

# Calculate bandgap for composition
x_Se = 0.30
Eg = interface.compute_bandgap(x_Se)
print(f"ZnS0.70Se0.30 bandgap: {Eg:.2f} eV")
# Output: ZnS0.70Se0.30 bandgap: 3.27 eV (Vegard's law)

# Measure voltage
measurement = interface.measure_voltage(x_Se)
print(f"Voc: {measurement['Voc_V']:.3f} V")
```

### Run Optimization

```python
from znsse_interface import ZnSSeInterface
from gp_optimizer import GPOptimizer

# Initialize with chosen mode and bandgap method
interface = ZnSSeInterface(mode='simulated', egap_method='vegard')
# Or use AFLOW: interface = ZnSSeInterface(mode='simulated', egap_method='aflow_with_fallback')
optimizer = GPOptimizer(interface, xi=0.01)

# Run optimization (10 iterations, 4 initial random)
results_df = optimizer.optimize(n_iterations=10, n_initial=4)

# Visualize
optimizer.plot_results(save_path='results/optimization.png')

# Get best composition
best_idx = results_df['Voc_V'].idxmax()
best = results_df.loc[best_idx]
print(f"Best: {best['formula']} at {best['Voc_V']:.3f} V")
```

---

## Hardware Integration

### MCP3008 ADC + Raspberry Pi Setup

The framework supports real hardware measurements using MCP3008 analog-to-digital converter.

**Wiring Diagram** (from paper):

```
MCP3008 Pin Layout:
CH0 ─┬─ Not used
CH1 ─┼─ DSSC voltage input (recommended)
CH2 ─┼─ Alternative channels
...  ┆
CH7 ─┘

VDD  ── RPi 3.3V
VREF ── RPi 3.3V
AGND ── RPi GND
DGND ── RPi GND

CLK  ── RPi SCLK (GPIO 11)
DOUT ── RPi MISO (GPIO 9)
DIN  ── RPi MOSI (GPIO 10)
CS   ── RPi CE0  (GPIO 8)
```

**Enable Hardware Mode**:

```python
interface = ZnSSeInterface(mode='hardware')
# Automatically initializes MCP3008 via SPI
```

**Requirements**:
- Raspberry Pi (any model with GPIO)
- MCP3008 ADC chip
- SPI enabled: `sudo raspi-config` → Interfacing Options → SPI → Enable
- Python packages: `spidev`, `RPi.GPIO`

---

## Results & Performance

### Typical Optimization Results

From demo run (simulated measurements):

```
Best Composition: ZnS₀.₇₅Se₀.₂₅
├─ x_Se: 0.250
├─ Bandgap: 3.31 eV
├─ Voltage: 0.726 V
└─ Found at iteration: 7/10

Performance:
├─ Initial Voc: 0.653 V
├─ Best Voc: 0.726 V
├─ Improvement: +11.2%
└─ GP Model MAE: 0.008 V
```

### Comparison with TiO₂

| Property | TiO₂ | Optimal ZnSSe | Advantage |
|----------|------|---------------|-----------|
| Bandgap | 3.20 eV | 3.31 eV | ↑ 3.4% (Higher Voc) |
| Voc range | 0.60-0.70 V | 0.65-0.75 V | ↑ Better voltage |
| Composition range | Fixed | Tunable (0.0-1.0) | ✓ Design flexibility |
| Cost | Moderate | Low | ✓ Earth-abundant |

---

## Physical Interpretation

### Bandgap-Voltage Relationship

Open-circuit voltage follows empirical correlation:

```
Voc ≈ 0.30 + 0.15·Eg + 0.025·ln(I)

Where:
- 0.30 V = base voltage
- 0.15 eV⁻¹ = bandgap sensitivity
- I = light intensity (relative)
```

### Optimal Composition Ranges

**For Different Applications**:

1. **High Voltage** (Voc maximization)
   - x_Se: 0.0 - 0.3
   - Eg: 3.2 - 3.68 eV
   - Use case: Space applications, low-light

2. **Visible Light Absorption** (Jsc maximization)
   - x_Se: 0.5 - 1.0
   - Eg: 2.70 - 3.0 eV
   - Use case: Standard AM1.5G illumination

3. **Balanced Performance**
   - x_Se: 0.2 - 0.4
   - Eg: 3.0 - 3.4 eV
   - Use case: General DSSC applications

---

## Experimental Protocol

### Materials Needed

1. **Photoanode Fabrication**:
   - ZnS and ZnSe nanoparticles
   - FTO glass
   - Tape (for masking)
   - Hot plate

2. **Dye Sensitization**:
   - Blackberry juice (natural dye, high anthocyanin)
   - Or: Ruthenium dyes (N719, N3)
   - Water for dilution

3. **Cell Assembly**:
   - Graphite pencil (counter electrode)
   - Iodide/triiodide electrolyte (I⁻/I₃⁻)
   - Binder clips
   - Alligator clips

4. **Measurement**:
   - Light source (torch or solar simulator)
   - Multimeter or MCP3008 ADC
   - Raspberry Pi (for automation)

### Procedure Summary

1. **Prepare ZnS₁₋ₓSeₓ paste** with desired composition
2. **Deposit on FTO glass** and sinter
3. **Dye with blackberry juice** (30 min soak)
4. **Assemble sandwich cell** with graphite counter electrode
5. **Add electrolyte** via capillary action
6. **Measure Voc under illumination**
7. **Run GP optimization** to propose next composition
8. **Repeat** until convergence

---

## Advanced Features

### Multi-Objective Optimization

Extend to optimize both Voc AND Jsc:

```python
# Modify objective function
def multi_objective(x_Se):
    measurement = interface.measure_full_iv_curve(x_Se)
    efficiency = measurement['Voc'] * measurement['Jsc'] * measurement['FF']
    return efficiency

optimizer = GPOptimizer(interface, objective=multi_objective)
```

### Custom Light Spectra

Test under space conditions (AM0) or filtered light:

```python
# Simulate AM0 spectrum (space)
interface = ZnSSeInterface(mode='simulated')
interface.set_spectrum('AM0')  # Default is AM1.5G

# Or custom filter
interface.set_filter(lambda_min=400, lambda_max=800)  # nm
```

### Batch Optimization

Optimize multiple compositions in parallel:

```python
compositions = [0.1, 0.3, 0.5, 0.7, 0.9]
results = interface.measure_batch(compositions)
```

---

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**2. MCP3008 not detected (hardware mode)**
```bash
# Check SPI enabled
ls /dev/spidev*
# Should show: /dev/spidev0.0

# If not, enable SPI:
sudo raspi-config
# → Interfacing Options → SPI → Enable
```

**3. GP model convergence issues**
```python
# Increase initial random samples
optimizer.optimize(n_iterations=20, n_initial=8)

# Adjust exploration parameter
optimizer = GPOptimizer(interface, xi=0.05)  # Higher xi = more exploration
```

**4. Low voltage measurements**
```
# Check:
- Light source intensity
- Dye coverage (should be dark purple/blue)
- Electrolyte penetration
- Electrical contacts
```

---

### Related References

1. **AFLOW Database**:
   - Rose, F., et al. (2017). "AFLUX: The LUX materials search API for the AFLOW data repositories." *Computational Materials Science*, 137, 362-370.

2. **LEGOLAS Framework**:
   - Saar, L. (2023). "Low-cost educational guided optimization lab for autonomous science (LEGOLAS)." *MRS Bulletin*, 47, 1078-1082.

3. **POCC Method**:
   - Yang, K., Oses, C., & Curtarolo, S. (2016). "Modeling off-stoichiometry materials with a high-throughput ab-initio approach." *Chemistry of Materials*, 28(18), 6484-6492.

---

## License

MIT License - Free for educational and research use.

---

## Contact & Support

**Authors**:
- Samridhi Chordia (schordi2@jh.edu)

**Issues**: Open an issue on the repository

**Questions**: Contact corresponding author or open a discussion

---

## Acknowledgments

This work builds upon:
- AFLOW materials database (Duke University)
- LEGOLAS educational framework
- scikit-learn machine learning library
- Open-source Python ecosystem

---

**Last Updated**: November 2025
**Version**: 1.0
**Status**: Production Ready ✓
