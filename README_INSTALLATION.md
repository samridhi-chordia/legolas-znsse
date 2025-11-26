# LEGOLAS ZnSSe Installation & Quick Start Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Navigate to project directory
cd LEGOLAS_ZnSSe

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import numpy, pandas, sklearn, matplotlib; print('✓ All dependencies installed')"
```

### Dependencies

```
numpy>=1.21.0       # Scientific computing
pandas>=1.3.0       # Data manipulation
scipy>=1.7.0        # Scientific functions
scikit-learn>=1.0.0 # Machine learning (Gaussian Process)
matplotlib>=3.4.0   # Visualization
```

---

## Quick Start (5 Minutes)

### 1. Test Interface

```bash
python3 znsse_interface.py
```

**Expected output:**
```
============================================================
ZnS(1-x)Se(x) DSSC Interface Demo
============================================================
...
Bandgap vs Composition
x_Se |         Formula |    Bandgap (eV) |    vs TiO2
0.00 |   ZnS1.00Se0.00 |            3.68 |           +0.48
0.25 |   ZnS0.75Se0.25 |            3.34 |           +0.14
...
```

### 2. Generate Training Data

```bash
python3 generate_dataset.py
```

**Creates:** `data/znsse_training_data.csv` (100 samples)

### 3. Run Complete Demo

```bash
python3 demo.py
```

**Runs:**
- Material properties demonstration
- Gaussian Process optimization (10 iterations)
- Results analysis and visualization
- Full composition space mapping

**Creates:**
- `results/optimization_results.csv`
- `results/optimization_plot.png`
- `results/composition_space_map.png`

**Time:** ~30 seconds

---

## Project Structure

```
LEGOLAS_ZnSSe/
├── znsse_interface.py          # ZnSSe hardware/simulation interface (11 KB)
├── gp_optimizer.py              # Gaussian Process optimizer (15 KB)
├── generate_dataset.py          # Dataset generator (3.1 KB)
├── demo.py                      # Complete demonstration (9.9 KB)
├── requirements.txt             # Python dependencies (470 B)
├── README.md                    # Full documentation (12 KB)
├── INSTALLATION.md              # This file
│
├── data/
│   └── znsse_training_data.csv  # Training dataset (9.1 KB, 100 samples)
│
└── results/
    ├── optimization_results.csv         # Optimization history (873 B)
    ├── optimization_plot.png            # GP fit visualization (97 KB)
    └── composition_space_map.png        # Full space map (122 KB)
```

**Total Size:** ~290 KB (code + data + results)

---

## Usage Examples

### Example 1: Basic Interface

```python
from znsse_interface import ZnSSeInterface

# Initialize
interface = ZnSSeInterface(mode='simulated')

# Calculate bandgap
Eg = interface.compute_bandgap(x_Se=0.30)
print(f"Bandgap: {Eg:.2f} eV")  # 3.27 eV

# Measure voltage
result = interface.measure_voltage(x_Se=0.30)
print(f"Voc: {result['Voc_V']:.3f} V")
```

### Example 2: Run Optimization

```python
from znsse_interface import ZnSSeInterface
from gp_optimizer import GPOptimizer

# Setup
interface = ZnSSeInterface(mode='simulated')
optimizer = GPOptimizer(interface)

# Optimize
results = optimizer.optimize(n_iterations=10, n_initial=4)

# Get best
best_idx = results['Voc_V'].idxmax()
print(f"Best: {results.loc[best_idx, 'formula']}")
print(f"Voc: {results.loc[best_idx, 'Voc_V']:.3f} V")
```

### Example 3: Custom Compositions

```python
from znsse_interface import ZnSSeInterface

interface = ZnSSeInterface(mode='simulated')

# Test specific compositions
compositions = [0.0, 0.25, 0.50, 0.75, 1.0]

for x_Se in compositions:
    result = interface.measure_voltage(x_Se)
    print(f"{result['formula']:>15}: Eg={result['Eg_eV']:.2f} eV, Voc={result['Voc_V']:.3f} V")
```

**Output:**
```
  ZnS1.00Se0.00: Eg=3.68 eV, Voc=0.848 V
  ZnS0.75Se0.25: Eg=3.34 eV, Voc=0.809 V
  ZnS0.50Se0.50: Eg=3.07 eV, Voc=0.774 V
  ZnS0.25Se0.75: Eg=2.85 eV, Voc=0.743 V
  ZnS0.00Se1.00: Eg=2.70 eV, Voc=0.720 V
```

---

## Hardware Setup (Optional)

### For Real DSSC Measurements

**Components needed:**
- Raspberry Pi (any model with GPIO)
- MCP3008 ADC chip
- DSSC device
- Connecting wires

**Enable Hardware Mode:**

```python
interface = ZnSSeInterface(mode='hardware')
# Automatically initializes MCP3008 via SPI
```

**Raspberry Pi Configuration:**

```bash
# Enable SPI interface
sudo raspi-config
# → Interfacing Options → SPI → Enable

# Install additional packages
pip install spidev RPi.GPIO

# Test connection
python3 -c "import spidev; spi = spidev.SpiDev(); spi.open(0,0); print('✓ SPI working')"
```

---

## Troubleshooting

### Issue: Import errors

```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: No module named 'sklearn'

```bash
# Solution: Install scikit-learn
pip install scikit-learn>=1.0.0
```

### Issue: matplotlib backend errors

```bash
# Solution: Use non-GUI backend
export MPLBACKEND=Agg
python3 demo.py
```

### Issue: Permission denied (Raspberry Pi)

```bash
# Solution: Add user to spi and gpio groups
sudo usermod -a -G spi,gpio $USER
# Log out and back in
```

---

## Validation

### Check Installation

```bash
# Run all tests
python3 << EOF
from znsse_interface import ZnSSeInterface
from gp_optimizer import GPOptimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✓ All modules imported successfully")

# Quick test
interface = ZnSSeInterface(mode='simulated')
Eg = interface.compute_bandgap(0.5)
assert 3.0 < Eg < 3.1, f"Bandgap calculation failed: {Eg}"
print(f"✓ Bandgap calculation correct: {Eg:.2f} eV")

print("\n✓✓✓ Installation validated successfully ✓✓✓")
EOF
```

**Expected output:**
```
[ZnSSe Interface] Initialized in 'simulated' mode
[ZnSSe Interface] Bandgap range: 2.70 - 3.68 eV
✓ All modules imported successfully
✓ Bandgap calculation correct: 3.07 eV

✓✓✓ Installation validated successfully ✓✓✓
```

---

## Performance Benchmarks

### Typical Run Times (on modern laptop)

| Task | Time | Output |
|------|------|--------|
| Interface demo | 2 sec | Terminal output |
| Generate dataset (100 samples) | 5 sec | CSV file (9 KB) |
| Optimization (10 iterations) | 10 sec | CSV + 2 PNG files |
| Full demo | 30 sec | All outputs |

### Memory Usage

- Peak RAM: ~150 MB
- Disk space: ~300 KB (including results)

---

## Next Steps

1. **Run the demo**: `python3 demo.py`
2. **Read the documentation**: Open `README.md`
3. **Explore the code**: Start with `znsse_interface.py`
4. **Customize parameters**: Modify `demo.py` for your needs
5. **Test hardware** (if available): Switch to `mode='hardware'`

---

## Support

- **Documentation**: `README.md`
- **Examples**: See usage examples above
- **Issues**: Check troubleshooting section
- **Contact**: corey@jhu.edu

---

**Installation Complete! 🎉**

Run `python3 demo.py` to see the framework in action.
