#!/usr/bin/env python3
"""
ZnS(1-x)Se(x) DSSC Hardware Interface
======================================

LEGOLAS framework interface for ZnS(1-x)Se(x) dye-sensitized solar cells.
Based on the methodology from Chordia et al. (2025).

This module provides:
- Bandgap calculations using Vegard's law with bowing parameter
- AFLOW POCC integration for DFT bandgap data
- Voltage measurement simulation/hardware interface
- Composition-to-property mappings

Key Features:
- Full composition range: x_Se = 0.0 (ZnS) to 1.0 (ZnSe)
- Bandgap range: 3.68 eV (ZnS) to 2.70 eV (ZnSe)
- POCC ensemble-averaged bandgaps for disordered alloys
- Compatible with Gaussian Process optimization

Authors: Based on work by Samridhi Chordia, Jaehyung Lee, and Corey Oses
         Johns Hopkins University, Department of Materials Science and Engineering
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

# Import AFLOW API interface (if available)
try:
    from aflow_api import AFLOWInterface
    AFLOW_AVAILABLE = True
except ImportError:
    AFLOW_AVAILABLE = False
    warnings.warn("aflow_api module not available. AFLOW integration disabled.")


class ZnSSeInterface:
    """
    Hardware interface for ZnS(1-x)Se(x) DSSC measurements.

    This class handles:
    - Bandgap calculations via Vegard's law or AFLOW API
    - Voltage measurements (simulated or hardware)
    - Composition string generation
    - POCC integration for DFT bandgap data
    - Multiple modes: hardware/simulated measurement, vegard/aflow bandgap calculation
    """

    def __init__(self, mode='simulated', egap_method='vegard'):
        """
        Initialize ZnSSe interface.

        Parameters:
        -----------
        mode : str
            Voltage measurement mode:
            - 'simulated': Physics-based simulation (always works)
            - 'hardware': MCP3008 ADC + Raspberry Pi (strict mode - fails if unavailable)
            - 'hardware_with_fallback': Try hardware first, use simulation if unavailable (recommended)

        egap_method : str
            Bandgap calculation method:
            - 'vegard': Vegard's law with bowing parameter (fast, always available)
            - 'aflow': AFLOW database query (strict mode - fails if unavailable)
            - 'aflow_with_fallback': Try AFLOW first, use Vegard's on failure (recommended)
        """
        self.mode = mode
        self.egap_method = egap_method
        self.original_mode = mode  # Track requested mode before fallback

        # Material properties from literature and POCC calculations
        self.Eg_ZnS = 3.68   # eV (bulk ZnS bandgap)
        self.Eg_ZnSe = 2.70  # eV (bulk ZnSe bandgap)
        self.bowing = 0.50   # eV (bowing parameter for ZnSSe)

        # TiO2 reference (common DSSC benchmark)
        self.Eg_TiO2 = 3.2   # eV

        # Initialize AFLOW interface if needed
        self.aflow_interface = None
        if egap_method in ['aflow', 'aflow_with_fallback'] and AFLOW_AVAILABLE:
            try:
                self.aflow_interface = AFLOWInterface()
                print(f"[ZnSSe Interface] AFLOW API integration enabled")
            except Exception as e:
                warnings.warn(f"AFLOW initialization failed: {e}. Using Vegard's law.")
                self.egap_method = 'vegard'

        # Attempt hardware initialization if requested
        if mode in ['hardware', 'hardware_with_fallback']:
            hardware_success = self._init_hardware()

            if not hardware_success:
                if mode == 'hardware':
                    # Strict hardware mode - fail if unavailable
                    raise RuntimeError("Hardware mode requested but initialization failed. "
                                     "Use 'hardware_with_fallback' for automatic fallback to simulation.")
                else:
                    # Fallback mode - switch to simulation
                    print(f"[ZnSSe Interface] Hardware unavailable, falling back to simulated mode")
                    self.mode = 'simulated'

        print(f"[ZnSSe Interface] Initialized:")
        print(f"  Measurement mode: '{self.mode}'{' (fallback from hardware)' if self.mode != self.original_mode else ''}")
        print(f"  Egap method: '{self.egap_method}'")
        print(f"  Bandgap range: {self.Eg_ZnSe:.2f} - {self.Eg_ZnS:.2f} eV")

    def _init_hardware(self) -> bool:
        """
        Initialize MCP3008 ADC hardware for voltage measurements.

        Returns:
        --------
        success : bool
            True if hardware initialized successfully, False otherwise
        """
        print("[Hardware] Attempting to initialize MCP3008 ADC...")

        try:
            import spidev
            import RPi.GPIO as GPIO

            print("[Hardware] Importing spidev and RPi.GPIO: OK")

            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Bus 0, Device 0
            self.spi.max_speed_hz = 1350000

            print("[Hardware] MCP3008 ADC initialized successfully")
            print("[Hardware] SPI: Bus 0, Device 0, Speed: 1.35 MHz")
            print("[Hardware] Connect DSSC to CH1")
            return True

        except ImportError as e:
            print(f"[Hardware] Import failed: {e}")
            print("[Hardware] Missing dependencies: RPi.GPIO or spidev")
            print("[Hardware] Install with: pip install RPi.GPIO spidev")
            return False

        except Exception as e:
            print(f"[Hardware] Initialization failed: {e}")
            print("[Hardware] Possible causes:")
            print("[Hardware]   - Not running on Raspberry Pi")
            print("[Hardware]   - SPI not enabled (run: sudo raspi-config)")
            print("[Hardware]   - MCP3008 not connected properly")
            return False

    def compute_bandgap(self, x_Se: float, method=None) -> Tuple[float, str]:
        """
        Calculate bandgap for ZnS(1-x)Se(x) composition.

        Uses Vegard's law with bowing parameter or AFLOW database:
        Eg(x) = (1-x)*Eg_ZnS + x*Eg_ZnSe - b*x*(1-x)

        Parameters:
        -----------
        x_Se : float
            Selenium composition (0.0 to 1.0)
        method : str, optional
            Override default egap_method:
            - 'vegard': Vegard's law with bowing
            - 'aflow': AFLOW database query (with automatic fallback)
            - 'aflow_with_fallback': Try AFLOW first, use Vegard's on failure
            - None: Use self.egap_method

        Returns:
        --------
        Eg : float
            Bandgap in eV
        source : str
            Data source ('vegard', 'aflow', 'aflow_fallback')

        Examples:
        ---------
        >>> interface = ZnSSeInterface()
        >>> interface.compute_bandgap(0.0)  # Pure ZnS
        (3.68, 'vegard')
        >>> interface.compute_bandgap(0.5)  # 50-50 alloy
        (3.07, 'vegard')
        >>> interface.compute_bandgap(1.0)  # Pure ZnSe
        (2.70, 'vegard')
        """
        if not (0.0 <= x_Se <= 1.0):
            raise ValueError(f"x_Se must be in [0, 1], got {x_Se}")

        # Use instance method if not specified
        if method is None:
            method = self.egap_method

        # Use AFLOW API if requested and available
        if method in ['aflow', 'aflow_with_fallback'] and self.aflow_interface is not None:
            try:
                Eg, source = self.aflow_interface.get_bandgap(x_Se, method=method)
                return Eg, source
            except Exception as e:
                if method == 'aflow':
                    # If aflow method explicitly requested, raise error
                    raise RuntimeError(f"AFLOW query failed: {e}")
                # Otherwise, fallback to Vegard's law
                warnings.warn(f"AFLOW query failed: {e}. Using Vegard's law fallback.")
                method = 'vegard'

        # Vegard's law with bowing parameter (default)
        if method == 'vegard' or method is None:
            Eg = (1 - x_Se) * self.Eg_ZnS + x_Se * self.Eg_ZnSe
            Eg -= self.bowing * x_Se * (1 - x_Se)
            return Eg, 'vegard'

        else:
            raise ValueError(f"Unknown method: {method}. Use 'vegard', 'aflow', or 'aflow_with_fallback'.")

    def composition_string(self, x_Se: float) -> str:
        """
        Generate composition string for ZnS(1-x)Se(x).

        Parameters:
        -----------
        x_Se : float
            Selenium composition

        Returns:
        --------
        formula : str
            Chemical formula (e.g., "ZnS0.75Se0.25")
        """
        x_S = 1.0 - x_Se
        return f"ZnS{x_S:.2f}Se{x_Se:.2f}"

    def export_vegard_data(self, n_points: int = 100, output_path=None):
        """
        Export Vegard's law bandgap data to CSV.

        Parameters:
        -----------
        n_points : int
            Number of composition points to sample
        output_path : str or Path, optional
            Path to save CSV. If None, returns DataFrame only

        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns: x_Se, Eg_eV, Eg_TiO2
        """
        import pandas as pd
        from datetime import datetime
        from pathlib import Path

        # Generate composition range
        x_Se_range = np.linspace(0.0, 1.0, n_points)

        # Calculate bandgaps using existing compute_bandgap method
        bandgaps = []
        for x_Se in x_Se_range:
            Eg, _ = self.compute_bandgap(x_Se)
            bandgaps.append(Eg)

        # Create DataFrame
        df = pd.DataFrame({
            'x_Se': x_Se_range,
            'Eg_eV': bandgaps,
            'Eg_TiO2': self.Eg_TiO2  # constant reference
        })

        # Export to CSV if path provided
        if output_path is not None:
            csv_path = Path(output_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Write metadata header
            with open(csv_path, 'w') as f:
                f.write(f"# Vegard's Law Bandgap Data\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# \n")
                f.write(f"# Material: ZnS(1-x)Se(x)\n")
                f.write(f"# Eg_ZnS = {self.Eg_ZnS:.2f} eV\n")
                f.write(f"# Eg_ZnSe = {self.Eg_ZnSe:.2f} eV\n")
                f.write(f"# Bowing parameter b = {self.bowing:.2f} eV\n")
                f.write(f"# \n")
                f.write(f"# Equation: Eg(x) = (1-x)*Eg_ZnS + x*Eg_ZnSe - b*x*(1-x)\n")
                f.write(f"# \n")
                f.write(f"# Columns:\n")
                f.write(f"#   x_Se    - Selenium composition (0=ZnS, 1=ZnSe)\n")
                f.write(f"#   Eg_eV   - Bandgap energy (eV)\n")
                f.write(f"#   Eg_TiO2 - Reference TiO2 bandgap ({self.Eg_TiO2} eV)\n")
                f.write(f"# \n")

            # Append data
            df.to_csv(csv_path, mode='a', index=False, float_format='%.6f')
            print(f"✓ Vegard data saved: {csv_path}")

        return df

    def export_composition_space(self, n_points: int = 50, output_path=None,
                                 light_intensity: float = 1.0):
        """
        Export complete composition space (bandgap + voltage) to CSV.

        Parameters:
        -----------
        n_points : int
            Number of composition points to sample
        output_path : str or Path, optional
            Path to save CSV
        light_intensity : float
            Light intensity for voltage calculation (1.0 = full sun)

        Returns:
        --------
        pandas.DataFrame
            DataFrame with composition space data
        """
        import pandas as pd
        from datetime import datetime
        from pathlib import Path

        # Generate composition range
        x_Se_range = np.linspace(0.0, 1.0, n_points)

        # Calculate bandgaps and voltages
        data = []
        for x_Se in x_Se_range:
            measurement = self.measure_voltage(x_Se, light_intensity=light_intensity)
            data.append({
                'x_Se': x_Se,
                'Eg_eV': measurement['Eg_eV'],
                'Voc_V': measurement['Voc_V']
            })

        # Create DataFrame with derived columns
        df = pd.DataFrame(data)
        df['Voc_noise_lower_V'] = df['Voc_V'] - 0.01
        df['Voc_noise_upper_V'] = df['Voc_V'] + 0.01
        df['Eg_TiO2'] = self.Eg_TiO2
        df['optimal_region'] = df['x_Se'] < 0.1

        # Export to CSV if path provided
        if output_path is not None:
            csv_path = Path(output_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Write metadata header
            with open(csv_path, 'w') as f:
                f.write(f"# Composition Space Data (Bandgap + Voltage)\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Mode: {self.mode}\n")
                f.write(f"# Light intensity: {light_intensity} (1.0 = full sun)\n")
                f.write(f"# \n")
                f.write(f"# Columns:\n")
                f.write(f"#   x_Se               - Selenium composition\n")
                f.write(f"#   Eg_eV              - Bandgap energy (eV)\n")
                f.write(f"#   Voc_V              - Open-circuit voltage (V)\n")
                f.write(f"#   Voc_noise_lower_V  - Voltage - 10 mV (noise band)\n")
                f.write(f"#   Voc_noise_upper_V  - Voltage + 10 mV (noise band)\n")
                f.write(f"#   Eg_TiO2            - Reference TiO2 bandgap\n")
                f.write(f"#   optimal_region     - True if x_Se < 0.1\n")
                f.write(f"# \n")

            # Append data
            df.to_csv(csv_path, mode='a', index=False, float_format='%.6f')
            print(f"✓ Composition space data saved: {csv_path}")

        return df

    def measure_voltage(self, x_Se: float, light_intensity: float = 1.0) -> Dict:
        """
        Measure open-circuit voltage (Voc) for given composition.

        Parameters:
        -----------
        x_Se : float
            Selenium composition
        light_intensity : float
            Relative light intensity (0.0 to 1.0)

        Returns:
        --------
        measurement : dict
            {
                'x_Se': composition,
                'formula': chemical formula,
                'Eg_eV': bandgap,
                'Eg_source': data source ('vegard' or 'aflow'),
                'Voc_V': open-circuit voltage,
                'light_intensity': light level,
                'measurement_mode': 'hardware' or 'simulated',
                'egap_method': bandgap calculation method
            }
        """
        # Calculate bandgap (returns tuple: Eg, source)
        Eg, Eg_source = self.compute_bandgap(x_Se)

        if self.mode == 'hardware':
            # Read from MCP3008 ADC
            Voc = self._read_adc_voltage(channel=1)

        else:
            # Simulated voltage based on bandgap
            # Empirical correlation: Voc ≈ 0.3 + 0.15*Eg
            # This matches typical DSSC behavior where Voc increases with bandgap
            Voc_base = 0.3 + 0.15 * Eg

            # Add light intensity dependence: Voc ∝ ln(I)
            Voc = Voc_base + 0.025 * np.log(light_intensity + 0.01)

            # Add measurement noise (±10 mV typical)
            Voc += np.random.normal(0, 0.010)

            # Physical limits
            Voc = np.clip(Voc, 0.1, 1.0)

        return {
            'x_Se': x_Se,
            'formula': self.composition_string(x_Se),
            'Eg_eV': Eg,
            'Eg_source': Eg_source,
            'Voc_V': Voc,
            'light_intensity': light_intensity,
            'measurement_mode': self.mode,
            'egap_method': self.egap_method
        }

    def _read_adc_voltage(self, channel: int = 1) -> float:
        """
        Read voltage from MCP3008 ADC channel.

        Parameters:
        -----------
        channel : int
            ADC channel (0-7)

        Returns:
        --------
        voltage : float
            Measured voltage in V
        """
        # MCP3008 SPI protocol
        adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]

        # Convert to voltage (10-bit ADC, 3.3V reference)
        voltage = (data / 1023.0) * 3.3

        return voltage

    def compare_to_tio2(self, x_Se: float) -> Dict:
        """
        Compare ZnSSe composition to TiO2 benchmark.

        Parameters:
        -----------
        x_Se : float
            Selenium composition

        Returns:
        --------
        comparison : dict
            Bandgap comparison with TiO2
        """
        Eg_znsse, Eg_source = self.compute_bandgap(x_Se)
        delta_Eg = Eg_znsse - self.Eg_TiO2

        if Eg_znsse < self.Eg_TiO2:
            recommendation = "Lower bandgap → Better visible absorption than TiO2"
        elif Eg_znsse > self.Eg_TiO2:
            recommendation = "Higher bandgap → Less visible absorption than TiO2"
        else:
            recommendation = "Similar bandgap to TiO2"

        return {
            'composition': self.composition_string(x_Se),
            'Eg_ZnSSe': Eg_znsse,
            'Eg_source': Eg_source,
            'Eg_TiO2': self.Eg_TiO2,
            'delta_Eg': delta_Eg,
            'recommendation': recommendation
        }

    def optimal_composition_range(self) -> Dict:
        """
        Suggest optimal composition range for DSSC applications.

        Based on:
        - Visible light absorption (Eg ~ 2.5-3.0 eV optimal)
        - Conduction band alignment with dyes
        - Electron injection efficiency

        Returns:
        --------
        ranges : dict
            Suggested composition ranges for different applications
        """
        compositions = np.linspace(0, 1, 100)
        bandgaps = [self.compute_bandgap(x)[0] for x in compositions]  # Extract Eg from tuple

        # Find compositions in optimal bandgap range
        visible_optimal = [(x, Eg) for x, Eg in zip(compositions, bandgaps)
                          if 2.5 <= Eg <= 3.0]

        if visible_optimal:
            x_opt_range = (visible_optimal[0][0], visible_optimal[-1][0])
            Eg_opt_range = (visible_optimal[0][1], visible_optimal[-1][1])
        else:
            x_opt_range = (0.5, 0.8)
            Eg_opt_range = (2.7, 3.0)

        return {
            'visible_light_optimal': {
                'x_Se_range': x_opt_range,
                'Eg_range_eV': Eg_opt_range,
                'description': 'Best for visible light absorption'
            },
            'high_voltage': {
                'x_Se_range': (0.0, 0.3),
                'Eg_range_eV': (3.2, 3.68),
                'description': 'Maximum Voc (wide bandgap)'
            },
            'high_current': {
                'x_Se_range': (0.7, 1.0),
                'Eg_range_eV': (2.7, 2.95),
                'description': 'Maximum Jsc (narrow bandgap)'
            }
        }

    def close(self):
        """Clean up hardware connections."""
        if self.mode == 'hardware' and hasattr(self, 'spi'):
            self.spi.close()
            print("[ZnSSe Interface] Hardware connections closed")


def demo_interface():
    """Demonstrate ZnSSe interface capabilities with AFLOW integration."""
    print("\n" + "="*80)
    print("ZnS(1-x)Se(x) DSSC Interface Demo - AFLOW Integration Version")
    print("="*80)

    # Demo 1: Vegard's law mode (simulated measurement)
    print("\n" + "-"*80)
    print("MODE 1: Simulated Measurement + Vegard's Law Bandgap")
    print("-"*80)
    interface = ZnSSeInterface(mode='simulated', egap_method='vegard')

    # Test compositions
    print(f"\n{'x_Se':>6} | {'Formula':>15} | {'Eg (eV)':>10} | {'Source':>12} | {'vs TiO2':>10}")
    print("-"*80)

    test_compositions = [0.0, 0.25, 0.50, 0.75, 1.0]
    for x_Se in test_compositions:
        Eg, source = interface.compute_bandgap(x_Se)
        formula = interface.composition_string(x_Se)
        delta = Eg - interface.Eg_TiO2
        print(f"{x_Se:6.2f} | {formula:>15} | {Eg:10.2f} | {source:>12} | {delta:+10.2f}")

    # Voltage measurements
    print("\n" + "-"*80)
    print("Simulated Voltage Measurements")
    print("-"*80)
    print(f"{'Formula':>15} | {'Eg (eV)':>10} | {'Voc (V)':>10} | {'Eg Source':>12}")
    print("-"*80)

    for x_Se in test_compositions:
        result = interface.measure_voltage(x_Se)
        print(f"{result['formula']:>15} | {result['Eg_eV']:10.2f} | {result['Voc_V']:10.3f} | {result['Eg_source']:>12}")

    # Demo 2: AFLOW mode (if available)
    print("\n" + "-"*80)
    print("MODE 2: Simulated Measurement + AFLOW API Bandgap (with fallback)")
    print("-"*80)

    interface_aflow = ZnSSeInterface(mode='simulated', egap_method='aflow_with_fallback')

    print("\nNote: AFLOW database may not have ZnSSe data, will fallback to Vegard's law")
    print(f"\n{'x_Se':>6} | {'Formula':>15} | {'Eg (eV)':>10} | {'Source':>15}")
    print("-"*80)

    for x_Se in [0.0, 0.5, 1.0]:
        Eg, source = interface_aflow.compute_bandgap(x_Se)
        formula = interface_aflow.composition_string(x_Se)
        print(f"{x_Se:6.2f} | {formula:>15} | {Eg:10.2f} | {source:>15}")

    # Optimal ranges
    print("\n" + "-"*80)
    print("Optimal Composition Ranges")
    print("-"*80)

    ranges = interface.optimal_composition_range()
    for app_name, app_data in ranges.items():
        print(f"\n{app_name.upper().replace('_', ' ')}:")
        print(f"  x_Se range: {app_data['x_Se_range'][0]:.2f} - {app_data['x_Se_range'][1]:.2f}")
        print(f"  Eg range:   {app_data['Eg_range_eV'][0]:.2f} - {app_data['Eg_range_eV'][1]:.2f} eV")
        print(f"  {app_data['description']}")

    print("\n" + "="*80)
    print("HARDWARE MODE USAGE (Raspberry Pi + MCP3008):")
    print("="*80)
    print("interface = ZnSSeInterface(mode='hardware', egap_method='aflow_with_fallback')")
    print("result = interface.measure_voltage(x_Se=0.5)")
    print("  → Reads voltage from ADC channel 1")
    print("  → Uses AFLOW API for bandgap (or Vegard's law fallback)")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_interface()
