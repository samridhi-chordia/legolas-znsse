#!/usr/bin/env python3
"""
Gaussian Process Optimizer for ZnS(1-x)Se(x) DSSC
==================================================

Implements the optimization methodology from Chordia et al. (2025):
- Gaussian Process Regression with RBF + White kernel
- Expected Improvement acquisition function
- Bandgap-guided composition optimization
- Iterative closed-loop experimental design

Mathematical Framework:
-----------------------
Predicted voltage at bandgap E*_g:
    μ(E*_g) = k*^T [K + σ²_n I]^(-1) y

Optimal bandgap:
    E*_g = argmax_{Eg} μ(Eg)

Where:
- k* = kernel values between new and training bandgaps
- K = covariance matrix (RBF kernel)
- σ²_n = noise variance
- y = observed voltages

Authors: Samridhi Chordia
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.exceptions import ConvergenceWarning
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    ConvergenceWarning = Warning  # Fallback
    warnings.warn("scikit-learn not available. Limited functionality.")

from znsse_interface import ZnSSeInterface


@dataclass
class OptimizationResult:
    """
    Results from a single optimization iteration.

    Attributes:
    -----------
    iteration : int
        Iteration number
    x_Se : float
        Selenium composition tested
    Eg_eV : float
        Bandgap energy
    Voc_V : float
        Open-circuit voltage measured
    predicted_Voc : float
        GP prediction (if available)
    uncertainty : float
        GP uncertainty (standard deviation)
    acquisition_value : float
        Expected Improvement value
    """
    iteration: int
    x_Se: float
    Eg_eV: float
    Voc_V: float
    predicted_Voc: Optional[float] = None
    uncertainty: Optional[float] = None
    acquisition_value: Optional[float] = None


class GPOptimizer:
    """
    Gaussian Process optimizer for ZnS(1-x)Se(x) composition.

    Implements the LEGOLAS optimization workflow:
    1. Measure voltage for initial compositions
    2. Train GP model on (bandgap, voltage) pairs
    3. Predict optimal bandgap using Expected Improvement
    4. Map bandgap back to composition
    5. Iterate until convergence
    """

    def __init__(self, interface: ZnSSeInterface, xi: float = 0.01, random_state: int = 42, n_candidates: int = 100,
                 save_figures: bool = False, save_data: bool = False, output_dir: str = 'results'):
        """
        Initialize GP optimizer.

        Parameters:
        -----------
        interface : ZnSSeInterface
            Hardware/simulation interface for measurements
        xi : float
            Exploration parameter for Expected Improvement (default: 0.01)
        random_state : int
            Random seed for reproducibility
        n_candidates : int
            Number of candidate points for acquisition function optimization (default: 100)
        save_figures : bool
            Save figures to output directory (default: False)
        save_data : bool
            Export CSV data files (default: False)
        output_dir : str
            Output directory for figures and data (default: 'results')
        """
        from pathlib import Path

        self.interface = interface
        self.xi = xi
        self.random_state = random_state
        self.n_candidates = n_candidates
        self.save_figures = save_figures
        self.save_data = save_data
        self.output_dir = Path(output_dir)
        np.random.seed(random_state)

        # Optimization history
        self.X_bandgaps: List[float] = []  # Bandgaps tested
        self.y_voltages: List[float] = []  # Voltages measured
        self.x_compositions: List[float] = []  # Compositions tested
        self.results: List[OptimizationResult] = []

        # GP model
        self.gp_model = None

        # Pre-compute candidate grid (optimization: compute once, not every iteration)
        print(f"[GP Optimizer] Pre-computing bandgap grid for {n_candidates} candidates...")
        self.x_Se_candidates = np.linspace(0.0, 1.0, n_candidates)
        self.Eg_candidates = np.array([
            self.interface.compute_bandgap(x)[0] for x in self.x_Se_candidates
        ]).reshape(-1, 1)
        print(f"[GP Optimizer] Bandgap grid: {self.Eg_candidates.min():.2f} to {self.Eg_candidates.max():.2f} eV")

        print(f"[GP Optimizer] Initialized with xi={xi}")

    def _build_gp_model(self):
        """
        Build Gaussian Process model using RBF + White kernel.

        Kernel structure (from paper):
            k(Eg, Eg') = RBF(Eg, Eg') + WhiteKernel

        - RBF captures smooth trends
        - WhiteKernel models measurement noise
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available. Cannot build GP model.")
            return

        # Define kernel: RBF for smoothness + White for noise
        # Note: Relaxed noise_level lower bound to prevent convergence warnings
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e1)) +
            WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e0))
        )

        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=self.random_state,
            normalize_y=True
        )

        # Train on observed data
        X = np.array(self.X_bandgaps).reshape(-1, 1)
        y = np.array(self.y_voltages)

        # Suppress convergence warnings for noise_level bounds
        # (These are informational - low noise is expected for simulated data)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            self.gp_model.fit(X, y)

        print(f"[GP Optimizer] Model trained on {len(X)} points")
        print(f"[GP Optimizer] Kernel: {self.gp_model.kernel_}")

    def _expected_improvement(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.

        EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)

        Where:
        - Z = (μ(x) - f_best - ξ) / σ(x)
        - Φ = cumulative normal distribution
        - φ = probability density function
        - ξ = exploration parameter

        Parameters:
        -----------
        X_candidates : ndarray, shape (n, 1)
            Candidate bandgaps to evaluate

        Returns:
        --------
        ei_values : ndarray, shape (n,)
            Expected improvement for each candidate
        """
        if self.gp_model is None:
            # Random exploration if no model
            return np.random.rand(len(X_candidates))

        # GP predictions
        mu, sigma = self.gp_model.predict(X_candidates, return_std=True)

        # Best observed voltage so far
        f_best = np.max(self.y_voltages)

        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)

        # Calculate Z-score
        Z = (mu - f_best - self.xi) / sigma

        # Expected Improvement
        from scipy.stats import norm
        ei = (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _propose_next_composition(self) -> float:
        """
        Propose next composition to test using Expected Improvement.

        Strategy:
        1. Use pre-computed bandgap grid (self.Eg_candidates)
        2. Evaluate Expected Improvement for each
        3. Select bandgap with highest EI
        4. Map bandgap back to composition

        Returns:
        --------
        x_Se_next : float
            Next composition to test
        """
        # Evaluate Expected Improvement using pre-computed bandgaps
        ei_values = self._expected_improvement(self.Eg_candidates)

        # Select best candidate
        best_idx = np.argmax(ei_values)
        x_Se_next = self.x_Se_candidates[best_idx]
        ei_best = ei_values[best_idx]

        print(f"[GP Optimizer] Next composition: x_Se = {x_Se_next:.3f} (EI = {ei_best:.4f})")

        return x_Se_next

    def _measure_composition(self, x_Se: float, iteration: int) -> OptimizationResult:
        """
        Measure voltage for given composition.

        Parameters:
        -----------
        x_Se : float
            Selenium composition
        iteration : int
            Iteration number

        Returns:
        --------
        result : OptimizationResult
            Measurement result
        """
        # Measure voltage
        measurement = self.interface.measure_voltage(x_Se)

        # Get GP prediction if model exists
        predicted_Voc = None
        uncertainty = None
        if self.gp_model is not None:
            Eg_test = np.array([[measurement['Eg_eV']]])
            predicted_Voc, uncertainty = self.gp_model.predict(Eg_test, return_std=True)
            predicted_Voc = predicted_Voc[0]
            uncertainty = uncertainty[0]

        result = OptimizationResult(
            iteration=iteration,
            x_Se=x_Se,
            Eg_eV=measurement['Eg_eV'],
            Voc_V=measurement['Voc_V'],
            predicted_Voc=predicted_Voc,
            uncertainty=uncertainty,
            acquisition_value=None  # Will be set if relevant
        )

        # Store in history
        self.X_bandgaps.append(measurement['Eg_eV'])
        self.y_voltages.append(measurement['Voc_V'])
        self.x_compositions.append(x_Se)
        self.results.append(result)

        return result

    def optimize(self, n_iterations: int = 10, n_initial: int = 4) -> pd.DataFrame:
        """
        Run Bayesian optimization loop.

        Workflow (from Chordia et al. 2025):
        1. Initial random sampling (n_initial compositions)
        2. Train GP model on (Eg, Voc) pairs
        3. Use Expected Improvement to propose next composition
        4. Measure proposed composition
        5. Update GP model
        6. Repeat until convergence or max iterations

        Parameters:
        -----------
        n_iterations : int
            Total number of measurements (default: 10)
        n_initial : int
            Number of initial random samples (default: 4)

        Returns:
        --------
        results_df : pd.DataFrame
            Optimization history with all measurements
        """
        # Store n_initial for use in plotting methods
        self.n_initial = n_initial

        print("\n" + "="*60)
        print("LEGOLAS ZnS(1-x)Se(x) Optimization")
        print("="*60)
        print(f"Total iterations: {n_iterations}")
        print(f"Initial random samples: {n_initial}")
        print(f"Bayesian optimization: {n_iterations - n_initial}")
        print("="*60 + "\n")

        for i in range(n_iterations):
            print(f"\n--- Iteration {i+1}/{n_iterations} ---")

            # Phase 1: Initial random exploration
            if i < n_initial:
                x_Se = np.random.uniform(0.0, 1.0)
                print(f"Random sampling: x_Se = {x_Se:.3f}")

            # Phase 2: GP-guided optimization
            else:
                if i == n_initial:
                    # Build initial GP model
                    print("\nBuilding initial GP model...")
                    self._build_gp_model()

                # Propose next composition
                x_Se = self._propose_next_composition()

            # Measure composition
            formula = self.interface.composition_string(x_Se)
            print(f"Testing: {formula}")

            result = self._measure_composition(x_Se, i)

            print(f"Bandgap: {result.Eg_eV:.2f} eV")
            print(f"Voltage: {result.Voc_V:.3f} V")

            if result.predicted_Voc is not None:
                error = abs(result.Voc_V - result.predicted_Voc)
                print(f"GP Prediction: {result.predicted_Voc:.3f} ± {result.uncertainty:.3f} V")
                print(f"Prediction error: {error:.3f} V")

            # Update GP model after each measurement (re-train)
            if i >= n_initial:
                self._build_gp_model()

        # Final summary
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)

        best_idx = np.argmax(self.y_voltages)
        best_result = self.results[best_idx]

        print(f"\nBest result:")
        print(f"  Composition: {self.interface.composition_string(best_result.x_Se)}")
        print(f"  x_Se: {best_result.x_Se:.3f}")
        print(f"  Bandgap: {best_result.Eg_eV:.2f} eV")
        print(f"  Voltage: {best_result.Voc_V:.3f} V")
        print(f"  Found at iteration: {best_result.iteration + 1}")

        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'iteration': r.iteration,
                'x_Se': r.x_Se,
                'formula': self.interface.composition_string(r.x_Se),
                'Eg_eV': r.Eg_eV,
                'Voc_V': r.Voc_V,
                'predicted_Voc_V': r.predicted_Voc,
                'uncertainty_V': r.uncertainty
            }
            for r in self.results
        ])

        return results_df

    def plot_results(self, save_path: Optional[str] = None, export_csv: Optional[str] = None):
        """
        Visualize optimization results.

        Creates single plot: Voltage vs Bandgap with GP fit

        Parameters:
        -----------
        save_path : str, optional
            Path to save figure
        export_csv : str, optional
            Path to save CSV data. If None, use self.save_data flag
            and derive from save_path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[GP Optimizer] matplotlib not available for plotting")
            return

        # Create single plot (only Panel A - GP Model)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        X_plot = np.linspace(2.6, 3.8, 100).reshape(-1, 1)

        if self.gp_model is not None:
            mu, sigma = self.gp_model.predict(X_plot, return_std=True)

            ax.plot(X_plot, mu, 'b-', label='GP mean', linewidth=2)
            ax.fill_between(
                X_plot.ravel(),
                mu - 1.96*sigma,
                mu + 1.96*sigma,
                alpha=0.2,
                label='95% confidence'
            )

        ax.scatter(self.X_bandgaps, self.y_voltages, c='red', s=100,
                   marker='o', edgecolors='black', label='Measurements', zorder=5)

        # Mark TiO2 reference
        ax.axvline(self.interface.Eg_TiO2, color='green', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='$\mathrm{TiO}_2$')

        ax.set_xlabel('Bandgap (eV)', fontsize=11)
        ax.set_ylabel('Open-Circuit Voltage (V)', fontsize=11)
        ax.set_title('Gaussian Process Regression Model', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[GP Optimizer] Figure saved to: {save_path}")

        # Export CSV data if requested
        if self.save_data or export_csv is not None:
            from pathlib import Path
            from datetime import datetime

            # Determine CSV path
            if export_csv is not None:
                csv_path = Path(export_csv)
            elif save_path is not None:
                csv_path = Path(save_path).parent.parent / 'data' / 'optimization_trajectory.csv'
            else:
                csv_path = self.output_dir / 'data' / 'optimization_trajectory.csv'

            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine n_initial from results (first phase is random)
            n_initial = len([r for r in self.results if r.acquisition_value is None])

            # Export optimization trajectory
            df_export = pd.DataFrame({
                'iteration': range(len(self.X_bandgaps)),
                'x_Se': self.x_compositions,
                'Eg_eV': self.X_bandgaps,
                'Voc_V': self.y_voltages,
                'phase': ['random' if i < n_initial else 'gp_guided'
                          for i in range(len(self.X_bandgaps))]
            })
            df_export['is_best'] = df_export['Voc_V'] == df_export['Voc_V'].max()

            with open(csv_path, 'w') as f:
                f.write(f"# Bayesian Optimization Results\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Method: Gaussian Process with Expected Improvement\n")
                if self.gp_model is not None:
                    f.write(f"# Kernel: {self.gp_model.kernel_}\n")
                f.write(f"# \n")
                f.write(f"# Columns:\n")
                f.write(f"#   iteration - Measurement order (0-{len(self.X_bandgaps)-1})\n")
                f.write(f"#   x_Se      - Selenium composition\n")
                f.write(f"#   Eg_eV     - Bandgap energy (eV)\n")
                f.write(f"#   Voc_V     - Measured open-circuit voltage (V)\n")
                f.write(f"#   phase     - random (0-{n_initial-1}) or gp_guided ({n_initial}+)\n")
                f.write(f"#   is_best   - True for optimal composition found\n")
                f.write(f"# \n")

            df_export.to_csv(csv_path, mode='a', index=False, float_format='%.6f')
            print(f"✓ Data saved: {csv_path}")

            # Also export GP predictions if model is trained
            if self.gp_model is not None:
                csv_pred = csv_path.parent / 'gp_regression_predictions.csv'
                X_plot_export = np.linspace(2.6, 3.8, 100).reshape(-1, 1)
                mu, sigma = self.gp_model.predict(X_plot_export, return_std=True)

                df_pred = pd.DataFrame({
                    'Eg_eV': X_plot_export.ravel(),
                    'Voc_mean_V': mu,
                    'Voc_std_V': sigma,
                    'Voc_ci_lower_V': mu - 1.96*sigma,
                    'Voc_ci_upper_V': mu + 1.96*sigma
                })

                with open(csv_pred, 'w') as f:
                    f.write(f"# Gaussian Process Model Predictions\n")
                    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Training points: {len(self.X_bandgaps)}\n")
                    f.write(f"# \n")
                    f.write(f"# Columns:\n")
                    f.write(f"#   Eg_eV          - Bandgap energy (eV)\n")
                    f.write(f"#   Voc_mean_V     - GP predicted mean voltage (V)\n")
                    f.write(f"#   Voc_std_V      - Prediction uncertainty (std dev)\n")
                    f.write(f"#   Voc_ci_lower_V - 95% CI lower bound (V)\n")
                    f.write(f"#   Voc_ci_upper_V - 95% CI upper bound (V)\n")
                    f.write(f"# \n")

                df_pred.to_csv(csv_pred, mode='a', index=False, float_format='%.6f')
                print(f"✓ Data saved: {csv_pred}")

        return fig

    def plot_optimization_convergence(self, save_path: Optional[str] = None):
        """
        Visualize optimization convergence trajectory (2-panel figure).

        Panel A: Voltage vs. iteration (orange = random, blue = GP-guided)
        Panel B: Composition space exploration (numbered points 1-10)

        Parameters:
        -----------
        save_path : str, optional
            Path to save figure (PDF/PNG)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[GP Optimizer] matplotlib not available for plotting")
            return

        # Create 2-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Panel A: Voltage vs. Iteration
        ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='top')

        iterations = np.arange(len(self.y_voltages))

        # Split into random exploration and GP-guided phases
        random_indices = iterations < self.n_initial
        gp_indices = iterations >= self.n_initial

        # Plot random exploration (orange)
        ax1.scatter(iterations[random_indices],
                   np.array(self.y_voltages)[random_indices],
                   c='orange', s=100, marker='o', edgecolors='black',
                   label='Random exploration', zorder=5)

        # Plot GP-guided optimization (blue)
        ax1.scatter(iterations[gp_indices],
                   np.array(self.y_voltages)[gp_indices],
                   c='blue', s=100, marker='o', edgecolors='black',
                   label='GP-guided', zorder=5)

        # Connect points with line
        ax1.plot(iterations, self.y_voltages, 'k--', alpha=0.3, linewidth=1)

        # Mark best point
        best_idx = np.argmax(self.y_voltages)
        ax1.scatter(best_idx, self.y_voltages[best_idx],
                   c='green', s=300, marker='*', edgecolors='black',
                   label='Best', zorder=10)

        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Open-Circuit Voltage (V)', fontsize=11)
        ax1.set_title('Optimization Convergence', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, len(iterations)-0.5)

        # Panel B: Composition Space Exploration
        ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
                fontsize=16, fontweight='bold', va='top')

        # Plot all points with same color, numbered labels
        # Use different colors for random vs GP-guided phases
        for i, (x, Eg) in enumerate(zip(self.x_compositions, self.X_bandgaps)):
            if i < self.n_initial:
                # Random exploration - orange
                color = 'orange'
            else:
                # GP-guided - blue
                color = 'blue'

            # Plot point
            ax2.scatter(x, Eg, c=color, s=100, marker='o',
                       edgecolors='black', linewidth=1.5, zorder=5)

            # Add iteration number label (1-indexed for readability)
            ax2.text(x, Eg, f'{i+1}', fontsize=8, ha='center', va='center',
                    color='white', weight='bold', zorder=6)

        # Mark best point with star
        ax2.scatter(self.x_compositions[best_idx], self.X_bandgaps[best_idx],
                   c='red', s=300, marker='*', edgecolors='black',
                   label='Optimal', zorder=10)

        # Add TiO2 reference line
        ax2.axhline(self.interface.Eg_TiO2, color='green', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='$\mathrm{TiO}_2$')

        ax2.set_xlabel('Se Composition ($x_{Se}$)', fontsize=11)
        ax2.set_ylabel('Bandgap (eV)', fontsize=11)
        ax2.set_title('Composition Space Exploration', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[GP Optimizer] Optimization convergence figure saved to: {save_path}")

        return fig


if __name__ == "__main__":
    print("GP Optimizer module loaded successfully")
    print("Import this module and use GPOptimizer class for optimization")
