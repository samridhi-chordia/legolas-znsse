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

    def __init__(self, interface: ZnSSeInterface, xi: float = 0.01, random_state: int = 42):
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
        """
        self.interface = interface
        self.xi = xi
        self.random_state = random_state
        np.random.seed(random_state)

        # Optimization history
        self.X_bandgaps: List[float] = []  # Bandgaps tested
        self.y_voltages: List[float] = []  # Voltages measured
        self.x_compositions: List[float] = []  # Compositions tested
        self.results: List[OptimizationResult] = []

        # GP model
        self.gp_model = None

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

    def _propose_next_composition(self, n_candidates: int = 100) -> float:
        """
        Propose next composition to test using Expected Improvement.

        Strategy:
        1. Generate candidate bandgaps uniformly in range
        2. Evaluate Expected Improvement for each
        3. Select bandgap with highest EI
        4. Map bandgap back to composition

        Parameters:
        -----------
        n_candidates : int
            Number of candidate points to evaluate

        Returns:
        --------
        x_Se_next : float
            Next composition to test
        """
        # Generate candidate compositions
        x_Se_candidates = np.linspace(0.0, 1.0, n_candidates)

        # Convert to bandgaps (extract Eg from tuple [Eg, source])
        Eg_candidates = np.array([
            self.interface.compute_bandgap(x)[0] for x in x_Se_candidates
        ]).reshape(-1, 1)

        # Evaluate Expected Improvement
        ei_values = self._expected_improvement(Eg_candidates)

        # Select best candidate
        best_idx = np.argmax(ei_values)
        x_Se_next = x_Se_candidates[best_idx]
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

    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualize optimization results.

        Creates two plots:
        1. Voltage vs Bandgap with GP fit
        2. Voltage vs Composition with tested points

        Parameters:
        -----------
        save_path : str, optional
            Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[GP Optimizer] matplotlib not available for plotting")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Voltage vs Bandgap with GP fit
        X_plot = np.linspace(2.6, 3.8, 100).reshape(-1, 1)

        if self.gp_model is not None:
            mu, sigma = self.gp_model.predict(X_plot, return_std=True)

            ax1.plot(X_plot, mu, 'b-', label='GP mean', linewidth=2)
            ax1.fill_between(
                X_plot.ravel(),
                mu - 1.96*sigma,
                mu + 1.96*sigma,
                alpha=0.2,
                label='95% confidence'
            )

        ax1.scatter(self.X_bandgaps, self.y_voltages, c='red', s=100,
                   marker='o', edgecolors='black', label='Measurements', zorder=5)
        ax1.set_xlabel('Bandgap (eV)')
        ax1.set_ylabel('Open-Circuit Voltage (V)')
        ax1.set_title('GP Model: Voltage vs Bandgap')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mark TiO2 reference
        ax1.axvline(self.interface.Eg_TiO2, color='green', linestyle='--',
                   alpha=0.5, label='TiO2')

        # Plot 2: Voltage vs Composition
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.x_compositions)))

        ax2.scatter(self.x_compositions, self.y_voltages, c=colors, s=100,
                   marker='o', edgecolors='black', zorder=5)

        # Add iteration numbers
        for i, (x, y) in enumerate(zip(self.x_compositions, self.y_voltages)):
            ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        ax2.set_xlabel('Se Composition (x_Se)')
        ax2.set_ylabel('Open-Circuit Voltage (V)')
        ax2.set_title('Voltage vs Composition')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)

        # Add labels for endpoints
        ax2.text(0.02, 0.95, 'ZnS', transform=ax2.transAxes,
                verticalalignment='top', fontsize=9, style='italic')
        ax2.text(0.98, 0.95, 'ZnSe', transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[GP Optimizer] Figure saved to: {save_path}")

        return fig


if __name__ == "__main__":
    print("GP Optimizer module loaded successfully")
    print("Import this module and use GPOptimizer class for optimization")
