#!/usr/bin/env python3
"""
LEGOLAS ZnS(1-x)Se(x) DSSC Optimization Demo
=============================================

Machine-Learning Based Optimization of ZnS1-xSex Composition in Dye-Sensitized Solar Cells Using LEGOLAS Framework

Based on methodology from:
    Chordia, S., Lee, J., & Oses, C. (2025).
    Machine-Learning Based Optimization of ZnS1-xSex Composition in Dye-Sensitized Solar Cells Using LEGOLAS Framework
    Johns Hopkins University

Workflow:
---------
1. Initialize ZnSSe interface
2. Run Gaussian Process optimization
3. Visualize results
4. Compare with TiO2 benchmark

Usage:
------
python demo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from znsse_interface import ZnSSeInterface
from gp_optimizer import GPOptimizer


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60 + "\n")


def demo_interface():
    """Demonstrate ZnSSe interface capabilities."""
    print_header("PART 1: ZnS(1-x)Se(x) Material Properties")

    interface = ZnSSeInterface(mode='hardware_with_fallback', egap_method='aflow_with_fallback')

    # Test compositions matching paper (x = 0.0, 0.25, 0.75, 1.0)
    test_compositions = [0.0, 0.25, 0.50, 0.75, 1.0]

    print("Bandgap vs Composition (Vegard's Law with Bowing):")
    print("-"*60)
    print(f"{'x_Se':>6} | {'Formula':>15} | {'Bandgap (eV)':>15} | {'vs TiO2 (eV)':>15}")
    print("-"*60)

    for x_Se in test_compositions:
        Eg, source = interface.compute_bandgap(x_Se)
        formula = interface.composition_string(x_Se)
        delta = Eg - interface.Eg_TiO2
        print(f"{x_Se:6.2f} | {formula:>15} | {Eg:15.2f} | {delta:+15.2f}")

    print("\nKey observations:")
    print("  • Pure ZnS (3.68 eV) has wider bandgap than TiO2 (3.2 eV)")
    print("  • Pure ZnSe (2.70 eV) has narrower bandgap → better visible absorption")
    print("  • Full composition range available (0.0 to 1.0)")
    print("  • Bowing parameter: 0.50 eV")


def demo_optimization():
    """Run Gaussian Process optimization."""
    print_header("PART 2: Bayesian Optimization with Gaussian Process")

    # Initialize interface and optimizer
    interface = ZnSSeInterface(mode='hardware_with_fallback', egap_method='aflow_with_fallback')
    optimizer = GPOptimizer(interface, xi=0.01, random_state=42)

    print("Starting optimization...")
    print("Objective: Maximize open-circuit voltage (Voc)\n")

    # Run optimization
    results_df = optimizer.optimize(n_iterations=10, n_initial=4)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / 'optimization_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Plot results
    fig_path = results_dir / 'optimization_plot.png'
    optimizer.plot_results(save_path=str(fig_path))

    return optimizer, results_df


def analyze_results(optimizer: GPOptimizer, results_df: pd.DataFrame):
    """Analyze and summarize optimization results."""
    print_header("PART 3: Results Analysis")

    # Find best result
    best_idx = results_df['Voc_V'].idxmax()
    best = results_df.loc[best_idx]

    print("Best Composition Found:")
    print("-"*60)
    print(f"  Iteration:      {int(best['iteration']) + 1}")
    print(f"  Composition:    {best['formula']}")
    print(f"  x_Se:           {best['x_Se']:.3f}")
    print(f"  Bandgap:        {best['Eg_eV']:.2f} eV")
    print(f"  Voltage:        {best['Voc_V']:.3f} V")

    # Performance improvement
    initial_Voc = results_df.loc[0, 'Voc_V']
    final_Voc = best['Voc_V']
    improvement = ((final_Voc - initial_Voc) / initial_Voc) * 100

    print(f"\nOptimization Performance:")
    print("-"*60)
    print(f"  Initial Voc:    {initial_Voc:.3f} V")
    print(f"  Best Voc:       {final_Voc:.3f} V")
    print(f"  Improvement:    {improvement:+.1f}%")
    print(f"  Iterations:     {len(results_df)}")

    # Compare to TiO2
    comparison = optimizer.interface.compare_to_tio2(best['x_Se'])
    print(f"\nComparison with TiO2 Benchmark:")
    print("-"*60)
    print(f"  TiO2 bandgap:   {comparison['Eg_TiO2']:.2f} eV")
    print(f"  Optimal ZnSSe:  {comparison['Eg_ZnSSe']:.2f} eV")
    print(f"  Difference:     {comparison['delta_Eg']:+.2f} eV")
    print(f"  {comparison['recommendation']}")

    # Composition analysis
    print(f"\nComposition Analysis:")
    print("-"*60)
    if best['x_Se'] < 0.3:
        region = "ZnS-rich (wide bandgap, high Voc)"
    elif best['x_Se'] > 0.7:
        region = "ZnSe-rich (narrow bandgap, high Jsc potential)"
    else:
        region = "Intermediate composition (balanced properties)"
    print(f"  Region: {region}")

    # GP model performance
    if optimizer.gp_model is not None:
        print(f"\nGaussian Process Model:")
        print("-"*60)
        print(f"  Kernel: {optimizer.gp_model.kernel_}")
        print(f"  Training points: {len(optimizer.X_bandgaps)}")

        # Calculate prediction accuracy
        predictions = results_df['predicted_Voc_V'].dropna()
        actuals = results_df.loc[predictions.index, 'Voc_V']
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))

        print(f"  MAE: {mae:.4f} V")
        print(f"  RMSE: {rmse:.4f} V")


def demo_full_exploration():
    """Create full composition-bandgap-voltage map."""
    print_header("PART 4: Full Composition Space Exploration")

    interface = ZnSSeInterface(mode='hardware_with_fallback', egap_method='aflow_with_fallback')

    # Sample entire composition range
    n_points = 50
    x_Se_range = np.linspace(0.0, 1.0, n_points)

    # Calculate bandgaps and voltages
    data = []
    for x_Se in x_Se_range:
        measurement = interface.measure_voltage(x_Se)
        data.append({
            'x_Se': x_Se,
            'Eg_eV': measurement['Eg_eV'],
            'Voc_V': measurement['Voc_V']
        })

    df = pd.DataFrame(data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Bandgap vs Composition
    ax1.plot(df['x_Se'], df['Eg_eV'], 'b-', linewidth=2)
    ax1.axhline(interface.Eg_TiO2, color='green', linestyle='--',
               label='TiO2', alpha=0.7)
    ax1.set_xlabel('Se Composition (x_Se)')
    ax1.set_ylabel('Bandgap (eV)')
    ax1.set_title('Bandgap vs Composition\n(Vegard\'s Law + Bowing)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Add endpoint labels
    ax1.text(0.02, 0.95, f'ZnS\n{interface.Eg_ZnS:.2f} eV',
            transform=ax1.transAxes, verticalalignment='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(0.98, 0.05, f'ZnSe\n{interface.Eg_ZnSe:.2f} eV',
            transform=ax1.transAxes, verticalalignment='bottom',
            horizontalalignment='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Voltage vs Composition
    ax2.plot(df['x_Se'], df['Voc_V'], 'r-', linewidth=2)
    ax2.fill_between(df['x_Se'], df['Voc_V'] - 0.01, df['Voc_V'] + 0.01,
                     alpha=0.2, label='±10 mV noise')
    ax2.set_xlabel('Se Composition (x_Se)')
    ax2.set_ylabel('Open-Circuit Voltage (V)')
    ax2.set_title('Voltage vs Composition\n(Simulated Measurements)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Find and mark optimal region
    optimal_idx = df['Voc_V'].idxmax()
    optimal_x = df.loc[optimal_idx, 'x_Se']
    optimal_V = df.loc[optimal_idx, 'Voc_V']
    ax2.plot(optimal_x, optimal_V, 'g*', markersize=15, label='Maximum Voc')
    ax2.legend()

    plt.tight_layout()

    # Save figure
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    fig_path = results_dir / 'composition_space_map.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Composition space map saved to: {fig_path}")

    print(f"\nOptimal composition from full scan:")
    print(f"  x_Se: {optimal_x:.3f}")
    print(f"  Voc: {optimal_V:.3f} V")
    print(f"  Formula: {interface.composition_string(optimal_x)}")


def main():
    """Run complete LEGOLAS ZnSSe demonstration."""
    print("\n" + "="*60)
    print("LEGOLAS: ZnS(1-x)Se(x) DSSC Optimization")
    print("="*60)
    print("\nLow-cost Educational Guided Optimization Lab")
    print("for Autonomous Science")
    print("\nBased on: Chordia, Lee, & Oses (2025)")
    print("Johns Hopkins University")
    print("="*60)

    try:
        # Part 1: Demonstrate material properties
        demo_interface()

        # Part 2: Run optimization
        optimizer, results_df = demo_optimization()

        # Part 3: Analyze results
        analyze_results(optimizer, results_df)

        # Part 4: Full composition space
        demo_full_exploration()

        # Final summary
        print_header("SUMMARY")
        print("✓ Material interface demonstrated")
        print("✓ Gaussian Process optimization completed")
        print("✓ Results analyzed and visualized")
        print("✓ Full composition space mapped")

        print("\nOutput files created:")
        print("  • results/optimization_results.csv")
        print("  • results/optimization_plot.png")
        print("  • results/composition_space_map.png")

        print("\nKey findings:")
        print("  • GP-based optimization efficiently finds optimal composition")
        print("  • Minimal experimental trials needed (< 10 measurements)")
        print("  • Framework is modular and extensible")
        print("  • Compatible with hardware integration (MCP3008 + RPi)")

        print("\nNext steps:")
        print("  • Validate with experimental DSSC measurements")
        print("  • Test with different dyes (blackberry, ruthenium, etc.)")
        print("  • Optimize for different light spectra (AM1.5G, AM0)")
        print("  • Extend to multi-objective optimization (Voc + Jsc)")

        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
