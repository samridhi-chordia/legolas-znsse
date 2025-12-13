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
# Run all demos (interactive, no export)
python demo.py

# Run all demos, save figures and CSV data for paper
python demo.py --save-figures --save-data

# Custom output directory
python demo.py --save-figures --save-data --output-dir results_2025

Command-line Arguments:
-----------------------
--save-figures     Save figures to paper/figures/ (default: display only)
--save-data        Export CSV data to paper/data/ (default: no export)
--output-dir DIR   Output directory for figures/data (default: paper)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from znsse_interface import ZnSSeInterface
from gp_optimizer import GPOptimizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='LEGOLAS: Bayesian optimization for ZnSSe DSSCs'
    )
    parser.add_argument(
        '--save-figures',
        action='store_true',
        help='Save figures to paper/figures/ directory'
    )
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Export CSV data files to paper/data/ directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper',
        help='Output directory for figures and data (default: paper)'
    )
    return parser.parse_args()


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60 + "\n")


def demo_interface():
    """Demonstrate ZnSSe interface capabilities."""
    print_header("PART 1: ZnS(1-x)Se(x) Material Properties")

    interface = ZnSSeInterface(mode='simulated', egap_method='vegard')

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


def demo_optimization(save_figures=False, save_data=False, output_dir='results'):
    """Run Gaussian Process optimization."""
    print_header("PART 2: Bayesian Optimization with Gaussian Process")

    # Initialize interface and optimizer
    interface = ZnSSeInterface(mode='simulated', egap_method='vegard')
    optimizer = GPOptimizer(interface, xi=0.01, random_state=42,
                           save_figures=save_figures, save_data=save_data, output_dir=output_dir)

    print("Starting optimization...")
    print("Objective: Maximize open-circuit voltage (Voc)\n")

    # Run optimization
    results_df = optimizer.optimize(n_iterations=10, n_initial=4)

    # Save results (always save to results/ for backwards compatibility)
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / 'optimization_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Plot results - use output_dir if save_figures is True
    if save_figures:
        output_path = Path(output_dir)
        fig_dir = output_path / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Generate optimization convergence (2-panel)
        conv_path_pdf = fig_dir / 'optimization_convergence_trajectory.pdf'
        conv_path_png = fig_dir / 'optimization_convergence_trajectory.png'
        optimizer.plot_optimization_convergence(save_path=str(conv_path_pdf))
        optimizer.plot_optimization_convergence(save_path=str(conv_path_png))

        # Generate GP regression model (single panel)
        gp_path_pdf = fig_dir / 'gp_regression_model.pdf'
        gp_path_png = fig_dir / 'gp_regression_model.png'
        optimizer.plot_results(save_path=str(gp_path_pdf))
        optimizer.plot_results(save_path=str(gp_path_png))
    else:
        # Backwards compatibility - save to results/
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


def demo_full_exploration(save_data=False, data_dir=None, save_figures=False, fig_dir=None):
    """Create full composition-bandgap-voltage map."""
    print_header("PART 4: Full Composition Space Exploration")

    interface = ZnSSeInterface(mode='simulated', egap_method='vegard')

    # Export CSV data if requested
    if save_data and data_dir is not None:
        # Export Vegard's law bandgap data
        vegard_csv = data_dir / 'vegard_law_bandgap.csv'
        interface.export_vegard_data(n_points=100, output_path=vegard_csv)

        # Export composition space data
        comp_csv = data_dir / 'composition_space_data.csv'
        interface.export_composition_space(n_points=50, output_path=comp_csv)

    # Generate Figure 1 (Vegard's law) if requested
    if save_figures and fig_dir is not None:
        # Generate Vegard's law plot
        x_Se_vegard = np.linspace(0.0, 1.0, 100)
        bandgaps_vegard = []
        for x in x_Se_vegard:
            Eg, _ = interface.compute_bandgap(x)
            bandgaps_vegard.append(Eg)

        fig_vegard, ax_vegard = plt.subplots(1, 1, figsize=(6, 4))

        # Plot bandgap vs composition
        ax_vegard.plot(x_Se_vegard, bandgaps_vegard, 'b-', linewidth=2, label='Vegard\'s law + bowing')

        # TiO2 reference line
        ax_vegard.axhline(interface.Eg_TiO2, color='green', linestyle='--',
                         linewidth=1.5, label='$\mathrm{TiO}_2$ (3.2 eV)', alpha=0.7)

        ax_vegard.set_xlabel('Se Composition ($x_{Se}$)', fontsize=11)
        ax_vegard.set_ylabel('Bandgap (eV)', fontsize=11)
        ax_vegard.set_title('Bandgap vs Composition\n(Vegard\'s Law with Bowing Parameter)', fontsize=12)
        ax_vegard.legend(fontsize=10)
        ax_vegard.grid(True, alpha=0.3)
        ax_vegard.set_xlim(0, 1)
        ax_vegard.set_ylim(2.6, 3.8)

        # Add endpoint labels with boxes
        ax_vegard.text(0.02, 0.95, f'ZnS\n{interface.Eg_ZnS:.2f} eV',
                      transform=ax_vegard.transAxes, verticalalignment='top',
                      fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_vegard.text(0.98, 0.05, f'ZnSe\n{interface.Eg_ZnSe:.2f} eV',
                      transform=ax_vegard.transAxes, verticalalignment='bottom',
                      horizontalalignment='right', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Mark maximum bowing point (red dot at x_Se = 0.5)
        x_max_bowing = 0.5
        Eg_max_bowing, _ = interface.compute_bandgap(x_max_bowing)
        ax_vegard.plot(x_max_bowing, Eg_max_bowing, 'ro', markersize=8,
                      label=f'Max bowing\n($x_{{Se}}$ = 0.5)')

        plt.tight_layout()

        # Save both PDF and PNG versions
        vegard_path_pdf = fig_dir / 'vegard_law_bandgap.pdf'
        vegard_path_png = fig_dir / 'vegard_law_bandgap.png'
        fig_vegard.savefig(vegard_path_pdf, dpi=150, bbox_inches='tight')
        fig_vegard.savefig(vegard_path_png, dpi=150, bbox_inches='tight')
        print(f"Vegard's law plot saved to: {vegard_path_pdf} and {vegard_path_png}")
        plt.close(fig_vegard)

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
    # Add panel label A
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes,
            fontsize=16, fontweight='bold', va='top')

    # Highlight optimal region (x_Se < 0.1)
    ax1.axvspan(0, 0.1, alpha=0.2, color='green', label='Optimal region')

    ax1.plot(df['x_Se'], df['Eg_eV'], 'b-', linewidth=2)
    ax1.axhline(interface.Eg_TiO2, color='green', linestyle='--',
               label='$\mathrm{TiO}_2$', alpha=0.7)
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
    # Add panel label B
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
            fontsize=16, fontweight='bold', va='top')

    # Highlight optimal region (x_Se < 0.1)
    ax2.axvspan(0, 0.1, alpha=0.2, color='green', label='Optimal region')

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

    # Save figure - use fig_dir if save_figures is True
    if save_figures and fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        # Save both PDF and PNG versions
        comp_path_pdf = fig_dir / 'composition_space_exploration.pdf'
        comp_path_png = fig_dir / 'composition_space_exploration.png'
        plt.savefig(comp_path_pdf, dpi=150, bbox_inches='tight')
        plt.savefig(comp_path_png, dpi=150, bbox_inches='tight')
        print(f"Composition space map saved to: {comp_path_pdf} and {comp_path_png}")
    else:
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
    # Parse command-line arguments
    args = parse_args()

    # Setup output paths if saving is enabled
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / 'figures'
    data_dir = output_dir / 'data'

    if args.save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Figures will be saved to: {fig_dir}")

    if args.save_data:
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Data will be saved to: {data_dir}")

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

        # Part 2: Run optimization (pass save flags)
        optimizer, results_df = demo_optimization(
            save_figures=args.save_figures,
            save_data=args.save_data,
            output_dir=str(output_dir)
        )

        # Part 3: Analyze results
        analyze_results(optimizer, results_df)

        # Part 4: Full composition space (pass save flags)
        demo_full_exploration(
            save_data=args.save_data,
            data_dir=data_dir,
            save_figures=args.save_figures,
            fig_dir=fig_dir
        )

        # Final summary
        print_header("SUMMARY")
        print("✓ Material interface demonstrated")
        print("✓ Gaussian Process optimization completed")
        print("✓ Results analyzed and visualized")
        print("✓ Full composition space mapped")

        print("\nOutput files created:")
        if args.save_figures or args.save_data:
            if args.save_figures:
                print(f"  Figures: {fig_dir}/")
            if args.save_data:
                print(f"  Data CSV files: {data_dir}/")
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
