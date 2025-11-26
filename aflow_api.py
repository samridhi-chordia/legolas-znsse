#!/usr/bin/env python3
"""
AFLOW API Integration for ZnS(1-x)Se(x) Bandgap Queries
========================================================

Interfaces with AFLOW database for DFT-calculated bandgaps using POCC method.

AFLOW (Automatic FLOW for Materials Discovery):
- Duke University materials database
- Over 3 million compounds
- DFT calculations via VASP
- POCC method for disordered alloys

API Documentation: http://aflowlib.org/API/

Based on:
    Rose et al. (2017). "AFLUX: The LUX materials search API"
    Computational Materials Science, 137, 362-370.

    Yang et al. (2016). "Modeling off-stoichiometry materials"
    Chemistry of Materials, 28(18), 6484-6492.
"""

import requests
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass


@dataclass
class AFLOWEntry:
    """Single entry from AFLOW database."""
    aurl: str  # AFLOW URL
    compound: str
    composition: str
    Egap: float  # eV
    Egap_type: str  # "direct" or "indirect"
    energy_cell: float  # eV/cell
    species: List[str]


class AFLOWInterface:
    """
    Interface to AFLOW database for ZnS(1-x)Se(x) bandgap queries.

    Uses AFLUX API to query DFT-calculated bandgaps for Zn-S-Se compounds.
    Falls back to Vegard's law if no data available.
    """

    def __init__(self, timeout: int = 30, cache: bool = True):
        """
        Initialize AFLOW interface.

        Parameters:
        -----------
        timeout : int
            API request timeout in seconds
        cache : bool
            Whether to cache query results
        """
        self.base_url = "http://aflow.org/API/aflux/"
        self.timeout = timeout
        self.cache_enabled = cache
        self.cache: Dict[float, float] = {}

        # Vegard's law parameters (fallback)
        self.Eg_ZnS = 3.68   # eV
        self.Eg_ZnSe = 2.70  # eV
        self.bowing = 0.50   # eV

        print("[AFLOW] Interface initialized")
        print(f"[AFLOW] Base URL: {self.base_url}")
        print(f"[AFLOW] Cache: {'enabled' if cache else 'disabled'}")

    def query_znsse_compounds(self, x_Se_target: Optional[float] = None) -> List[AFLOWEntry]:
        """
        Query AFLOW for Zn-S-Se ternary compounds.

        Parameters:
        -----------
        x_Se_target : float, optional
            Target Se composition (if None, returns all Zn-S-Se compounds)

        Returns:
        --------
        entries : list of AFLOWEntry
            Matching compounds with bandgap data
        """
        # Build AFLUX query
        # Format: /API/aflux/?species(Zn,S,Se),nspecies(3),Egap
        query_params = [
            "species(Zn,S,Se)",
            "nspecies(3)",
            "Egap"
        ]

        query_url = self.base_url + "?" + ",".join(query_params)

        print(f"[AFLOW] Querying: {query_url}")

        try:
            response = requests.get(query_url, timeout=self.timeout)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            if not data:
                print("[AFLOW] No results found")
                return []

            # Parse entries
            entries = []
            for item in data:
                try:
                    entry = AFLOWEntry(
                        aurl=item.get('aurl', ''),
                        compound=item.get('compound', ''),
                        composition=item.get('composition', []),
                        Egap=float(item.get('Egap', 0.0)),
                        Egap_type=item.get('Egap_type', 'unknown'),
                        energy_cell=float(item.get('energy_cell', 0.0)),
                        species=item.get('species', [])
                    )
                    entries.append(entry)
                except (KeyError, ValueError) as e:
                    print(f"[AFLOW] Warning: Failed to parse entry: {e}")
                    continue

            print(f"[AFLOW] Found {len(entries)} entries")

            # Filter by composition if specified
            if x_Se_target is not None:
                filtered = self._filter_by_composition(entries, x_Se_target)
                print(f"[AFLOW] Filtered to {len(filtered)} entries near x_Se={x_Se_target:.2f}")
                return filtered

            return entries

        except requests.exceptions.Timeout:
            print(f"[AFLOW] Error: Request timeout after {self.timeout}s")
            return []
        except requests.exceptions.RequestException as e:
            print(f"[AFLOW] Error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"[AFLOW] Error: Failed to parse JSON response: {e}")
            return []

    def _filter_by_composition(self, entries: List[AFLOWEntry], x_Se_target: float,
                               tolerance: float = 0.1) -> List[AFLOWEntry]:
        """
        Filter entries by Se composition.

        Parameters:
        -----------
        entries : list
            AFLOW entries
        x_Se_target : float
            Target Se fraction
        tolerance : float
            Composition tolerance (±)

        Returns:
        --------
        filtered : list
            Entries within tolerance
        """
        filtered = []

        for entry in entries:
            # Parse composition from entry
            x_Se = self._extract_se_composition(entry)

            if x_Se is not None:
                if abs(x_Se - x_Se_target) <= tolerance:
                    filtered.append(entry)

        return filtered

    def _extract_se_composition(self, entry: AFLOWEntry) -> Optional[float]:
        """
        Extract Se composition from AFLOW entry.

        Parameters:
        -----------
        entry : AFLOWEntry
            AFLOW database entry

        Returns:
        --------
        x_Se : float or None
            Selenium fraction
        """
        try:
            # Parse composition array
            # Format: [{"name": "Zn", "count": X}, {"name": "S", "count": Y}, ...]
            composition = entry.composition

            if isinstance(composition, str):
                composition = json.loads(composition)

            counts = {}
            for elem in composition:
                name = elem.get('name', '')
                count = float(elem.get('count', 0))
                counts[name] = count

            # Calculate Se fraction: Se / (S + Se)
            n_S = counts.get('S', 0)
            n_Se = counts.get('Se', 0)

            if (n_S + n_Se) > 0:
                x_Se = n_Se / (n_S + n_Se)
                return x_Se

        except (KeyError, ValueError, json.JSONDecodeError):
            pass

        return None

    def get_bandgap(self, x_Se: float, method: str = 'aflow_with_fallback') -> Tuple[float, str]:
        """
        Get bandgap for ZnS(1-x)Se(x) composition.

        Parameters:
        -----------
        x_Se : float
            Selenium composition (0.0 to 1.0)
        method : str
            'aflow' - AFLOW only (may fail)
            'aflow_with_fallback' - AFLOW with Vegard's law fallback (default)
            'vegard' - Vegard's law only

        Returns:
        --------
        Eg : float
            Bandgap in eV
        source : str
            'aflow' or 'vegard' (data source)
        """
        if not (0.0 <= x_Se <= 1.0):
            raise ValueError(f"x_Se must be in [0, 1], got {x_Se}")

        # Check cache
        if self.cache_enabled and x_Se in self.cache:
            print(f"[AFLOW] Using cached value for x_Se={x_Se:.3f}")
            return self.cache[x_Se], 'cached'

        # Method selection
        if method == 'vegard':
            Eg = self._vegard_law(x_Se)
            source = 'vegard'

        elif method == 'aflow':
            entries = self.query_znsse_compounds(x_Se_target=x_Se)

            if entries:
                # Use average if multiple entries
                Eg = np.mean([e.Egap for e in entries])
                source = 'aflow'
                print(f"[AFLOW] x_Se={x_Se:.3f}: Eg={Eg:.2f} eV (from {len(entries)} entries)")
            else:
                raise ValueError(f"No AFLOW data for x_Se={x_Se:.3f}")

        elif method == 'aflow_with_fallback':
            entries = self.query_znsse_compounds(x_Se_target=x_Se)

            if entries:
                Eg = np.mean([e.Egap for e in entries])
                source = 'aflow'
                print(f"[AFLOW] x_Se={x_Se:.3f}: Eg={Eg:.2f} eV (from {len(entries)} entries)")
            else:
                print(f"[AFLOW] No data for x_Se={x_Se:.3f}, using Vegard's law")
                Eg = self._vegard_law(x_Se)
                source = 'vegard'

        else:
            raise ValueError(f"Unknown method: {method}")

        # Cache result
        if self.cache_enabled:
            self.cache[x_Se] = Eg

        return Eg, source

    def _vegard_law(self, x_Se: float) -> float:
        """
        Calculate bandgap using Vegard's law with bowing.

        Eg(x) = (1-x)*Eg_ZnS + x*Eg_ZnSe - b*x*(1-x)

        Parameters:
        -----------
        x_Se : float
            Selenium composition

        Returns:
        --------
        Eg : float
            Bandgap in eV
        """
        Eg = (1 - x_Se) * self.Eg_ZnS + x_Se * self.Eg_ZnSe
        Eg -= self.bowing * x_Se * (1 - x_Se)
        return Eg

    def build_composition_database(self, n_points: int = 20) -> Dict:
        """
        Build database of bandgaps across composition range.

        Parameters:
        -----------
        n_points : int
            Number of compositions to query

        Returns:
        --------
        database : dict
            {x_Se: (Eg, source), ...}
        """
        print(f"\n[AFLOW] Building composition database ({n_points} points)...")
        print("="*60)

        database = {}
        compositions = np.linspace(0.0, 1.0, n_points)

        for i, x_Se in enumerate(compositions):
            print(f"\nQuerying {i+1}/{n_points}: x_Se = {x_Se:.3f}")

            Eg, source = self.get_bandgap(x_Se, method='aflow_with_fallback')
            database[x_Se] = (Eg, source)

            print(f"  → Eg = {Eg:.2f} eV (source: {source})")

        print("\n" + "="*60)
        print(f"Database complete: {len(database)} compositions")

        # Statistics
        aflow_count = sum(1 for (_, src) in database.values() if src == 'aflow')
        vegard_count = sum(1 for (_, src) in database.values() if src == 'vegard')

        print(f"  AFLOW data: {aflow_count}/{n_points}")
        print(f"  Vegard's law: {vegard_count}/{n_points}")

        return database


def demo_aflow_integration():
    """Demonstrate AFLOW API integration."""
    print("\n" + "="*60)
    print("AFLOW API Integration Demo")
    print("="*60)

    aflow = AFLOWInterface(timeout=30, cache=True)

    # Test 1: Query all Zn-S-Se compounds
    print("\n" + "-"*60)
    print("TEST 1: Query all Zn-S-Se ternary compounds")
    print("-"*60)

    entries = aflow.query_znsse_compounds()

    if entries:
        print(f"\nFound {len(entries)} compounds:")
        for i, entry in enumerate(entries[:5]):  # Show first 5
            print(f"  {i+1}. {entry.compound}")
            print(f"     Bandgap: {entry.Egap:.2f} eV ({entry.Egap_type})")
            print(f"     URL: {entry.aurl}")
    else:
        print("\nNo AFLOW data available (this is expected - database may not have ZnSSe)")
        print("Will use Vegard's law fallback in practice")

    # Test 2: Get bandgap for specific composition
    print("\n" + "-"*60)
    print("TEST 2: Get bandgap for specific compositions")
    print("-"*60)

    test_compositions = [0.0, 0.25, 0.50, 0.75, 1.0]

    print(f"\n{'x_Se':>6} | {'Bandgap (eV)':>15} | {'Source':>10}")
    print("-"*60)

    for x_Se in test_compositions:
        Eg, source = aflow.get_bandgap(x_Se, method='aflow_with_fallback')
        print(f"{x_Se:6.2f} | {Eg:15.2f} | {source:>10}")

    # Test 3: Build composition database
    print("\n" + "-"*60)
    print("TEST 3: Build composition database (5 points)")
    print("-"*60)

    database = aflow.build_composition_database(n_points=5)

    print("\nNote: Since AFLOW may not have ZnSSe data, Vegard's law")
    print("      is used as fallback. This is the expected behavior.")
    print("\nIn real experiments, you would:")
    print("  1. Use AFLOW data when available")
    print("  2. Supplement with experimental measurements")
    print("  3. Use Vegard's law for interpolation")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo_aflow_integration()
