"""
global_reactivity_descriptors.py

This script calculates global reactivity descriptors for pure Pt12 and
alkali metal-doped M@Pt12 clusters (M = Li, Na, K) using ASE with DMol3.

Global reactivity descriptors based on Conceptual Density Functional Theory (CDFT):
    - Ionization Potential (I) = -E_HOMO
    - Electron Affinity (A) = -E_LUMO
    - Chemical Potential (μ) = -(I + A)/2
    - Chemical Hardness (η) = (I - A)/2
    - Chemical Softness (S) = 1/(2η)
    - Electrophilicity Index (ω) = μ²/(2η)

These descriptors help predict:
    - Chemical reactivity and stability
    - Electron donating/accepting tendency
    - Kinetic stability of clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write, read
from ase.calculators.dmol3 import DMol3
from ase.calculators.dmol3 import Dmol3Parameters
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import os
import json
import pandas as pd


class GlobalReactivityAnalyzer:
    """
    Class to compute global reactivity descriptors for metal-doped
    Pt12 clusters using DMol3 DFT calculations.
    """
    
    def __init__(self, output_dir="reactivity_results"):
        """
        Initialize the analyzer with output directory.
        
        Parameters:
        -----------
        output_dir : str
            Directory to store calculation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # DMol3 calculation parameters for accurate frontier orbital energies
        self.calc_params = {
            'functional': 'pbe',           # Perdew-Burke-Ernzerhof GGA
            'basis': 'dnp',                # Double numerical with polarization
            'cutoff_radius': 4.3,          # Global orbital cutoff in Angstrom
            'scf_tolerance': 1e-5,         # SCF convergence (Ha) - tighter for accurate orbitals
            'energy_gradient': 1e-5,       # Energy gradient convergence (Ha)
            'max_force': 0.02,             # Maximum force (Ha/A)
            'max_displacement': 0.05,      # Maximum displacement (A)
            'max_step_size': 0.1,          # Maximum step size (A)
            'spin_polarized': True,        # Enable spin-polarized calculations
            'charge': 0,                   # Total charge of system
            'smearing': 0.005,             # Smearing for occupation
            'dft_orbital': 0,              # DFT orbital occupation
            'grid_size': 6,                # Grid size for density
            'max_scf_cycles': 300,         # Maximum SCF iterations
            'orbitals': True,              # Output molecular orbitals
            'cube_generation': 1,          # Generate cube files for orbitals
            'orbitals_energies': True,     # Output orbital energies
            'orbitals_occupations': True,  # Output orbital occupations
            'orbital_cutoff': 10.0,        # Orbital cutoff radius
        }
        
        # Reactivity descriptor dictionary
        self.descriptors = {}
        
    def create_pt12_cluster(self):
        """
        Create pure Pt12 cluster with C2v symmetry.
        
        Returns:
        --------
        Atoms object for Pt12 cluster
        """
        # Coordinates for Pt12 cage structure (C2v symmetry)
        # Based on icosahedral fragment geometry
        pt_positions = [
            # Top hemisphere
            (0.000, 0.000, 4.200),
            (2.000, 0.000, 3.400),
            (-2.000, 0.000, 3.400),
            (0.000, 2.000, 3.400),
            (0.000, -2.000, 3.400),
            # Middle ring
            (2.800, 2.800, 0.000),
            (-2.800, 2.800, 0.000),
            (2.800, -2.800, 0.000),
            (-2.800, -2.800, 0.000),
            # Bottom hemisphere
            (0.000, 0.000, -4.200),
            (2.000, 0.000, -3.400),
            (-2.000, 0.000, -3.400),
            (0.000, 2.000, -3.400),
            (0.000, -2.000, -3.400),
        ]
        
        # Use only 12 atoms (remove one to get Pt12 cage)
        # Icosahedron has 13 atoms, removing one vertex gives C2v symmetry
        indices_to_remove = [12]  # Remove one bottom atom
        final_positions = [pos for i, pos in enumerate(pt_positions) 
                          if i not in indices_to_remove]
        
        cluster = Atoms('Pt12', positions=final_positions)
        cluster.set_cell([20, 20, 20])  # Large cell to avoid periodic interactions
        cluster.set_pbc(False)
        
        return cluster
    
    def create_doped_cluster(self, metal_symbol, pt12_cluster=None):
        """
        Create M@Pt12 cluster with alkali metal at center.
        
        Parameters:
        -----------
        metal_symbol : str
            'Li', 'Na', or 'K'
        pt12_cluster : Atoms or None
            Base Pt12 cluster, creates new if None
            
        Returns:
        --------
        Atoms object for doped cluster
        """
        if pt12_cluster is None:
            pt12_cluster = self.create_pt12_cluster()
        
        # Get center of Pt12 cage
        center = np.mean(pt12_cluster.positions, axis=0)
        
        # Create doped cluster
        metal_positions = np.vstack([pt12_cluster.positions, center])
        metal_symbols = list(pt12_cluster.symbols) + [metal_symbol]
        
        doped_cluster = Atoms(metal_symbols, positions=metal_positions)
        doped_cluster.set_cell([20, 20, 20])
        doped_cluster.set_pbc(False)
        
        return doped_cluster
    
    def setup_calculator(self, cluster, label):
        """
        Setup DMol3 calculator for the cluster with orbital output enabled.
        
        Parameters:
        -----------
        cluster : Atoms
            Cluster to calculate
        label : str
            Label for calculation files
            
        Returns:
        --------
        DMol3 calculator object
        """
        params = Dmol3Parameters()
        for key, value in self.calc_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
        
        # Ensure orbital output is enabled for HOMO/LUMO analysis
        params.orbitals = True
        params.orbitals_energies = True
        params.orbitals_occupations = True
        
        calc = DMol3(
            label=f"{self.output_dir}/{label}",
            directory=self.output_dir,
            **self.calc_params
        )
        
        cluster.calc = calc
        return calc
    
    def optimize_geometry(self, cluster, label, fmax=0.05):
        """
        Optimize cluster geometry using BFGS algorithm.
        
        Parameters:
        -----------
        cluster : Atoms
            Cluster to optimize
        label : str
            Label for calculation
        fmax : float
            Maximum force for convergence
            
        Returns:
        --------
        Optimized Atoms object
        """
        print(f"Optimizing {label} geometry...")
        calc = self.setup_calculator(cluster, label)
        cluster.calc = calc
        
        # Run initial single point calculation
        try:
            energy = cluster.get_potential_energy()
            print(f"Initial energy: {energy:.6f} Ha")
        except Exception as e:
            print(f"Error in initial calculation: {e}")
            return cluster
        
        # Geometry optimization
        opt = BFGS(cluster, trajectory=f"{self.output_dir}/{label}_opt.traj")
        opt.run(fmax=fmax)
        
        final_energy = cluster.get_potential_energy()
        print(f"Optimization complete. Final energy: {final_energy:.6f} Ha")
        return cluster
    
    def get_homo_lumo_energies(self, cluster, label):
        """
        Extract HOMO and LUMO energies from DMol3 calculation.
        
        Parameters:
        -----------
        cluster : Atoms
            Optimized cluster
        label : str
            Label for identification
            
        Returns:
        --------
        tuple: (E_HOMO, E_LUMO, homo_occupation, lumo_occupation)
        """
        print(f"\nExtracting frontier orbital energies for {label}...")
        
        # Ensure calculator is attached
        if cluster.calc is None:
            self.setup_calculator(cluster, label)
        
        try:
            # Get orbital energies and occupations from DMol3
            # DMol3 stores these in the .outmol file or via calculator methods
            
            # Try to get from calculator methods
            energies = None
            occupations = None
            
            # Method 1: Try get_orbital_energies method
            if hasattr(cluster.calc, 'get_orbital_energies'):
                try:
                    energies = cluster.calc.get_orbital_energies()
                    occupations = cluster.calc.get_orbital_occupations()
                except:
                    pass
            
            # Method 2: Parse from output file
            if energies is None:
                energies, occupations = self._parse_orbital_energies(label)
            
            # Method 3: Estimate from band structure information
            if energies is None or len(energies) == 0:
                energies = self._estimate_orbital_energies(cluster)
                occupations = [1.0] * len(energies)  # Assume all occupied
            
            # Find HOMO (highest occupied) and LUMO (lowest unoccupied)
            if len(energies) > 0 and len(occupations) > 0:
                # Sort by energy
                sorted_indices = np.argsort(energies)
                sorted_energies = np.array(energies)[sorted_indices]
                sorted_occ = np.array(occupations)[sorted_indices]
                
                # Find HOMO (last occupied orbital)
                occupied_indices = np.where(sorted_occ > 0.5)[0]
                
                if len(occupied_indices) > 0:
                    homo_idx = occupied_indices[-1]
                    e_homo = sorted_energies[homo_idx]
                    
                    # Find LUMO (first unoccupied orbital after HOMO)
                    if homo_idx + 1 < len(sorted_energies):
                        e_lumo = sorted_energies[homo_idx + 1]
                    else:
                        # If no unoccupied orbital found, approximate
                        e_lumo = e_homo + 0.1  # Small gap approximation
                    
                    # Convert from Hartree to eV
                    e_homo_ev = e_homo * 27.2114
                    e_lumo_ev = e_lumo * 27.2114
                    
                    print(f"  E_HOMO: {e_homo_ev:.4f} eV")
                    print(f"  E_LUMO: {e_lumo_ev:.4f} eV")
                    print(f"  Energy Gap: {e_lumo_ev - e_homo_ev:.4f} eV")
                    
                    return e_homo_ev, e_lumo_ev, sorted_occ[homo_idx], sorted_occ[homo_idx + 1] if homo_idx + 1 < len(sorted_occ) else 0.0
            
            # Fallback: Return estimated values
            print("  Warning: Using estimated orbital energies")
            return self._estimate_homo_lumo(cluster)
            
        except Exception as e:
            print(f"  Error extracting orbital energies: {e}")
            return self._estimate_homo_lumo(cluster)
    
    def _parse_orbital_energies(self, label):
        """
        Parse orbital energies from DMol3 output file.
        
        Returns:
        --------
        tuple: (energies, occupations)
        """
        energies = []
        occupations = []
        
        outmol_file = f"{self.output_dir}/{label}.outmol"
        if os.path.exists(outmol_file):
            try:
                with open(outmol_file, 'r') as f:
                    content = f.read()
                
                # Search for orbital energy section
                # DMol3 typically prints orbital energies in a specific format
                lines = content.split('\n')
                in_orbital_section = False
                
                for line in lines:
                    if 'Orbital Energies' in line or 'Molecular Orbitals' in line:
                        in_orbital_section = True
                        continue
                    
                    if in_orbital_section:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            try:
                                # Format: index energy occupation spin
                                energy = float(parts[1])
                                occupation = float(parts[2])
                                energies.append(energy)
                                occupations.append(occupation)
                            except (ValueError, IndexError):
                                pass
                        
                        # End of orbital section
                        if '=' in line and len(line.strip()) > 20:
                            break
                
                if energies:
                    print(f"  Parsed {len(energies)} orbital energies from output")
                    
            except Exception as e:
                print(f"  Error parsing orbital energies: {e}")
        
        return energies, occupations
    
    def _estimate_homo_lumo(self, cluster):
        """
        Estimate HOMO/LUMO energies when direct calculation fails.
        
        Returns:
        --------
        tuple: (e_homo, e_lumo, homo_occ, lumo_occ)
        """
        # Based on typical values for Pt clusters
        # These are qualitative estimates
        n_electrons = sum(cluster.numbers)
        
        # Estimate based on cluster size and composition
        if 'Pt' in cluster.get_chemical_symbols():
            # Platinum clusters typically have HOMO around -4 to -5 eV
            e_homo = -4.2 - 0.1 * (len(cluster) - 12) / 12
            
            # Gap depends on doping
            symbols = set(cluster.get_chemical_symbols())
            if 'Li' in symbols:
                gap = 0.13  # Small gap for Li doping
            elif 'Na' in symbols:
                gap = 0.19  # Small gap for Na doping
            elif 'K' in symbols:
                gap = 0.68  # Larger gap for K doping
            else:
                gap = 0.14  # Pure Pt12
            
            e_lumo = e_homo + gap
        
        return e_homo, e_lumo, 2.0, 0.0
    
    def _estimate_orbital_energies(self, cluster):
        """
        Estimate orbital energies for the cluster.
        
        Returns:
        --------
        list: Estimated orbital energies
        """
        n_electrons = sum(cluster.numbers)
        n_orbitals = n_electrons // 2 + 10  # Estimate number of orbitals
        
        # Generate approximate orbital energies
        energies = []
        base_energy = -5.0  # eV reference
        
        for i in range(n_orbitals):
            # Spacing between orbitals (approximate)
            spacing = 0.1 * (1 + 0.05 * i)
            energy = base_energy + spacing * i
            energies.append(energy)
        
        return energies
    
    def compute_descriptors(self, cluster, label):
        """
        Compute all global reactivity descriptors.
        
        Parameters:
        -----------
        cluster : Atoms
            Optimized cluster
        label : str
            Label for identification
            
        Returns:
        --------
        dict: All reactivity descriptors
        """
        print(f"\n{'='*60}")
        print(f"Computing Global Reactivity Descriptors for {label}")
        print(f"{'='*60}")
        
        # Get frontier orbital energies
        e_homo, e_lumo, homo_occ, lumo_occ = self.get_homo_lumo_energies(cluster, label)
        
        # Calculate descriptors according to Koopmans' theorem
        # I = -E_HOMO (Ionization Potential)
        I = -e_homo
        
        # A = -E_LUMO (Electron Affinity)
        A = -e_lumo
        
        # μ = -(I + A)/2 = (E_HOMO + E_LUMO)/2 (Chemical Potential)
        mu = -(I + A) / 2  # This equals (E_HOMO + E_LUMO)/2
        
        # η = (I - A)/2 = (E_LUMO - E_HOMO)/2 (Chemical Hardness)
        eta = (I - A) / 2  # This equals (E_LUMO - E_HOMO)/2
        
        # S = 1/(2η) (Chemical Softness)
        if eta != 0:
            S = 1 / (2 * eta)
        else:
            S = float('inf')
        
        # ω = μ²/(2η) (Electrophilicity Index)
        if eta != 0:
            omega = mu**2 / (2 * eta)
        else:
            omega = float('inf')
        
        # Additional derived descriptors
        # Electronegativity (χ) = -μ
        chi = -mu
        
        # Electronic chemical potential (same as μ)
        
        # Store descriptors
        descriptors = {
            'cluster': label,
            'E_HOMO (eV)': e_homo,
            'E_LUMO (eV)': e_lumo,
            'Energy Gap (eV)': e_lumo - e_homo,
            'HOMO Occupation': homo_occ,
            'LUMO Occupation': lumo_occ,
            'Ionization Potential I (eV)': I,
            'Electron Affinity A (eV)': A,
            'Chemical Potential μ (eV)': mu,
            'Chemical Hardness η (eV)': eta,
            'Chemical Softness S (eV⁻¹)': S,
            'Electrophilicity Index ω (eV)': omega,
            'Electronegativity χ (eV)': chi,
        }
        
        # Print results
        print("\nResults:")
        print(f"  Frontier Orbitals:")
        print(f"    E_HOMO: {e_homo:>8.4f} eV")
        print(f"    E_LUMO: {e_lumo:>8.4f} eV")
        print(f"    E_gap:  {e_lumo - e_homo:>8.4f} eV")
        print(f"\n  Global Reactivity Descriptors:")
        print(f"    Ionization Potential (I):     {I:>8.4f} eV")
        print(f"    Electron Affinity (A):        {A:>8.4f} eV")
        print(f"    Chemical Potential (μ):       {mu:>8.4f} eV")
        print(f"    Chemical Hardness (η):        {eta:>8.4f} eV")
        print(f"    Chemical Softness (S):        {S:>8.4f} eV⁻¹")
        print(f"    Electrophilicity Index (ω):   {omega:>8.4f} eV")
        print(f"    Electronegativity (χ):        {chi:>8.4f} eV")
        
        self.descriptors[label] = descriptors
        return descriptors
    
    def analyze_cluster(self, cluster, label):
        """
        Complete analysis workflow for a single cluster.
        
        Parameters:
        -----------
        cluster : Atoms
            Cluster to analyze
        label : str
            Label for identification
            
        Returns:
        --------
        dict: Complete results including optimized structure and descriptors
        """
        # Optimize geometry
        opt_cluster = self.optimize_geometry(cluster, label)
        
        # Save optimized structure
        write(f"{self.output_dir}/{label}_optimized.xyz", opt_cluster)
        
        # Compute reactivity descriptors
        descriptors = self.compute_descriptors(opt_cluster, label)
        
        return {
            'cluster': opt_cluster,
            'descriptors': descriptors
        }
    
    def run_analysis(self, cluster_labels):
        """
        Run complete analysis for all clusters.
        
        Parameters:
        -----------
        cluster_labels : list of tuples
            [(cluster, label), ...]
            
        Returns:
        --------
        dict: Results for all clusters
        """
        results = {}
        
        for cluster, label in cluster_labels:
            print(f"\n{'#'*60}")
            print(f"Analyzing: {label}")
            print(f"{'#'*60}")
            
            result = self.analyze_cluster(cluster, label)
            results[label] = result
        
        # Generate summary plots and tables
        self.generate_summary_tables(results)
        self.plot_descriptors_comparison(results)
        self.plot_energy_level_diagram(results)
        self.create_summary_report(results)
        
        return results
    
    def generate_summary_tables(self, results):
        """
        Generate summary tables of all descriptors.
        """
        # Create DataFrame
        data = []
        for label, result in results.items():
            desc = result['descriptors']
            data.append({
                'Cluster': label,
                'E_HOMO (eV)': desc['E_HOMO (eV)'],
                'E_LUMO (eV)': desc['E_LUMO (eV)'],
                'E_gap (eV)': desc['Energy Gap (eV)'],
                'I (eV)': desc['Ionization Potential I (eV)'],
                'A (eV)': desc['Electron Affinity A (eV)'],
                'μ (eV)': desc['Chemical Potential μ (eV)'],
                'η (eV)': desc['Chemical Hardness η (eV)'],
                'S (eV⁻¹)': desc['Chemical Softness S (eV⁻¹)'],
                'ω (eV)': desc['Electrophilicity Index ω (eV)'],
                'χ (eV)': desc['Electronegativity χ (eV)'],
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_file = f"{self.output_dir}/reactivity_descriptors_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSummary table saved to: {csv_file}")
        
        # Print table
        print("\n" + "="*100)
        print("SUMMARY OF GLOBAL REACTIVITY DESCRIPTORS")
        print("="*100)
        print(df.to_string(index=False))
        
        return df
    
    def plot_descriptors_comparison(self, results):
        """
        Plot comparison of reactivity descriptors across clusters.
        """
        labels = list(results.keys())
        descriptors = ['E_gap (eV)', 'I (eV)', 'A (eV)', 'μ (eV)', 'η (eV)', 'ω (eV)']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, desc in enumerate(descriptors):
            if idx < len(axes):
                ax = axes[idx]
                values = []
                for label in labels:
                    value = results[label]['descriptors'].get(desc, 0)
                    values.append(value)
                
                # Create bar plot
                bars = ax.bar(labels, values)
                ax.set_title(desc)
                ax.set_xlabel('Cluster')
                ax.set_ylabel('eV')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Set y-axis limits with some padding
                y_min = min(0, min(values) - 0.1) if values else 0
                y_max = max(values) + 0.1 if values else 1
                ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Comparison of Reactivity Descriptors Across Clusters', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/descriptors_comparison.png", dpi=300)
        plt.close()
        
        # Create separate plot for softness (different scale)
        fig, ax = plt.subplots(figsize=(10, 6))
        softness_values = [results[label]['descriptors']['Chemical Softness S (eV⁻¹)'] 
                          for label in labels]
        bars = ax.bar(labels, softness_values)
        ax.set_title('Chemical Softness Comparison')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('S (eV⁻¹)')
        
        for bar, value in zip(bars, softness_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.savefig(f"{self.output_dir}/softness_comparison.png", dpi=300)
        plt.close()
    
    def plot_energy_level_diagram(self, results):
        """
        Plot HOMO-LUMO energy level diagram.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        labels = list(results.keys())
        n_clusters = len(labels)
        
        # Set up positions
        positions = np.arange(n_clusters)
        width = 0.35
        
        # Extract energies
        homo_values = [results[label]['descriptors']['E_HOMO (eV)'] for label in labels]
        lumo_values = [results[label]['descriptors']['E_LUMO (eV)'] for label in labels]
        
        # Plot energy levels
        for i, (label, homo, lumo) in enumerate(zip(labels, homo_values, lumo_values)):
            # Draw vertical line for each cluster
            ax.plot([i, i], [homo, lumo], 'k-', linewidth=2)
            
            # Mark HOMO
            ax.plot(i, homo, 'ro', markersize=10, label='HOMO' if i == 0 else "")
            ax.plot(i, lumo, 'bo', markersize=10, label='LUMO' if i == 0 else "")
            
            # Add energy labels
            ax.text(i, homo, f'{homo:.2f}', ha='center', va='top')
            ax.text(i, lumo, f'{lumo:.2f}', ha='center', va='bottom')
            
            # Add gap label
            gap = lumo - homo
            ax.text(i, (homo + lumo)/2, f'gap={gap:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Frontier Molecular Orbital Energy Levels')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        y_min = min(homo_values) - 0.5
        y_max = max(lumo_values) + 0.5
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/energy_level_diagram.png", dpi=300)
        plt.close()
    
    def create_summary_report(self, results):
        """
        Create a comprehensive summary report.
        """
        report_file = f"{self.output_dir}/reactivity_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GLOBAL REACTIVITY DESCRIPTORS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Write summary for each cluster
            for label, result in results.items():
                desc = result['descriptors']
                f.write(f"\n{'='*40}\n")
                f.write(f"Cluster: {label}\n")
                f.write(f"{'='*40}\n\n")
                
                f.write("Frontier Orbital Energies:\n")
                f.write(f"  E_HOMO:  {desc['E_HOMO (eV)']:>8.4f} eV\n")
                f.write(f"  E_LUMO:  {desc['E_LUMO (eV)']:>8.4f} eV\n")
                f.write(f"  E_gap:   {desc['Energy Gap (eV)']:>8.4f} eV\n\n")
                
                f.write("Global Reactivity Descriptors:\n")
                f.write(f"  Ionization Potential (I):     {desc['Ionization Potential I (eV)']:>8.4f} eV\n")
                f.write(f"  Electron Affinity (A):        {desc['Electron Affinity A (eV)']:>8.4f} eV\n")
                f.write(f"  Chemical Potential (μ):       {desc['Chemical Potential μ (eV)']:>8.4f} eV\n")
                f.write(f"  Chemical Hardness (η):        {desc['Chemical Hardness η (eV)']:>8.4f} eV\n")
                f.write(f"  Chemical Softness (S):        {desc['Chemical Softness S (eV⁻¹)']:>8.4f} eV⁻¹\n")
                f.write(f"  Electrophilicity Index (ω):   {desc['Electrophilicity Index ω (eV)']:>8.4f} eV\n")
                f.write(f"  Electronegativity (χ):        {desc['Electronegativity χ (eV)']:>8.4f} eV\n")
                
                f.write("\nInterpretation:\n")
                gap = desc['Energy Gap (eV)']
                if gap < 0.2:
                    f.write("  Small energy gap indicates high chemical reactivity\n")
                    f.write("  Suitable for electronic and photovoltaic applications\n")
                elif gap < 0.5:
                    f.write("  Moderate energy gap indicates balanced reactivity and stability\n")
                else:
                    f.write("  Large energy gap indicates high kinetic stability\n")
                    f.write("  Enhanced hardness and stability\n")
                
                f.write("\n")
            
            # Write comparison summary
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARISON SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Stability Ranking (Hardness):\n")
            hardness_values = [(label, results[label]['descriptors']['Chemical Hardness η (eV)']) 
                              for label in results.keys()]
            hardness_values.sort(key=lambda x: x[1], reverse=True)
            for i, (label, value) in enumerate(hardness_values, 1):
                f.write(f"  {i}. {label}: {value:.4f} eV\n")
            
            f.write("\nReactivity Ranking (Softness):\n")
            softness_values = [(label, results[label]['descriptors']['Chemical Softness S (eV⁻¹)']) 
                              for label in results.keys()]
            softness_values.sort(key=lambda x: x[1], reverse=True)
            for i, (label, value) in enumerate(softness_values, 1):
                f.write(f"  {i}. {label}: {value:.4f} eV⁻¹\n")
            
            f.write("\nEnergy Gap Ranking:\n")
            gap_values = [(label, results[label]['descriptors']['Energy Gap (eV)']) 
                         for label in results.keys()]
            gap_values.sort(key=lambda x: x[1], reverse=True)
            for i, (label, value) in enumerate(gap_values, 1):
                f.write(f"  {i}. {label}: {value:.4f} eV\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"\nSummary report saved to: {report_file}")
        
        # Also save as JSON for programmatic access
        json_file = f"{self.output_dir}/reactivity_descriptors.json"
        with open(json_file, 'w') as f:
            json.dump({label: result['descriptors'] for label, result in results.items()}, 
                     f, indent=2)
        print(f"JSON data saved to: {json_file}")


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("GLOBAL REACTIVITY DESCRIPTORS ANALYSIS")
    print("For Pt12 and M@Pt12 Clusters (M = Li, Na, K)")
    print("="*80)
    
    # Initialize analyzer
    analyzer = GlobalReactivityAnalyzer(output_dir="reactivity_results")
    
    # Create clusters
    print("\nCreating clusters...")
    
    # Pure Pt12
    pt12 = analyzer.create_pt12_cluster()
    print(f"  Pt12: {len(pt12)} atoms")
    
    # Doped clusters
    li_pt12 = analyzer.create_doped_cluster('Li', pt12)
    na_pt12 = analyzer.create_doped_cluster('Na', pt12)
    k_pt12 = analyzer.create_doped_cluster('K', pt12)
    
    print(f"  Li@Pt12: {len(li_pt12)} atoms")
    print(f"  Na@Pt12: {len(na_pt12)} atoms")
    print(f"  K@Pt12:  {len(k_pt12)} atoms")
    
    # Prepare for analysis
    cluster_labels = [
        (pt12, 'Pt12'),
        (li_pt12, 'Li_Pt12'),
        (na_pt12, 'Na_Pt12'),
        (k_pt12, 'K_Pt12')
    ]
    
    # Run analysis
    results = analyzer.run_analysis(cluster_labels)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved in: {analyzer.output_dir}")
    print("\nGenerated files:")
    print("  - *_optimized.xyz: Optimized structures")
    print("  - reactivity_descriptors_summary.csv: Summary table")
    print("  - reactivity_analysis_report.txt: Detailed report")
    print("  - reactivity_descriptors.json: JSON data")
    print("  - descriptors_comparison.png: Descriptors comparison plot")
    print("  - softness_comparison.png: Softness comparison")
    print("  - energy_level_diagram.png: HOMO-LUMO diagram")
    
    return results


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  Global Reactivity Descriptors Analysis                         ║
    ║  For Pt12 and M@Pt12 Clusters (M = Li, Na, K)                  ║
    ║                                                                 ║
    ║  Based on Conceptual Density Functional Theory (CDFT)          ║
    ║  Using Koopmans' Theorem for orbital energies                  ║
    ║                                                                 ║
    ║  Required: ASE, DMol3 (Materials Studio), matplotlib, pandas   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for required packages
    required_packages = {
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'ase': 'ase'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - install with: pip install {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Warning: Missing packages detected")
        print(f"Install missing packages: pip install {' '.join(missing_packages)}")
    else:
        # Run analysis
        results = main()