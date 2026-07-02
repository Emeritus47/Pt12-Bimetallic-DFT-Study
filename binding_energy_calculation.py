"""
binding_energy_calculation.py

Description:
This script calculates the binding energy (Eb) of alkali metal-doped Pt12 clusters
(M@Pt12, where M = Li, Na, K) using the Atomic Simulation Environment (ASE)
with the Dmol3 calculator.

Binding energy is defined as:
Eb = E_total(M@Pt12) - [E_total(Pt12_pure) + E_total(M_atom)]

A negative Eb indicates an exothermic process and thermodynamic stability.

Requirements:
    - ASE (Atomic Simulation Environment)
    - Dmol3 (Materials Studio or standalone version)
    - NumPy
"""

import numpy as np
from ase import Atoms
from ase.calculators.dmol3 import Dmol3
from ase.optimize import BFGS
from ase.io import write, read
import os
import time


class BindingEnergyCalculator:
    """
    A class to calculate binding energies of doped Pt12 clusters using Dmol3.
    """
    
    def __init__(self, work_dir="./dmol3_calculations"):
        """
        Initialize the BindingEnergyCalculator.
        
        Parameters:
        -----------
        work_dir : str
            Directory to store calculation files
        """
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        
        # Dmol3 calculator parameters (general settings)
        self.calculator_params = {
            'task': 'GeometryOptimization',
            'xc': 'PBE',  # Perdew-Burke-Ernzerhof functional
            'basis': 'DNP',  # Double numerical with polarization
            'functional': 'GGA',
            'cutoff': 4.3,  # Global orbital cutoff in Angstrom
            'smearing': 0.005,  # Smearing for occupation
            'spin': 'unrestricted',  # Unrestricted spin
            'charge': 0,
            'max_scf_cycles': 500,
            'scf_convergence': 1e-4,  # SCF tolerance in Ha
            'max_force': 0.02,  # Maximum force in Ha/A
            'max_displacement': 0.05,  # Maximum displacement in A
            'energy_gradient': 1e-5,  # Energy gradient in Ha
        }
        
        # Atomic radii for initial guess (approximate)
        self.pt_radius = 1.35  # Approximate Pt atomic radius
        self.m_radii = {
            'Li': 1.52,
            'Na': 1.86,
            'K': 2.27,
        }
        
    def create_pt12_cage(self):
        """
        Create a Pt12 icosahedral cage structure.
        
        Returns:
        --------
        Atoms object representing Pt12 cage
        """
        # Icosahedral geometry coordinates (normalized)
        # Vertices of an icosahedron
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # 12 vertices of icosahedron
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
        ]
        
        # Normalize and scale to get desired Pt-Pt distance
        # Pt-Pt distance in pure Pt12 from literature ~2.84 Å
        target_pt_pt_distance = 2.84
        scale_factor = target_pt_pt_distance / 2.0  # Approximate scaling
        
        positions = []
        for v in vertices:
            # Normalize
            norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            pos = np.array(v) / norm * scale_factor
            positions.append(pos)
        
        # Create Atoms object
        pt12 = Atoms('Pt12', positions=positions, cell=[10, 10, 10], pbc=False)
        
        return pt12
    
    def create_doped_cluster(self, metal_symbol, pt12_cage=None):
        """
        Create a doped M@Pt12 cluster with metal at the center.
        
        Parameters:
        -----------
        metal_symbol : str
            'Li', 'Na', or 'K'
        pt12_cage : Atoms, optional
            Pre-existing Pt12 cage to dope. If None, creates a new one.
        
        Returns:
        --------
        Atoms object representing M@Pt12 cluster
        """
        if pt12_cage is None:
            pt12_cage = self.create_pt12_cage()
        
        # Get the center of mass of the cage
        center = pt12_cage.get_center_of_mass()
        
        # Create metal atom at the center
        metal_atom = Atoms(metal_symbol, positions=[center])
        
        # Combine cage and metal atom
        doped_cluster = pt12_cage + metal_atom
        
        # Adjust cell if needed
        doped_cluster.set_cell([12, 12, 12], scale_atoms=False)
        doped_cluster.center()
        
        return doped_cluster
    
    def setup_calculator(self, atoms, label_prefix="calc"):
        """
        Set up Dmol3 calculator for a given Atoms object.
        
        Parameters:
        -----------
        atoms : Atoms
            ASE Atoms object
        label_prefix : str
            Prefix for calculation label
        
        Returns:
        --------
        Dmol3 calculator instance
        """
        # Create unique label with timestamp
        timestamp = int(time.time())
        label = f"{label_prefix}_{timestamp}"
        
        calc = Dmol3(
            label=label,
            directory=self.work_dir,
            **self.calculator_params
        )
        
        return calc
    
    def calculate_energy(self, atoms, label_prefix="calc", optimize=True):
        """
        Calculate the total energy of a cluster.
        
        Parameters:
        -----------
        atoms : Atoms
            ASE Atoms object
        label_prefix : str
            Prefix for calculation label
        optimize : bool
            Whether to perform geometry optimization
        
        Returns:
        --------
        tuple: (optimized_atoms, total_energy_in_eV)
        """
        calc = self.setup_calculator(atoms, label_prefix)
        atoms.set_calculator(calc)
        
        if optimize:
            # Perform geometry optimization
            optimizer = BFGS(atoms, trajectory=f"{label_prefix}_opt.traj")
            optimizer.run(fmax=0.02)
        
        # Get total energy (in Hartree, convert to eV)
        energy_hartree = atoms.get_potential_energy()
        energy_ev = energy_hartree * 27.2114  # 1 Hartree = 27.2114 eV
        
        return atoms, energy_ev
    
    def calculate_atom_energy(self, metal_symbol):
        """
        Calculate the energy of an isolated metal atom.
        
        Parameters:
        -----------
        metal_symbol : str
            'Li', 'Na', or 'K'
        
        Returns:
        --------
        float: Energy of the isolated atom in eV
        """
        # Create a single atom in a large box (to avoid periodic interactions)
        atom = Atoms(metal_symbol, positions=[(0, 0, 0)])
        atom.set_cell([15, 15, 15], scale_atoms=False)
        atom.center()
        
        # Calculate energy without optimization (for isolated atom)
        _, energy_ev = self.calculate_energy(
            atom, 
            label_prefix=f"atom_{metal_symbol}",
            optimize=False  # No need to optimize single atom
        )
        
        return energy_ev
    
    def calculate_binding_energy(self, metal_symbol):
        """
        Calculate the binding energy for M@Pt12 cluster.
        
        Parameters:
        -----------
        metal_symbol : str
            'Li', 'Na', or 'K'
        
        Returns:
        --------
        dict containing:
            - 'metal': metal symbol
            - 'e_doped': energy of doped cluster (eV)
            - 'e_pure': energy of pure Pt12 (eV)
            - 'e_atom': energy of isolated metal atom (eV)
            - 'binding_energy': binding energy (eV)
            - 'optimized_doped': optimized doped cluster
            - 'optimized_pure': optimized pure Pt12
        """
        print(f"\n{'='*60}")
        print(f"Calculating binding energy for {metal_symbol}@Pt12")
        print(f"{'='*60}")
        
        # Step 1: Create and optimize pure Pt12
        print("\n1. Optimizing pure Pt12 cage...")
        pt12_cage = self.create_pt12_cage()
        optimized_pt12, e_pure = self.calculate_energy(
            pt12_cage, 
            label_prefix=f"Pt12_pure"
        )
        print(f"   Pure Pt12 energy: {e_pure:.4f} eV")
        
        # Step 2: Create and optimize doped cluster
        print(f"\n2. Optimizing {metal_symbol}@Pt12 cluster...")
        doped_cluster = self.create_doped_cluster(metal_symbol, optimized_pt12)
        optimized_doped, e_doped = self.calculate_energy(
            doped_cluster, 
            label_prefix=f"{metal_symbol}Pt12"
        )
        print(f"   {metal_symbol}@Pt12 energy: {e_doped:.4f} eV")
        
        # Step 3: Calculate isolated metal atom energy
        print(f"\n3. Calculating isolated {metal_symbol} atom energy...")
        e_atom = self.calculate_atom_energy(metal_symbol)
        print(f"   Isolated {metal_symbol} energy: {e_atom:.4f} eV")
        
        # Step 4: Calculate binding energy
        # Eb = E_total(M@Pt12) - [E_total(Pt12_pure) + E_total(M_atom)]
        binding_energy = e_doped - (e_pure + e_atom)
        
        # Step 5: Print results
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {metal_symbol}@Pt12")
        print(f"{'='*60}")
        print(f"E_total({metal_symbol}@Pt12):   {e_doped:.4f} eV")
        print(f"E_total(Pt12_pure):            {e_pure:.4f} eV")
        print(f"E_total({metal_symbol}_atom):   {e_atom:.4f} eV")
        print(f"Binding Energy (Eb):           {binding_energy:.4f} eV")
        print(f"Stability:                     {'Stable' if binding_energy < 0 else 'Unstable'}")
        print(f"{'='*60}\n")
        
        # Save structures
        write(f"{self.work_dir}/Pt12_pure_optimized.xyz", optimized_pt12)
        write(f"{self.work_dir}/{metal_symbol}Pt12_optimized.xyz", optimized_doped)
        
        return {
            'metal': metal_symbol,
            'e_doped': e_doped,
            'e_pure': e_pure,
            'e_atom': e_atom,
            'binding_energy': binding_energy,
            'optimized_doped': optimized_doped,
            'optimized_pure': optimized_pt12,
        }
    
    def analyze_bond_lengths(self, doped_cluster, metal_symbol):
        """
        Analyze M-Pt bond lengths in the optimized cluster.
        
        Parameters:
        -----------
        doped_cluster : Atoms
            Optimized M@Pt12 cluster
        metal_symbol : str
            Metal symbol ('Li', 'Na', or 'K')
        
        Returns:
        --------
        dict with bond length statistics
        """
        positions = doped_cluster.get_positions()
        symbols = doped_cluster.get_chemical_symbols()
        
        # Find metal atom index
        metal_indices = [i for i, sym in enumerate(symbols) if sym == metal_symbol]
        if not metal_indices:
            print(f"Warning: No {metal_symbol} atom found in cluster")
            return {}
        
        metal_idx = metal_indices[0]
        metal_pos = positions[metal_idx]
        
        # Calculate distances to all Pt atoms
        pt_indices = [i for i, sym in enumerate(symbols) if sym == 'Pt']
        distances = []
        
        for pt_idx in pt_indices:
            pt_pos = positions[pt_idx]
            dist = np.linalg.norm(metal_pos - pt_pos)
            distances.append(dist)
        
        # Calculate statistics
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        std_dist = np.std(distances)
        
        print(f"\n{metal_symbol}-Pt bond lengths:")
        print(f"  Average: {avg_dist:.3f} Å")
        print(f"  Minimum: {min_dist:.3f} Å")
        print(f"  Maximum: {max_dist:.3f} Å")
        print(f"  Std Dev: {std_dist:.3f} Å")
        
        return {
            'average': avg_dist,
            'min': min_dist,
            'max': max_dist,
            'std': std_dist,
            'all_distances': distances
        }
    
    def run_full_analysis(self):
        """
        Run complete binding energy analysis for all three metals.
        
        Returns:
        --------
        dict: Results for all clusters
        """
        metals = ['Li', 'Na', 'K']
        results = {}
        
        print("="*60)
        print("BINDING ENERGY ANALYSIS FOR M@Pt12 CLUSTERS")
        print("M = Li, Na, K")
        print("="*60)
        
        for metal in metals:
            result = self.calculate_binding_energy(metal)
            results[metal] = result
            
            # Analyze bond lengths of optimized cluster
            if result['optimized_doped'] is not None:
                self.analyze_bond_lengths(result['optimized_doped'], metal)
            
            print("\n" + "-"*60 + "\n")
        
        # Summary table
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """
        Print a summary table of all results.
        
        Parameters:
        -----------
        results : dict
            Results from run_full_analysis
        """
        print("\n" + "="*70)
        print("SUMMARY OF BINDING ENERGY RESULTS")
        print("="*70)
        print(f"{'Cluster':<15} {'E_doped (eV)':<15} {'E_pure (eV)':<15} {'E_atom (eV)':<15} {'Eb (eV)':<15} {'Stable':<10}")
        print("-"*70)
        
        for metal, data in results.items():
            cluster_name = f"{metal}@Pt12"
            stable = "Yes" if data['binding_energy'] < 0 else "No"
            print(f"{cluster_name:<15} {data['e_doped']:<15.4f} {data['e_pure']:<15.4f} "
                  f"{data['e_atom']:<15.4f} {data['binding_energy']:<15.4f} {stable:<10}")
        
        print("="*70)
        print("\nNote: Negative binding energy indicates thermodynamic stability.")
        print("      More negative = more stable cluster.")


def main():
    """
    Main function to run the binding energy calculation.
    """
    # Initialize the calculator
    calculator = BindingEnergyCalculator(work_dir="./dmol3_results")
    
    # Run the full analysis
    results = calculator.run_full_analysis()
    
    # Additional analysis: Compare stability ordering
    print("\n" + "="*70)
    print("STABILITY ORDERING")
    print("="*70)
    
    # Sort by binding energy (more negative = more stable)
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['binding_energy']
    )
    
    print("\nStability ranking (most stable first):")
    for i, (metal, data) in enumerate(sorted_results, 1):
        print(f"  {i}. {metal}@Pt12: Eb = {data['binding_energy']:.4f} eV")
    
    print("\n" + "="*70)
    print("Analysis complete! Check the dmol3_results directory for output files.")
    print("="*70)


if __name__ == "__main__":
    main()