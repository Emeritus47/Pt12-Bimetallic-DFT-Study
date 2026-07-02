"""
DFT Cluster Optimization Script using ASE with Dmol3
For optimizing M@Pt12 bimetallic clusters (M = Li, Na, K)
"""

import numpy as np
from ase import Atoms
from ase.io import write, read
from ase.optimize import BFGS
from ase.calculators.dmol3 import Dmol3
from ase.constraints import FixSymmetry
import os
import time

# ============================================================================
# PARAMETERS AND SETUP
# ============================================================================

# Dmol3 calculation parameters
DMOL3_PARAMS = {
    'task': 'GeometryOptimization',
    'functional': 'pbe',  # Perdew-Burke-Ernzerhof GGA
    'basis': 'dnp',       # Double numerical with polarization
    'pseudo': 'dspp',     # Density functional semi-core pseudopotential
    'cutoff': 4.3,        # Global orbital cutoff in Angstroms
    'scf_tolerance': 1e-4,  # SCF convergence in Hartree
    'max_scf_cycles': 200,
    'smearing': 0.005,    # Thermal smearing for SCF convergence
    'spin_polarized': True,
    'charge': 0,
    'multiplicity': 2,    # For open-shell systems
}

# Convergence criteria for geometry optimization
OPTIMIZATION_PARAMS = {
    'energy_criterion': 1e-5,    # Ha
    'force_criterion': 0.02,     # Ha/Angstrom
    'max_displacement': 0.05,    # Angstrom
    'max_steps': 1000,
}

# ============================================================================
# CLUSTER GENERATION FUNCTIONS
# ============================================================================

def create_icosahedral_pt12():
    """
    Create a Pt12 cluster with icosahedral-like geometry (missing two atoms)
    Returns: Atoms object
    """
    # Icosahedral vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Standard icosahedron vertices
    vertices = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    
    # Scale to get reasonable Pt-Pt bond length (~2.8 Angstroms)
    scale_factor = 2.8 / 2  # Normalize to get ~2.8 A bonds
    vertices = np.array(vertices) * scale_factor
    
    # Create Pt12 cluster
    symbols = ['Pt'] * 12
    positions = vertices
    
    cluster = Atoms(symbols=symbols, positions=positions)
    cluster.center(vacuum=10.0)  # Add vacuum for isolation
    
    return cluster

def add_alkali_metal(cluster, metal_symbol='Li'):
    """
    Add an alkali metal atom at the center of the Pt12 cage
    Args:
        cluster: Pt12 Atoms object
        metal_symbol: 'Li', 'Na', or 'K'
    Returns:
        M@Pt12 Atoms object
    """
    # Get center of mass of Pt12 cluster
    center = cluster.get_center_of_mass()
    
    # Create new atoms with metal at center
    symbols = list(cluster.get_chemical_symbols()) + [metal_symbol]
    positions = list(cluster.get_positions()) + [center]
    
    new_cluster = Atoms(symbols=symbols, positions=positions)
    new_cluster.center(vacuum=10.0)
    
    return new_cluster

def generate_all_clusters():
    """
    Generate pure Pt12 and M@Pt12 clusters
    Returns: dict of Atoms objects
    """
    clusters = {}
    
    # Pure Pt12
    print("Generating Pt12 cluster...")
    clusters['Pt12'] = create_icosahedral_pt12()
    
    # Doped clusters
    for metal in ['Li', 'Na', 'K']:
        print(f"Generating {metal}@Pt12 cluster...")
        pt12 = create_icosahedral_pt12()
        clusters[f'{metal}@Pt12'] = add_alkali_metal(pt12, metal)
    
    return clusters

# ============================================================================
# CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(work_dir='./dmol3_work'):
    """
    Set up Dmol3 calculator with appropriate parameters
    Args:
        work_dir: Working directory for Dmol3 files
    Returns:
        Dmol3 calculator object
    """
    # Create working directory
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # Initialize Dmol3 calculator
    calc = Dmol3(
        label=os.path.join(work_dir, 'dmol3'),
        **DMOL3_PARAMS
    )
    
    return calc

# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================

def optimize_cluster(cluster, metal_name, work_dir='./dmol3_work'):
    """
    Optimize the geometry of a cluster using Dmol3
    Args:
        cluster: Atoms object
        metal_name: String identifier for the cluster
        work_dir: Working directory
    Returns:
        Optimized Atoms object
    """
    print(f"\n{'='*60}")
    print(f"Optimizing {metal_name} cluster")
    print(f"{'='*60}")
    
    # Set up calculator
    calc_dir = os.path.join(work_dir, metal_name)
    calc = setup_dmol3_calculator(calc_dir)
    
    # Assign calculator to cluster
    cluster.calc = calc
    
    # Set up geometry optimization
    optimizer = BFGS(
        cluster,
        trajectory=f'{metal_name}_opt.traj',
        logfile=f'{metal_name}_opt.log'
    )
    
    # Run optimization
    print(f"Starting optimization of {metal_name}...")
    start_time = time.time()
    
    try:
        optimizer.run(
            fmax=OPTIMIZATION_PARAMS['force_criterion'],
            steps=OPTIMIZATION_PARAMS['max_steps']
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Saving current structure...")
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Save optimized structure
    write(f'{metal_name}_optimized.xyz', cluster)
    write(f'{metal_name}_optimized.traj', cluster)
    
    # Get final energy
    try:
        final_energy = cluster.get_potential_energy()
        print(f"Final energy of {metal_name}: {final_energy:.6f} Ha")
    except:
        print("Could not retrieve final energy")
    
    return cluster

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_optimized_cluster(cluster, metal_name):
    """
    Perform basic analysis on optimized cluster
    Args:
        cluster: Optimized Atoms object
        metal_name: String identifier
    """
    print(f"\n{'='*60}")
    print(f"Analysis for {metal_name}")
    print(f"{'='*60}")
    
    # Get positions
    positions = cluster.get_positions()
    symbols = cluster.get_chemical_symbols()
    
    # Find metal atom (last atom in our construction)
    metal_index = len(symbols) - 1
    metal_position = positions[metal_index]
    pt_positions = positions[:metal_index]
    
    # Calculate average M-Pt bond length
    distances = []
    for pt_pos in pt_positions:
        dist = np.linalg.norm(metal_position - pt_pos)
        distances.append(dist)
    
    avg_m_pt = np.mean(distances)
    std_m_pt = np.std(distances)
    
    print(f"Number of atoms: {len(cluster)}")
    print(f"Metal atom: {symbols[metal_index]}")
    print(f"Average M-Pt bond length: {avg_m_pt:.3f} ± {std_m_pt:.3f} Angstroms")
    print(f"Min M-Pt bond length: {np.min(distances):.3f} Angstroms")
    print(f"Max M-Pt bond length: {np.max(distances):.3f} Angstroms")
    
    # Calculate average Pt-Pt bond length
    pt_distances = []
    for i in range(len(pt_positions)):
        for j in range(i+1, len(pt_positions)):
            dist = np.linalg.norm(pt_positions[i] - pt_positions[j])
            pt_distances.append(dist)
    
    avg_pt_pt = np.mean(pt_distances)
    std_pt_pt = np.std(pt_distances)
    
    print(f"Average Pt-Pt bond length: {avg_pt_pt:.3f} ± {std_pt_pt:.3f} Angstroms")
    
    # Calculate cluster diameter
    com = cluster.get_center_of_mass()
    max_dist = 0
    for pos in positions:
        dist = np.linalg.norm(pos - com)
        if dist > max_dist:
            max_dist = dist
    
    print(f"Cluster radius: {max_dist:.3f} Angstroms")
    print(f"Cluster diameter: {2*max_dist:.3f} Angstroms")
    
    # Save analysis data
    analysis_data = {
        'metal_name': metal_name,
        'num_atoms': len(cluster),
        'avg_m_pt': avg_m_pt,
        'std_m_pt': std_m_pt,
        'avg_pt_pt': avg_pt_pt,
        'std_pt_pt': std_pt_pt,
        'cluster_radius': max_dist,
    }
    
    return analysis_data

def calculate_binding_energy(doped_cluster, pure_pt12, metal_atom, metal_name):
    """
    Calculate binding energy for doped cluster
    Eb = E_total(M@Pt12) - [E(Pt12) + E(M)]
    Args:
        doped_cluster: Optimized M@Pt12 cluster
        pure_pt12: Optimized pure Pt12 cluster
        metal_atom: Optimized isolated metal atom
        metal_name: String identifier
    Returns:
        Binding energy in eV
    """
    try:
        e_doped = doped_cluster.get_potential_energy()
        e_pt12 = pure_pt12.get_potential_energy()
        e_metal = metal_atom.get_potential_energy()
        
        binding_energy = e_doped - (e_pt12 + e_metal)
        binding_energy_ev = binding_energy * 27.2114  # Hartree to eV
        
        print(f"\nBinding Energy for {metal_name}@Pt12:")
        print(f"E(M@Pt12) = {e_doped:.6f} Ha")
        print(f"E(Pt12) = {e_pt12:.6f} Ha")
        print(f"E(M) = {e_metal:.6f} Ha")
        print(f"Eb = {binding_energy:.6f} Ha = {binding_energy_ev:.3f} eV")
        
        if binding_energy_ev < 0:
            print(f"{metal_name}@Pt12 is thermodynamically stable (exothermic)")
        else:
            print(f"{metal_name}@Pt12 is not thermodynamically stable (endothermic)")
        
        return binding_energy_ev
    
    except Exception as e:
        print(f"Could not calculate binding energy: {e}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run DFT optimization of M@Pt12 clusters
    """
    print("="*60)
    print("DFT CLUSTER OPTIMIZATION USING DMOL3")
    print("M@Pt12 Bimetallic Clusters (M = Li, Na, K)")
    print("="*60)
    
    # Create working directory
    work_dir = './dmol3_optimization'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # Generate clusters
    print("\nGenerating cluster geometries...")
    clusters = generate_all_clusters()
    
    # Save initial structures
    for name, cluster in clusters.items():
        write(f'initial_{name}.xyz', cluster)
    
    # Optimize each cluster
    optimized_clusters = {}
    for name, cluster in clusters.items():
        optimized = optimize_cluster(
            cluster,
            name,
            work_dir=work_dir
        )
        optimized_clusters[name] = optimized
    
    # Analyze optimized clusters
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    all_analysis = {}
    for name, cluster in optimized_clusters.items():
        analysis = analyze_optimized_cluster(cluster, name)
        all_analysis[name] = analysis
    
    # Save summary
    summary_file = os.path.join(work_dir, 'optimization_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DFT OPTIMIZATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"DMol3 Parameters:\n")
        for key, value in DMOL3_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nOptimization Parameters:\n")
        for key, value in OPTIMIZATION_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nCluster Analysis:\n")
        f.write("-"*60 + "\n")
        for name, analysis in all_analysis.items():
            f.write(f"\n{name}:\n")
            for key, value in analysis.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Note about binding energy calculation
    print("\n" + "="*60)
    print("ADDITIONAL CALCULATIONS NEEDED")
    print("="*60)
    print("To calculate binding energies, you need to:")
    print("1. Optimize isolated Pt12 cluster (already done)")
    print("2. Optimize isolated Li, Na, and K atoms")
    print("3. Run binding energy calculation")
    print("\nTo optimize isolated atoms, use the following script:")
    print("  atom = Atoms([metal_symbol], positions=[[0,0,0]])")
    print("  atom.center(vacuum=10.0)")
    print("  # Set up calculator and optimize")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()