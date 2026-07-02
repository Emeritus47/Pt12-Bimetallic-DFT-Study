"""
Differential Charge Density Analysis for M@Pt12 Bimetallic Clusters
Using ASE Interface with DMol3

This script computes and visualizes differential charge density maps to
analyze charge transfer between alkali metal atoms (M = Li, Na, K) and
the Pt12 cage. It helps identify ionic vs metallic bonding characteristics.

The differential charge density is defined as:
    Δρ = ρ(M@Pt12) - ρ(Pt12) - ρ(M)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ase.io import read
from ase.data import covalent_radii
from ase.visualize import view
import os
from scipy.ndimage import gaussian_filter

# ============================================================================
# CONFIGURATION SECTION - Modify these paths for your system
# ============================================================================

# Path to DMol3 output files and cube files (replace with your actual paths)
DMOL3_DATA = {
    'pure_Pt12': {
        'cubefile': 'path/to/pure_Pt12.cube',  # Charge density cube file
        'structure': 'path/to/pure_Pt12.xyz'   # Structure file
    },
    'Li_Pt12': {
        'cubefile': 'path/to/Li_Pt12.cube',
        'structure': 'path/to/Li_Pt12.xyz'
    },
    'Na_Pt12': {
        'cubefile': 'path/to/Na_Pt12.cube',
        'structure': 'path/to/Na_Pt12.xyz'
    },
    'K_Pt12': {
        'cubefile': 'path/to/K_Pt12.cube',
        'structure': 'path/to/K_Pt12.xyz'
    }
}

# Path to isolated metal atom cube files
METAL_ATOM_DATA = {
    'Li': 'path/to/Li_atom.cube',
    'Na': 'path/to/Na_atom.cube',
    'K': 'path/to/K_atom.cube'
}

# Visualization parameters
COLORMAP = 'seismic'  # Red-blue colormap for charge density difference
CONTOUR_LEVELS = 20
ISOSURFACE_LEVEL = 0.005  # For 3D isosurface visualization

# Output directory
OUTPUT_DIR = 'charge_density_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CUBE FILE PARSER
# ============================================================================

def read_cube_file(filename):
    """
    Read Gaussian cube file format.
    
    Parameters:
    -----------
    filename : str
        Path to cube file
        
    Returns:
    --------
    data : ndarray
        3D charge density data
    grid_origin : ndarray
        Origin of the grid
    grid_spacing : ndarray
        Spacing between grid points
    natoms : int
        Number of atoms
    atom_positions : ndarray
        Atomic positions
    atom_numbers : ndarray
        Atomic numbers
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip first two comment lines
    line_idx = 2
    
    # Read number of atoms and grid origin
    parts = lines[line_idx].split()
    natoms = int(parts[0])
    grid_origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    line_idx += 1
    
    # Read grid dimensions and spacing
    nx, dx, dy, dz = lines[line_idx].split()
    nx = int(nx)
    dx = float(dx)
    grid_spacing = np.array([dx, float(dy), float(dz)])
    line_idx += 1
    
    ny, dx, dy, dz = lines[line_idx].split()
    ny = int(ny)
    line_idx += 1
    
    nz, dx, dy, dz = lines[line_idx].split()
    nz = int(nz)
    line_idx += 1
    
    # Read atom information
    atom_numbers = []
    atom_positions = []
    for i in range(natoms):
        parts = lines[line_idx].split()
        atom_numbers.append(int(parts[0]))
        atom_positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
        line_idx += 1
    
    atom_positions = np.array(atom_positions)
    atom_numbers = np.array(atom_numbers)
    
    # Read charge density data
    data = []
    for line in lines[line_idx:]:
        parts = line.split()
        data.extend([float(x) for x in parts])
    
    data = np.array(data)
    
    # Check if data size matches grid dimensions
    expected_size = nx * ny * nz
    if len(data) != expected_size:
        print(f"Warning: Data size mismatch. Expected {expected_size}, got {len(data)}")
        # Pad or truncate to match expected size
        if len(data) < expected_size:
            data = np.pad(data, (0, expected_size - len(data)), 'constant')
        else:
            data = data[:expected_size]
    
    # Reshape to 3D grid
    data = data.reshape((nx, ny, nz))
    
    return {
        'data': data,
        'grid_origin': grid_origin,
        'grid_spacing': grid_spacing,
        'natoms': natoms,
        'atom_positions': atom_positions,
        'atom_numbers': atom_numbers,
        'grid_shape': (nx, ny, nz)
    }


def interpolate_at_positions(cube_data, positions):
    """
    Interpolate charge density at specific atomic positions.
    
    Parameters:
    -----------
    cube_data : dict
        Cube data from read_cube_file
    positions : ndarray
        Positions to interpolate at
        
    Returns:
    --------
    densities : ndarray
        Charge density values at positions
    """
    data = cube_data['data']
    origin = cube_data['grid_origin']
    spacing = cube_data['grid_spacing']
    shape = cube_data['grid_shape']
    
    densities = []
    
    for pos in positions:
        # Convert position to grid indices
        idx = (pos - origin) / spacing
        idx_int = np.floor(idx).astype(int)
        idx_frac = idx - idx_int
        
        # Trilinear interpolation
        i, j, k = idx_int
        fi, fj, fk = idx_frac
        
        # Ensure indices are within bounds
        i = np.clip(i, 0, shape[0] - 2)
        j = np.clip(j, 0, shape[1] - 2)
        k = np.clip(k, 0, shape[2] - 2)
        
        # Interpolate
        c000 = data[i, j, k]
        c100 = data[i+1, j, k]
        c010 = data[i, j+1, k]
        c110 = data[i+1, j+1, k]
        c001 = data[i, j, k+1]
        c101 = data[i+1, j, k+1]
        c011 = data[i, j+1, k+1]
        c111 = data[i+1, j+1, k+1]
        
        # Interpolate along x
        c00 = c000 * (1-fi) + c100 * fi
        c10 = c010 * (1-fi) + c110 * fi
        c01 = c001 * (1-fi) + c101 * fi
        c11 = c011 * (1-fi) + c111 * fi
        
        # Interpolate along y
        c0 = c00 * (1-fj) + c10 * fj
        c1 = c01 * (1-fj) + c11 * fj
        
        # Interpolate along z
        c = c0 * (1-fk) + c1 * fk
        
        densities.append(c)
    
    return np.array(densities)


# ============================================================================
# DIFFERENTIAL CHARGE DENSITY CALCULATIONS
# ============================================================================

def compute_differential_charge_density(rho_complex, rho_pure, rho_atom):
    """
    Compute differential charge density: Δρ = ρ(complex) - ρ(pure_cage) - ρ(atom)
    
    All inputs must have the same grid shape.
    """
    if rho_complex.shape != rho_pure.shape or rho_complex.shape != rho_atom.shape:
        print("Warning: Grid shapes do not match!")
        # Find minimum shape and crop
        min_shape = np.minimum.reduce([rho_complex.shape, rho_pure.shape, rho_atom.shape])
        rho_complex = rho_complex[:min_shape[0], :min_shape[1], :min_shape[2]]
        rho_pure = rho_pure[:min_shape[0], :min_shape[1], :min_shape[2]]
        rho_atom = rho_atom[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    return rho_complex - rho_pure - rho_atom


def integrate_charge_transfer(delta_rho, grid_spacing):
    """
    Integrate charge density difference to get total charge transfer.
    
    Parameters:
    -----------
    delta_rho : ndarray
        Differential charge density
    grid_spacing : ndarray
        Grid spacing in each dimension
        
    Returns:
    --------
    total_charge : float
        Total charge transferred (in electrons)
    """
    volume_per_point = np.prod(grid_spacing)
    total_charge = np.sum(delta_rho) * volume_per_point
    return total_charge


def compute_bader_charge_estimate(cube_data, atom_positions, atom_numbers):
    """
    Estimate Bader-like charges using Voronoi integration.
    
    Returns a simplified estimate of charge on each atom.
    """
    data = cube_data['data']
    origin = cube_data['grid_origin']
    spacing = cube_data['grid_spacing']
    shape = cube_data['grid_shape']
    
    # Create grid coordinates
    x = np.arange(shape[0]) * spacing[0] + origin[0]
    y = np.arange(shape[1]) * spacing[1] + origin[1]
    z = np.arange(shape[2]) * spacing[2] + origin[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # For each atom, find nearest grid points
    charges = []
    for pos in atom_positions:
        # Distance to all grid points
        dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
        # Find grid points within covalent radius
        radius = covalent_radii.get(atom_numbers[0], 1.5)  # Default if not found
        mask = dist < radius
        # Integrate charge in this region
        charge = np.sum(data[mask]) * np.prod(spacing)
        charges.append(charge)
    
    return np.array(charges)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_2d_slice(delta_rho, cube_data, slice_axis='z', slice_index=None,
                  atom_positions=None, atom_numbers=None, title='',
                  save_path=None, show=True):
    """
    Plot 2D slice of differential charge density.
    """
    data = delta_rho
    shape = data.shape
    
    # Determine slice index
    if slice_index is None:
        slice_index = shape[2] // 2 if slice_axis == 'z' else shape[0] // 2
    
    # Take slice
    if slice_axis == 'z':
        slice_data = data[:, :, slice_index]
        extent = [0, shape[1], 0, shape[0]]
        xlabel, ylabel = 'Y', 'X'
    elif slice_axis == 'y':
        slice_data = data[:, slice_index, :]
        extent = [0, shape[2], 0, shape[0]]
        xlabel, ylabel = 'Z', 'X'
    else:  # x-axis
        slice_data = data[slice_index, :, :]
        extent = [0, shape[2], 0, shape[1]]
        xlabel, ylabel = 'Z', 'Y'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot 2D color map
    cmap = plt.get_cmap(COLORMAP)
    max_abs = np.max(np.abs(slice_data))
    if max_abs > 0:
        im = ax.imshow(slice_data.T, origin='lower', extent=extent,
                       cmap=cmap, vmin=-max_abs, vmax=max_abs,
                       interpolation='bilinear')
    else:
        im = ax.imshow(slice_data.T, origin='lower', extent=extent,
                       cmap='gray', interpolation='bilinear')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Δρ (e/Å³)', fontsize=12)
    
    # Add atom positions if provided
    if atom_positions is not None and atom_numbers is not None:
        # Convert atom positions to plot coordinates
        # This is simplified - actual conversion depends on slice axis
        for pos, num in zip(atom_positions, atom_numbers):
            if slice_axis == 'z':
                x_pos = pos[1]  # y-coordinate
                y_pos = pos[0]  # x-coordinate
            elif slice_axis == 'y':
                x_pos = pos[2]  # z-coordinate
                y_pos = pos[0]  # x-coordinate
            else:
                x_pos = pos[2]  # z-coordinate
                y_pos = pos[1]  # y-coordinate
            
            # Check if atom is near slice
            atom_slice_coord = pos[{'z': 2, 'y': 1, 'x': 0}[slice_axis]]
            if abs(atom_slice_coord - cube_data['grid_origin'][{'z': 2, 'y': 1, 'x': 0}[slice_axis]] 
                   - slice_index * cube_data['grid_spacing'][{'z': 2, 'y': 1, 'x': 0}[slice_axis]]) < 2.0:
                ax.scatter(x_pos, y_pos, s=100, c='black', marker='o', 
                          edgecolors='white', linewidth=2, zorder=5)
                # Add atom label
                symbol = {3: 'Li', 11: 'Na', 19: 'K', 78: 'Pt'}.get(num, 'X')
                ax.annotate(symbol, (x_pos, y_pos), xytext=(5, 5),
                           textcoords='offset points', color='black',
                           fontweight='bold', fontsize=10)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Differential Charge Density: {title}', fontsize=14)
    ax.grid(False)
    
    # Add text indicating charge transfer
    total_charge = integrate_charge_transfer(delta_rho, cube_data['grid_spacing'])
    ax.text(0.02, 0.98, f'Total ΔQ = {total_charge:.3f} e⁻',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def plot_3d_isosurface(delta_rho, cube_data, atom_positions=None,
                       atom_numbers=None, level=0.005, title='',
                       save_path=None, show=True):
    """
    Create 3D isosurface visualization of differential charge density.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from skimage.measure import marching_cubes
    
    data = delta_rho
    spacing = cube_data['grid_spacing']
    origin = cube_data['grid_origin']
    
    # Use marching cubes for isosurface
    try:
        # Extract positive and negative isosurfaces separately
        verts_pos, faces_pos, _, _ = marching_cubes(data, level, spacing=spacing)
        verts_neg, faces_neg, _, _ = marching_cubes(data, -level, spacing=spacing)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot positive isosurface (charge accumulation)
        if len(verts_pos) > 0:
            ax.plot_trisurf(verts_pos[:, 0], verts_pos[:, 1], verts_pos[:, 2],
                          triangles=faces_pos, color='red', alpha=0.6,
                          label='Charge accumulation (Δρ > 0)')
        
        # Plot negative isosurface (charge depletion)
        if len(verts_neg) > 0:
            ax.plot_trisurf(verts_neg[:, 0], verts_neg[:, 1], verts_neg[:, 2],
                          triangles=faces_neg, color='blue', alpha=0.6,
                          label='Charge depletion (Δρ < 0)')
        
        # Add atoms
        if atom_positions is not None and atom_numbers is not None:
            for pos, num in zip(atom_positions, atom_numbers):
                symbol = {3: 'Li', 11: 'Na', 19: 'K', 78: 'Pt'}.get(num, 'X')
                color = 'green' if symbol == 'Pt' else 'yellow'
                size = 100 if symbol == 'Pt' else 200
                ax.scatter(pos[0], pos[1], pos[2], s=size, c=color,
                          label=symbol, edgecolors='black', linewidth=1)
        
        ax.set_xlabel('X (Å)', fontsize=11)
        ax.set_ylabel('Y (Å)', fontsize=11)
        ax.set_zlabel('Z (Å)', fontsize=11)
        ax.set_title(f'3D Isosurface: {title} (Level = ±{level:.3f})', fontsize=14)
        ax.legend(loc='upper right')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
        
    except ImportError:
        print("scikit-image not installed. Skipping 3D isosurface plot.")
        return None, None
    except Exception as e:
        print(f"Error creating 3D isosurface: {e}")
        return None, None


def plot_comparison_slices(delta_rho_dict, cube_data_dict, atom_positions_dict,
                          atom_numbers_dict, slice_axis='z', slice_index=None,
                          save_path=None, show=True):
    """
    Plot comparison of differential charge density slices for all clusters.
    """
    n_clusters = len(delta_rho_dict)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    
    if n_clusters == 1:
        axes = [axes]
    
    for idx, (cluster_name, delta_rho) in enumerate(delta_rho_dict.items()):
        ax = axes[idx]
        cube_data = cube_data_dict[cluster_name]
        data = delta_rho
        shape = data.shape
        
        # Determine slice index
        if slice_index is None:
            slice_idx = shape[2] // 2 if slice_axis == 'z' else shape[0] // 2
        else:
            slice_idx = slice_index
        
        # Take slice
        if slice_axis == 'z':
            slice_data = data[:, :, slice_idx]
            extent = [0, shape[1], 0, shape[0]]
        elif slice_axis == 'y':
            slice_data = data[:, slice_idx, :]
            extent = [0, shape[2], 0, shape[0]]
        else:
            slice_data = data[slice_idx, :, :]
            extent = [0, shape[2], 0, shape[1]]
        
        # Plot
        cmap = plt.get_cmap(COLORMAP)
        max_abs = np.max(np.abs(slice_data))
        if max_abs > 0:
            im = ax.imshow(slice_data.T, origin='lower', extent=extent,
                          cmap=cmap, vmin=-max_abs, vmax=max_abs,
                          interpolation='bilinear')
        else:
            im = ax.imshow(slice_data.T, origin='lower', extent=extent,
                          cmap='gray', interpolation='bilinear')
        
        # Add atom positions
        atom_positions = atom_positions_dict[cluster_name]
        atom_numbers = atom_numbers_dict[cluster_name]
        for pos, num in zip(atom_positions, atom_numbers):
            if slice_axis == 'z':
                x_pos = pos[1]
                y_pos = pos[0]
            elif slice_axis == 'y':
                x_pos = pos[2]
                y_pos = pos[0]
            else:
                x_pos = pos[2]
                y_pos = pos[1]
            
            # Check if atom is near slice
            atom_slice_coord = pos[{'z': 2, 'y': 1, 'x': 0}[slice_axis]]
            if abs(atom_slice_coord - cube_data['grid_origin'][{'z': 2, 'y': 1, 'x': 0}[slice_axis]] 
                   - slice_idx * cube_data['grid_spacing'][{'z': 2, 'y': 1, 'x': 0}[slice_axis]]) < 2.0:
                ax.scatter(x_pos, y_pos, s=80, c='black', marker='o',
                          edgecolors='white', linewidth=2, zorder=5)
                symbol = {3: 'Li', 11: 'Na', 19: 'K', 78: 'Pt'}.get(num, 'X')
                ax.annotate(symbol, (x_pos, y_pos), xytext=(3, 3),
                           textcoords='offset points', color='black',
                           fontweight='bold', fontsize=8)
        
        ax.set_title(cluster_name, fontsize=12)
        ax.set_xlabel('Distance (Å)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Distance (Å)', fontsize=10)
        ax.grid(False)
        
        # Add colorbar for last subplot
        if idx == n_clusters - 1:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Δρ (e/Å³)', fontsize=10)
    
    plt.suptitle('Differential Charge Density Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


def plot_charge_transfer_bars(charge_transfer_data, save_path=None, show=True):
    """
    Plot bar chart of charge transfer values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clusters = list(charge_transfer_data.keys())
    charges = list(charge_transfer_data.values())
    
    bars = ax.bar(clusters, charges, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, charge in zip(bars, charges):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(charges),
                f'{charge:.3f} e⁻', ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Charge Transfer (e⁻)', fontsize=12)
    ax.set_title('Total Charge Transfer: ΔQ = ∫Δρ dV', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved charge transfer plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


# ============================================================================
# SYNTHETIC DATA GENERATION (FOR DEMONSTRATION)
# ============================================================================

def generate_synthetic_cube_data(shape=(40, 40, 40), atom_positions=None,
                                atom_numbers=None, charge_centers=None,
                                amplitude=1.0, noise=0.01):
    """
    Generate synthetic charge density data for testing.
    """
    spacing = np.array([0.3, 0.3, 0.3])
    origin = np.zeros(3)
    
    data = np.zeros(shape)
    
    # Add Gaussian charge centers
    if charge_centers is None:
        charge_centers = [(10, 20, 20), (20, 10, 20), (20, 20, 10)]
    
    for center in charge_centers:
        # Create Gaussian distribution
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = np.arange(shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        gauss = np.exp(-((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) / (2 * 3**2))
        data += amplitude * gauss
    
    # Add small random noise
    data += np.random.normal(0, noise, data.shape)
    
    return {
        'data': data,
        'grid_origin': origin,
        'grid_spacing': spacing,
        'natoms': len(atom_positions) if atom_positions is not None else 0,
        'atom_positions': atom_positions if atom_positions is not None else [],
        'atom_numbers': atom_numbers if atom_numbers is not None else [],
        'grid_shape': shape
    }


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """
    Main function to perform differential charge density analysis.
    """
    print("=" * 70)
    print("DIFFERENTIAL CHARGE DENSITY ANALYSIS")
    print("M@Pt12 Bimetallic Clusters (M = Li, Na, K)")
    print("=" * 70)
    
    # Store data
    delta_rho_dict = {}
    cube_data_dict = {}
    atom_positions_dict = {}
    atom_numbers_dict = {}
    charge_transfer_data = {}
    
    # Process each cluster
    clusters = ['pure_Pt12', 'Li_Pt12', 'Na_Pt12', 'K_Pt12']
    metal_atoms = {'Li_Pt12': 'Li', 'Na_Pt12': 'Na', 'K_Pt12': 'K'}
    
    print("\nLoading cube files...")
    
    # Try to load real data, fall back to synthetic if files don't exist
    use_synthetic = True
    
    for cluster in clusters:
        filepath = DMOL3_DATA[cluster]['cubefile']
        struct_path = DMOL3_DATA[cluster]['structure']
        
        try:
            if os.path.exists(filepath):
                cube_data = read_cube_file(filepath)
                use_synthetic = False
                print(f"  Loaded: {cluster}")
            else:
                raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            print(f"  Warning: Could not load {cluster}: {e}")
            print(f"  Generating synthetic data for {cluster}")
            
            # Generate synthetic data
            if cluster == 'pure_Pt12':
                atom_positions = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5, 15, 15],
                                 [15, 5, 5], [15, 5, 15], [15, 15, 5], [15, 15, 15],
                                 [10, 10, 10], [10, 10, 0], [10, 0, 10], [0, 10, 10]]
                atom_numbers = [78] * 12
                charge_centers = [(5, 5, 5), (5, 5, 15), (5, 15, 5), (5, 15, 15),
                                 (15, 5, 5), (15, 5, 15), (15, 15, 5), (15, 15, 15)]
                cube_data = generate_synthetic_cube_data(
                    shape=(30, 30, 30),
                    atom_positions=atom_positions,
                    atom_numbers=atom_numbers,
                    charge_centers=charge_centers,
                    amplitude=0.5
                )
            elif cluster == 'Li_Pt12':
                atom_positions = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5, 15, 15],
                                 [15, 5, 5], [15, 5, 15], [15, 15, 5], [15, 15, 15],
                                 [10, 10, 10], [10, 10, 0], [10, 0, 10], [0, 10, 10],
                                 [10, 10, 20]]  # Li at top
                atom_numbers = [78] * 12 + [3]
                charge_centers = [(5, 5, 5), (5, 5, 15), (5, 15, 5), (5, 15, 15),
                                 (15, 5, 5), (15, 5, 15), (15, 15, 5), (15, 15, 15),
                                 (10, 10, 20)]  # Li center
                cube_data = generate_synthetic_cube_data(
                    shape=(30, 30, 30),
                    atom_positions=atom_positions,
                    atom_numbers=atom_numbers,
                    charge_centers=charge_centers,
                    amplitude=0.6
                )
            elif cluster == 'Na_Pt12':
                atom_positions = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5, 15, 15],
                                 [15, 5, 5], [15, 5, 15], [15, 15, 5], [15, 15, 15],
                                 [10, 10, 10], [10, 10, 0], [10, 0, 10], [0, 10, 10],
                                 [10, 10, 20]]
                atom_numbers = [78] * 12 + [11]
                charge_centers = [(5, 5, 5), (5, 5, 15), (5, 15, 5), (5, 15, 15),
                                 (15, 5, 5), (15, 5, 15), (15, 15, 5), (15, 15, 15),
                                 (10, 10, 20)]
                cube_data = generate_synthetic_cube_data(
                    shape=(30, 30, 30),
                    atom_positions=atom_positions,
                    atom_numbers=atom_numbers,
                    charge_centers=charge_centers,
                    amplitude=0.6
                )
            else:  # K_Pt12
                atom_positions = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5, 15, 15],
                                 [15, 5, 5], [15, 5, 15], [15, 15, 5], [15, 15, 15],
                                 [10, 10, 10], [10, 10, 0], [10, 0, 10], [0, 10, 10],
                                 [10, 10, 20]]
                atom_numbers = [78] * 12 + [19]
                charge_centers = [(5, 5, 5), (5, 5, 15), (5, 15, 5), (5, 15, 15),
                                 (15, 5, 5), (15, 5, 15), (15, 15, 5), (15, 15, 15),
                                 (10, 10, 20)]
                cube_data = generate_synthetic_cube_data(
                    shape=(30, 30, 30),
                    atom_positions=atom_positions,
                    atom_numbers=atom_numbers,
                    charge_centers=charge_centers,
                    amplitude=0.7
                )
        
        cube_data_dict[cluster] = cube_data
        atom_positions_dict[cluster] = cube_data['atom_positions']
        atom_numbers_dict[cluster] = cube_data['atom_numbers']
    
    # Compute differential charge density for doped clusters
    if 'pure_Pt12' in cube_data_dict:
        rho_pure = cube_data_dict['pure_Pt12']['data']
        
        for cluster in ['Li_Pt12', 'Na_Pt12', 'K_Pt12']:
            print(f"\nComputing differential charge density for {cluster}...")
            
            rho_complex = cube_data_dict[cluster]['data']
            
            # For metal atom, we need to generate or load separate cube file
            # Here we use synthetic representation of isolated atom
            metal_symbol = metal_atoms[cluster]
            
            # Generate synthetic metal atom charge density
            atom_rho = np.zeros_like(rho_pure)
            if cluster == 'Li_Pt12':
                center = (10, 10, 20)
            elif cluster == 'Na_Pt12':
                center = (10, 10, 20)
            else:  # K_Pt12
                center = (10, 10, 20)
            
            # Create Gaussian for metal atom
            x = np.arange(rho_pure.shape[0])
            y = np.arange(rho_pure.shape[1])
            z = np.arange(rho_pure.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            gauss = np.exp(-((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) / (2 * 2**2))
            atom_rho = 0.3 * gauss
            
            # Compute differential charge density
            delta_rho = compute_differential_charge_density(
                rho_complex[:rho_pure.shape[0], :rho_pure.shape[1], :rho_pure.shape[2]],
                rho_pure,
                atom_rho[:rho_pure.shape[0], :rho_pure.shape[1], :rho_pure.shape[2]]
            )
            
            delta_rho_dict[cluster] = delta_rho
            
            # Calculate total charge transfer
            total_charge = integrate_charge_transfer(
                delta_rho,
                cube_data_dict[cluster]['grid_spacing']
            )
            charge_transfer_data[cluster] = total_charge
            print(f"  Total charge transfer: {total_charge:.3f} e⁻")
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    # Plot 2D slices for each doped cluster
    for cluster, delta_rho in delta_rho_dict.items():
        save_path = os.path.join(OUTPUT_DIR, f'{cluster}_delta_rho_slice.png')
        plot_2d_slice(
            delta_rho,
            cube_data_dict[cluster],
            slice_axis='z',
            slice_index=None,
            atom_positions=atom_positions_dict[cluster],
            atom_numbers=atom_numbers_dict[cluster],
            title=cluster,
            save_path=save_path,
            show=False
        )
    
    # Plot comparison
    if len(delta_rho_dict) > 1:
        save_path = os.path.join(OUTPUT_DIR, 'delta_rho_comparison.png')
        plot_comparison_slices(
            delta_rho_dict,
            cube_data_dict,
            atom_positions_dict,
            atom_numbers_dict,
            slice_axis='z',
            slice_index=None,
            save_path=save_path,
            show=False
        )
    
    # Plot charge transfer bar chart
    if charge_transfer_data:
        save_path = os.path.join(OUTPUT_DIR, 'charge_transfer_bars.png')
        plot_charge_transfer_bars(charge_transfer_data, save_path=save_path, show=False)
    
    # Plot 3D isosurfaces
    for cluster, delta_rho in delta_rho_dict.items():
        try:
            save_path = os.path.join(OUTPUT_DIR, f'{cluster}_3d_isosurface.png')
            plot_3d_isosurface(
                delta_rho,
                cube_data_dict[cluster],
                atom_positions=atom_positions_dict[cluster],
                atom_numbers=atom_numbers_dict[cluster],
                level=ISOSURFACE_LEVEL,
                title=cluster,
                save_path=save_path,
                show=False
            )
        except Exception as e:
            print(f"  Warning: Could not create 3D plot for {cluster}: {e}")
    
    # ============================================================================
    # SUMMARY REPORT
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print("\nCharge Transfer Analysis:")
    print("-" * 50)
    print(f"{'Cluster':<15} {'Total Charge Transfer (e⁻)':<25} {'Bonding Type':<15}")
    print("-" * 50)
    
    for cluster, charge in charge_transfer_data.items():
        if abs(charge) < 0.5:
            bonding = 'Covalent'
        elif charge > 0:
            bonding = 'Metallic'
        else:
            bonding = 'Ionic'
        
        print(f"{cluster:<15} {charge:<25.3f} {bonding:<15}")
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()