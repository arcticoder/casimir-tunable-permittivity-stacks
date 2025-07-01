#!/usr/bin/env python3
"""
Frequency-Dependent Permittivity Control
=======================================

Advanced frequency-dependent permittivity optimization for tunable Casimir stacks.

Mathematical Foundation:
- Drude-Lorentz permittivity: ε(ω) = 1 - ωp²/(ω² + iγω) + Σⱼ fⱼωpⱼ²/(ωⱼ² - ω² - iγⱼω)
- Frequency-dependent optimization across 10-100 THz
- Monte Carlo uncertainty propagation
- Cross-domain correlation management

Author: GitHub Copilot
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.constants import c, epsilon_0, pi, hbar
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class FrequencyDependentPermittivityController:
    """
    Advanced frequency-dependent permittivity control system.
    
    Provides precise control over Re[ε(ω)] across 10-100 THz frequency range
    with target tolerance ΔRe[ε(ω)]/ε < 5%.
    """
    
    def __init__(self, frequency_range: Tuple[float, float] = (10e12, 100e12)):
        """
        Initialize frequency-dependent permittivity controller.
        
        Args:
            frequency_range: (min_freq, max_freq) in Hz
        """
        self.frequency_range = frequency_range
        self.material_database = self._initialize_material_database()
        self.dispersion_models = {}
        
        print(f"✅ FrequencyDependentPermittivityController initialized")
        print(f"   📊 Frequency range: {frequency_range[0]/1e12:.1f}-{frequency_range[1]/1e12:.1f} THz")
    
    def _initialize_material_database(self) -> Dict:
        """Initialize comprehensive material parameter database."""
        return {
            'gold': {
                'plasma_frequency': 1.36e16,    # rad/s
                'collision_rate': 1.45e14,     # rad/s
                'oscillators': [
                    {'strength': 0.76, 'omega_p': 2.3e15, 'omega_0': 2.4e15, 'gamma': 1.0e14},
                    {'strength': 0.024, 'omega_p': 2.8e15, 'omega_0': 2.9e15, 'gamma': 5.0e13}
                ],
                'epsilon_inf': 1.0
            },
            'silver': {
                'plasma_frequency': 1.38e16,
                'collision_rate': 2.73e13,
                'oscillators': [
                    {'strength': 0.845, 'omega_p': 6.5e15, 'omega_0': 6.7e15, 'gamma': 9.6e14}
                ],
                'epsilon_inf': 1.0
            },
            'aluminum': {
                'plasma_frequency': 2.24e16,
                'collision_rate': 1.22e14,
                'oscillators': [
                    {'strength': 0.523, 'omega_p': 1.5e15, 'omega_0': 1.6e15, 'gamma': 2.4e14}
                ],
                'epsilon_inf': 1.0
            },
            'silicon': {
                'plasma_frequency': 0,
                'collision_rate': 0,
                'oscillators': [
                    {'strength': 11.68, 'omega_p': 0, 'omega_0': 3.4e15, 'gamma': 1e13}
                ],
                'epsilon_inf': 1.0
            },
            'silicon_dioxide': {
                'plasma_frequency': 0,
                'collision_rate': 0,
                'oscillators': [
                    {'strength': 1.1, 'omega_p': 0, 'omega_0': 2.0e16, 'gamma': 1e14}
                ],
                'epsilon_inf': 2.1
            }
        }
    
    def calculate_drude_lorentz_permittivity(self, material: str, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate frequency-dependent permittivity using Drude-Lorentz model.
        
        Args:
            material: Material name from database
            frequencies: Frequency array [Hz]
        
        Returns:
            Complex permittivity array ε(ω)
        """
        if material not in self.material_database:
            raise ValueError(f"Material '{material}' not in database. Available: {list(self.material_database.keys())}")
        
        params = self.material_database[material]
        omega = 2 * np.pi * frequencies
        
        # Initialize with epsilon_infinity
        epsilon = np.full_like(omega, params['epsilon_inf'], dtype=complex)
        
        # Drude term: -ωp²/(ω² + iγω)
        if params['plasma_frequency'] > 0:
            omega_p = params['plasma_frequency']
            gamma = params['collision_rate']
            drude_term = -omega_p**2 / (omega**2 + 1j*gamma*omega)
            epsilon += drude_term
        
        # Lorentz oscillator terms: Σⱼ fⱼωpⱼ²/(ωⱼ² - ω² - iγⱼω)
        for osc in params['oscillators']:
            fj = osc['strength']
            omega_pj = osc['omega_p']
            omega_0j = osc['omega_0']
            gamma_j = osc['gamma']
            
            if omega_pj > 0:  # Finite oscillator strength
                lorentz_term = fj * omega_pj**2 / (omega_0j**2 - omega**2 - 1j*gamma_j*omega)
            else:  # Static contribution
                lorentz_term = fj / (1 + (omega/omega_0j)**2 + 1j*(omega*gamma_j)/(omega_0j**2))
            
            epsilon += lorentz_term
        
        return epsilon
    
    def optimize_frequency_dependent_permittivity(self, 
                                                target_permittivity: Union[float, np.ndarray, Callable],
                                                materials: List[str],
                                                layer_fractions: Optional[np.ndarray] = None,
                                                frequency_points: int = 1000,
                                                tolerance: float = 0.05) -> Dict:
        """
        Optimize material composition for target frequency-dependent permittivity.
        
        Args:
            target_permittivity: Target ε(ω) - constant, array, or function
            materials: List of available materials
            layer_fractions: Volume fractions for each material (optimized if None)
            frequency_points: Number of frequency points
            tolerance: Target tolerance ΔRe[ε(ω)]/ε
        
        Returns:
            Optimization results with achieved performance
        """
        print(f"🎯 OPTIMIZING FREQUENCY-DEPENDENT PERMITTIVITY")
        print(f"   Materials: {materials}")
        print(f"   Frequency points: {frequency_points}")
        print(f"   Target tolerance: ±{tolerance*100:.1f}%")
        
        # Generate frequency grid
        frequencies = np.linspace(self.frequency_range[0], self.frequency_range[1], frequency_points)
        
        # Process target permittivity
        if callable(target_permittivity):
            target_eps = target_permittivity(frequencies)
        elif isinstance(target_permittivity, (int, float)):
            target_eps = np.full(frequency_points, target_permittivity)
        else:
            target_eps = np.array(target_permittivity)
            if len(target_eps) != frequency_points:
                # Interpolate to match frequency grid
                freq_interp = np.linspace(0, 1, len(target_eps))
                freq_new = np.linspace(0, 1, frequency_points)
                target_eps = interp1d(freq_interp, target_eps, kind='cubic')(freq_new)
        
        # Calculate material permittivities
        material_permittivities = {}
        for material in materials:
            material_permittivities[material] = self.calculate_drude_lorentz_permittivity(material, frequencies)
        
        # Optimization setup
        n_materials = len(materials)
        
        def objective_function(fractions):
            # Normalize fractions to sum to 1
            fractions = np.abs(fractions)
            fractions = fractions / np.sum(fractions)
            
            # Calculate effective permittivity using effective medium theory
            effective_eps = self._calculate_effective_permittivity(
                material_permittivities, fractions, materials
            )
            
            # Relative error in real part
            relative_errors = np.abs(effective_eps.real - target_eps.real) / np.abs(target_eps.real)
            
            # Penalty for tolerance violations
            max_error = np.max(relative_errors)
            tolerance_penalty = max(0, max_error - tolerance) * 100
            
            return max_error + tolerance_penalty
        
        # Bounds: volume fractions between 0 and 1
        bounds = [(0.01, 0.99) for _ in range(n_materials)]
        
        # Constraint: fractions sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        
        # Initial guess
        initial_fractions = np.ones(n_materials) / n_materials
        
        # Optimization
        print("   🔄 Running optimization...")
        result = minimize(
            objective_function,
            initial_fractions,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_fractions = result.x
            optimal_fractions = optimal_fractions / np.sum(optimal_fractions)  # Normalize
            
            # Calculate achieved performance
            achieved_eps = self._calculate_effective_permittivity(
                material_permittivities, optimal_fractions, materials
            )
            
            achieved_errors = np.abs(achieved_eps.real - target_eps.real) / np.abs(target_eps.real)
            
            optimization_result = {
                'success': True,
                'optimal_fractions': dict(zip(materials, optimal_fractions)),
                'achieved_permittivity': achieved_eps,
                'target_permittivity': target_eps,
                'frequencies': frequencies,
                'max_relative_error': np.max(achieved_errors),
                'mean_relative_error': np.mean(achieved_errors),
                'std_relative_error': np.std(achieved_errors),
                'tolerance_compliance': np.sum(achieved_errors < tolerance) / len(achieved_errors),
                'optimization_result': result
            }
            
            print(f"   ✅ Optimization successful!")
            print(f"      Max error: {optimization_result['max_relative_error']*100:.2f}%")
            print(f"      Compliance: {optimization_result['tolerance_compliance']*100:.1f}%")
            
            return optimization_result
        
        else:
            print(f"   ❌ Optimization failed: {result.message}")
            return {'success': False, 'message': result.message}
    
    def _calculate_effective_permittivity(self, material_permittivities: Dict, 
                                        fractions: np.ndarray, materials: List[str]) -> np.ndarray:
        """
        Calculate effective permittivity using effective medium theory.
        
        Uses Maxwell-Garnett mixing rule for metal-dielectric composites.
        """
        # For simplicity, use volume-weighted average (Wiener bounds)
        # Real implementation would use Maxwell-Garnett or Bruggeman theory
        
        effective_eps = np.zeros_like(list(material_permittivities.values())[0])
        
        for i, material in enumerate(materials):
            effective_eps += fractions[i] * material_permittivities[material]
        
        return effective_eps
    
    def validate_frequency_control(self, optimization_result: Dict, tolerance: float = 0.05) -> Dict:
        """
        Validate frequency-dependent permittivity control performance.
        
        Args:
            optimization_result: Result from optimize_frequency_dependent_permittivity
            tolerance: Required tolerance
        
        Returns:
            Detailed validation results
        """
        if not optimization_result['success']:
            return {'validation_passed': False, 'message': 'No successful optimization to validate'}
        
        print(f"🔬 VALIDATING FREQUENCY CONTROL")
        
        achieved_eps = optimization_result['achieved_permittivity']
        target_eps = optimization_result['target_permittivity']
        frequencies = optimization_result['frequencies']
        
        # Detailed error analysis
        relative_errors = np.abs(achieved_eps.real - target_eps.real) / np.abs(target_eps.real)
        
        # Frequency band analysis
        freq_bands = self._analyze_frequency_bands(frequencies, relative_errors, tolerance)
        
        # Statistical metrics
        max_error = np.max(relative_errors)
        mean_error = np.mean(relative_errors)
        std_error = np.std(relative_errors)
        p95_error = np.percentile(relative_errors, 95)
        compliance_rate = np.sum(relative_errors < tolerance) / len(relative_errors)
        
        # Validation criteria
        validation_passed = (
            max_error < tolerance and
            p95_error < tolerance * 0.8 and  # 95% of points within 80% of tolerance
            compliance_rate > 0.95  # 95% compliance rate
        )
        
        validation_result = {
            'validation_passed': validation_passed,
            'max_relative_error': max_error,
            'mean_relative_error': mean_error,
            'std_relative_error': std_error,
            'p95_relative_error': p95_error,
            'compliance_rate': compliance_rate,
            'frequency_bands': freq_bands,
            'target_tolerance': tolerance,
            'frequency_range_THz': (frequencies[0]/1e12, frequencies[-1]/1e12)
        }
        
        # Print validation summary
        print(f"   📊 Validation Results:")
        print(f"      Max error: {max_error*100:.2f}%")
        print(f"      95th percentile: {p95_error*100:.2f}%")
        print(f"      Compliance rate: {compliance_rate*100:.1f}%")
        print(f"      Status: {'✅ PASS' if validation_passed else '❌ FAIL'}")
        
        return validation_result
    
    def _analyze_frequency_bands(self, frequencies: np.ndarray, relative_errors: np.ndarray,
                               tolerance: float) -> Dict:
        """Analyze performance across frequency bands."""
        # Divide into 10 bands across frequency range
        n_bands = 10
        freq_edges = np.linspace(frequencies[0], frequencies[-1], n_bands + 1)
        
        band_analysis = {}
        for i in range(n_bands):
            band_start, band_end = freq_edges[i], freq_edges[i+1]
            band_mask = (frequencies >= band_start) & (frequencies < band_end)
            
            if np.any(band_mask):
                band_errors = relative_errors[band_mask]
                band_analysis[f'band_{i+1}'] = {
                    'freq_range_THz': (band_start/1e12, band_end/1e12),
                    'max_error': np.max(band_errors),
                    'mean_error': np.mean(band_errors),
                    'compliance_rate': np.sum(band_errors < tolerance) / len(band_errors),
                    'n_points': len(band_errors)
                }
        
        return band_analysis
    
    def monte_carlo_uncertainty_propagation(self, optimization_result: Dict,
                                          parameter_uncertainties: Dict,
                                          n_samples: int = 10000) -> Dict:
        """
        Monte Carlo uncertainty propagation for manufacturing tolerances.
        
        Args:
            optimization_result: Optimization result to analyze
            parameter_uncertainties: Dict of parameter uncertainty levels
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Uncertainty propagation results
        """
        print(f"🎲 MONTE CARLO UNCERTAINTY PROPAGATION")
        print(f"   Samples: {n_samples}")
        
        if not optimization_result['success']:
            return {'success': False, 'message': 'No successful optimization for MC analysis'}
        
        # Extract base parameters
        base_fractions = list(optimization_result['optimal_fractions'].values())
        materials = list(optimization_result['optimal_fractions'].keys())
        frequencies = optimization_result['frequencies']
        target_eps = optimization_result['target_permittivity']
        
        # Default uncertainties if not provided
        if parameter_uncertainties is None:
            parameter_uncertainties = {
                'material_fraction': 0.02,  # 2% uncertainty in volume fractions
                'material_parameters': 0.05  # 5% uncertainty in material parameters
            }
        
        # Monte Carlo sampling
        mc_errors = []
        mc_permittivities = []
        
        for sample in range(n_samples):
            # Perturb volume fractions
            fraction_std = parameter_uncertainties.get('material_fraction', 0.02)
            perturbed_fractions = np.array(base_fractions) + np.random.normal(0, fraction_std, len(base_fractions))
            perturbed_fractions = np.abs(perturbed_fractions)  # Ensure positive
            perturbed_fractions = perturbed_fractions / np.sum(perturbed_fractions)  # Normalize
            
            # Perturb material parameters
            param_uncertainty = parameter_uncertainties.get('material_parameters', 0.05)
            perturbed_materials = {}
            
            for material in materials:
                # Create perturbed material parameters
                base_params = self.material_database[material].copy()
                
                # Perturb plasma frequency and collision rate
                if base_params['plasma_frequency'] > 0:
                    base_params['plasma_frequency'] *= (1 + np.random.normal(0, param_uncertainty))
                    base_params['collision_rate'] *= (1 + np.random.normal(0, param_uncertainty))
                
                # Perturb oscillator parameters
                for osc in base_params['oscillators']:
                    for key in ['strength', 'omega_p', 'omega_0', 'gamma']:
                        if osc[key] > 0:
                            osc[key] *= (1 + np.random.normal(0, param_uncertainty))
                
                # Store temporarily
                original_params = self.material_database[material]
                self.material_database[material] = base_params
                
                # Calculate perturbed permittivity
                perturbed_materials[material] = self.calculate_drude_lorentz_permittivity(material, frequencies)
                
                # Restore original parameters
                self.material_database[material] = original_params
            
            # Calculate effective permittivity with perturbations
            perturbed_eps = self._calculate_effective_permittivity(
                perturbed_materials, perturbed_fractions, materials
            )
            
            # Calculate errors
            sample_errors = np.abs(perturbed_eps.real - target_eps.real) / np.abs(target_eps.real)
            mc_errors.append(np.max(sample_errors))
            mc_permittivities.append(perturbed_eps)
            
            # Progress update
            if (sample + 1) % (n_samples // 10) == 0:
                progress = (sample + 1) / n_samples * 100
                print(f"   Progress: {progress:.0f}%")
        
        # Statistical analysis
        mc_errors = np.array(mc_errors)
        
        mc_results = {
            'success': True,
            'n_samples': n_samples,
            'mean_max_error': np.mean(mc_errors),
            'std_max_error': np.std(mc_errors),
            'p95_max_error': np.percentile(mc_errors, 95),
            'p99_max_error': np.percentile(mc_errors, 99),
            'worst_case_error': np.max(mc_errors),
            'parameter_uncertainties': parameter_uncertainties,
            'mc_error_distribution': mc_errors,
            'tolerance_compliance_rate': np.sum(mc_errors < 0.05) / len(mc_errors)  # 5% tolerance
        }
        
        print(f"   📊 MC Results:")
        print(f"      Mean max error: {mc_results['mean_max_error']*100:.2f}%")
        print(f"      95th percentile: {mc_results['p95_max_error']*100:.2f}%")
        print(f"      Tolerance compliance: {mc_results['tolerance_compliance_rate']*100:.1f}%")
        
        return mc_results
    
    def plot_frequency_response(self, optimization_result: Dict, save_path: Optional[str] = None):
        """
        Plot frequency-dependent permittivity response.
        
        Args:
            optimization_result: Result from optimization
            save_path: Optional path to save plot
        """
        if not optimization_result['success']:
            print("❌ Cannot plot - no successful optimization result")
            return
        
        frequencies = optimization_result['frequencies']
        achieved_eps = optimization_result['achieved_permittivity']
        target_eps = optimization_result['target_permittivity']
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Frequency in THz
        freq_THz = frequencies / 1e12
        
        # Plot 1: Real permittivity
        ax1.plot(freq_THz, achieved_eps.real, 'b-', linewidth=2, label='Achieved Re[ε(ω)]')
        ax1.plot(freq_THz, target_eps.real, 'r--', linewidth=2, label='Target Re[ε(ω)]')
        ax1.set_ylabel('Real Permittivity')
        ax1.set_title('Frequency-Dependent Permittivity Control')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Imaginary permittivity
        ax2.plot(freq_THz, achieved_eps.imag, 'g-', linewidth=2, label='Achieved Im[ε(ω)]')
        if hasattr(target_eps, 'imag'):
            ax2.plot(freq_THz, target_eps.imag, 'orange', linestyle='--', linewidth=2, label='Target Im[ε(ω)]')
        ax2.set_ylabel('Imaginary Permittivity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relative error
        relative_errors = np.abs(achieved_eps.real - target_eps.real) / np.abs(target_eps.real)
        ax3.plot(freq_THz, relative_errors * 100, 'purple', linewidth=2, label='Relative Error')
        ax3.axhline(5.0, color='red', linestyle='--', alpha=0.7, label='5% Tolerance')
        ax3.set_xlabel('Frequency (THz)')
        ax3.set_ylabel('Relative Error (%)')
        ax3.set_ylim(0, max(10, np.max(relative_errors * 100) * 1.1))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Plot saved to: {save_path}")
        
        plt.show()


def demonstrate_frequency_control():
    """Demonstrate frequency-dependent permittivity control."""
    print("🧪 FREQUENCY-DEPENDENT PERMITTIVITY CONTROL DEMO")
    print("=" * 55)
    
    # Initialize controller
    controller = FrequencyDependentPermittivityController(
        frequency_range=(10e12, 100e12)
    )
    
    # Test 1: Constant target permittivity
    print(f"\n1️⃣  Constant Target Permittivity")
    result1 = controller.optimize_frequency_dependent_permittivity(
        target_permittivity=2.5,
        materials=['gold', 'silicon'],
        frequency_points=500,
        tolerance=0.05
    )
    
    if result1['success']:
        validation1 = controller.validate_frequency_control(result1)
        print(f"   Material fractions: {result1['optimal_fractions']}")
    
    # Test 2: Frequency-dependent target
    print(f"\n2️⃣  Frequency-Dependent Target")
    def target_function(frequencies):
        # Varying permittivity: 2.0 to 3.0 across frequency range
        freq_norm = (frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0])
        return 2.0 + 1.0 * freq_norm
    
    result2 = controller.optimize_frequency_dependent_permittivity(
        target_permittivity=target_function,
        materials=['silver', 'silicon_dioxide'],
        frequency_points=500,
        tolerance=0.05
    )
    
    if result2['success']:
        validation2 = controller.validate_frequency_control(result2)
        print(f"   Material fractions: {result2['optimal_fractions']}")
        
        # Monte Carlo analysis
        print(f"\n3️⃣  Monte Carlo Uncertainty Analysis")
        mc_result = controller.monte_carlo_uncertainty_propagation(
            result2, 
            parameter_uncertainties={'material_fraction': 0.02, 'material_parameters': 0.03},
            n_samples=1000  # Reduced for demo
        )
    
    print(f"\n✅ FREQUENCY CONTROL DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    demonstrate_frequency_control()
