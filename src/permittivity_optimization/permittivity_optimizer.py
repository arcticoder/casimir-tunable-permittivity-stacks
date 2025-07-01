#!/usr/bin/env python3
"""
Permittivity Optimization System
===============================

Advanced permittivity optimization for tunable Casimir force engineering
with precision frequency-dependent control and metamaterial enhancement.

Mathematical Foundation:
- Drude-Lorentz permittivity optimization
- Metamaterial-enhanced Casimir force control
- Multi-objective optimization across ε(ω), μ(ω), and geometry
- Differential evolution with constraint handling

Integration Points:
- DrudeLorentzPermittivity from unified-lqg-qft
- MetamaterialCasimir optimization from lqg-anec-framework
- Multilayer enhancement from negative-energy-generator

Author: GitHub Copilot
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.constants import c, epsilon_0, pi, hbar, mu_0
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from scipy.interpolate import interp1d
import warnings


class PermittivityOptimizer:
    """
    Advanced permittivity optimization system for tunable Casimir engineering.
    
    Optimizes material compositions and multilayer configurations for:
    - Precise frequency-dependent permittivity control
    - Casimir force sign and magnitude engineering
    - Manufacturing tolerance compliance
    """
    
    def __init__(self, frequency_range: Tuple[float, float] = (10e12, 100e12),
                 tolerance_target: float = 0.05):
        """
        Initialize permittivity optimizer.
        
        Args:
            frequency_range: Target frequency range [Hz]
            tolerance_target: Target ΔRe[ε(ω)]/ε tolerance
        """
        self.frequency_range = frequency_range
        self.tolerance_target = tolerance_target
        
        # Initialize material and optimization databases
        self.material_library = self._initialize_material_library()
        self.optimization_history = []
        self.constraint_handlers = {}
        
        print(f"✅ PermittivityOptimizer initialized")
        print(f"   📊 Frequency range: {frequency_range[0]/1e12:.1f}-{frequency_range[1]/1e12:.1f} THz")
        print(f"   🎯 Tolerance target: ±{tolerance_target*100:.1f}%")
    
    def _initialize_material_library(self) -> Dict:
        """Initialize comprehensive material library with validated parameters."""
        return {
            'gold': {
                'type': 'drude_metal',
                'plasma_frequency': 1.36e16,  # rad/s
                'collision_rate': 1.45e14,   # rad/s
                'oscillators': [
                    {'strength': 0.76, 'omega_p': 2.3e15, 'omega_0': 2.4e15, 'gamma': 1.0e14},
                    {'strength': 0.024, 'omega_p': 2.8e15, 'omega_0': 2.9e15, 'gamma': 5.0e13}
                ],
                'epsilon_inf': 1.0,
                'density': 19300,
                'cost_factor': 3.0,  # Relative cost
                'uncertainty': 0.027  # 2.7% max uncertainty (validated)
            },
            'silver': {
                'type': 'drude_metal',
                'plasma_frequency': 1.38e16,
                'collision_rate': 2.73e13,
                'oscillators': [
                    {'strength': 0.845, 'omega_p': 6.5e15, 'omega_0': 6.7e15, 'gamma': 9.6e14}
                ],
                'epsilon_inf': 1.0,
                'density': 10490,
                'cost_factor': 2.5,
                'uncertainty': 0.037  # 3.7% max uncertainty (validated)
            },
            'aluminum': {
                'type': 'drude_metal',
                'plasma_frequency': 2.24e16,
                'collision_rate': 1.22e14,
                'oscillators': [
                    {'strength': 0.523, 'omega_p': 1.5e15, 'omega_0': 1.6e15, 'gamma': 2.4e14}
                ],
                'epsilon_inf': 1.0,
                'density': 2700,
                'cost_factor': 1.0,
                'uncertainty': 0.051  # 5.1% max uncertainty (borderline)
            },
            'silicon': {
                'type': 'dielectric',
                'plasma_frequency': 0,
                'collision_rate': 0,
                'oscillators': [
                    {'strength': 11.68, 'omega_p': 0, 'omega_0': 3.4e15, 'gamma': 1e13}
                ],
                'epsilon_inf': 1.0,
                'density': 2329,
                'cost_factor': 1.2,
                'uncertainty': 0.02
            },
            'silicon_dioxide': {
                'type': 'dielectric',
                'plasma_frequency': 0,
                'collision_rate': 0,
                'oscillators': [
                    {'strength': 1.1, 'omega_p': 0, 'omega_0': 2.0e16, 'gamma': 1e14}
                ],
                'epsilon_inf': 2.1,
                'density': 2203,
                'cost_factor': 0.8,
                'uncertainty': 0.01
            },
            'metamaterial_negative': {
                'type': 'metamaterial',
                'epsilon_real': -2.0,
                'epsilon_imag': 0.1,
                'mu_real': -1.5,
                'mu_imag': 0.05,
                'frequency_dependent': True,
                'cost_factor': 5.0,  # High cost for custom metamaterials
                'uncertainty': 0.1   # Higher uncertainty for engineered materials
            }
        }
    
    def calculate_material_permittivity(self, material_name: str, 
                                      frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate frequency-dependent permittivity for given material.
        
        Args:
            material_name: Material from library
            frequencies: Frequency array [Hz]
        
        Returns:
            Complex permittivity array ε(ω)
        """
        if material_name not in self.material_library:
            raise ValueError(f"Material '{material_name}' not in library")
        
        material = self.material_library[material_name]
        omega = 2 * np.pi * frequencies
        
        if material['type'] == 'metamaterial':
            # Simple frequency-independent metamaterial
            eps_real = material['epsilon_real']
            eps_imag = material['epsilon_imag']
            return eps_real + 1j * eps_imag * np.ones_like(omega)
        
        # Drude-Lorentz model
        epsilon = np.full_like(omega, material['epsilon_inf'], dtype=complex)
        
        # Drude term
        if material['plasma_frequency'] > 0:
            omega_p = material['plasma_frequency']
            gamma = material['collision_rate']
            drude_term = -omega_p**2 / (omega**2 + 1j*gamma*omega)
            epsilon += drude_term
        
        # Lorentz oscillators
        for osc in material['oscillators']:
            fj = osc['strength']
            omega_pj = osc['omega_p']
            omega_0j = osc['omega_0']
            gamma_j = osc['gamma']
            
            if omega_pj > 0:
                lorentz_term = fj * omega_pj**2 / (omega_0j**2 - omega**2 - 1j*gamma_j*omega)
            else:
                # Static contribution for dielectrics
                lorentz_term = fj / (1 + (omega/omega_0j)**2 + 1j*(omega*gamma_j)/(omega_0j**2))
            
            epsilon += lorentz_term
        
        return epsilon
    
    def optimize_single_material_stack(self, 
                                     target_permittivity: Union[float, np.ndarray, Callable],
                                     available_materials: List[str],
                                     n_layers: int = 10,
                                     optimization_objectives: List[str] = ['permittivity_control', 'casimir_enhancement']) -> Dict:
        """
        Optimize single-material multilayer stack for target properties.
        
        Args:
            target_permittivity: Target ε(ω) profile
            available_materials: List of materials to choose from
            n_layers: Number of layers
            optimization_objectives: Objectives to optimize
        
        Returns:
            Optimization results
        """
        print(f"🎯 OPTIMIZING SINGLE-MATERIAL STACK")
        print(f"   Materials: {available_materials}")
        print(f"   Layers: {n_layers}")
        print(f"   Objectives: {optimization_objectives}")
        
        # Generate frequency grid
        frequencies = np.linspace(self.frequency_range[0], self.frequency_range[1], 1000)
        
        # Process target permittivity
        target_eps = self._process_target_permittivity(target_permittivity, frequencies)
        
        # Setup optimization variables
        # Variables: [material_indices, layer_thicknesses, layer_spacings]
        n_vars = n_layers * 3
        
        def objective_function(params):
            return self._evaluate_single_material_objective(
                params, n_layers, available_materials, frequencies, target_eps, optimization_objectives
            )
        
        # Setup bounds
        bounds = []
        for i in range(n_layers):
            bounds.append((0, len(available_materials) - 1))  # Material index
            bounds.append((10e-9, 200e-9))  # Layer thickness [m]
            bounds.append((15e-9, 500e-9))  # Layer spacing [m]
        
        # Setup constraints
        constraints = self._setup_optimization_constraints(n_layers, available_materials)
        
        # Run optimization
        print("   🔄 Running differential evolution...")
        result = differential_evolution(
            objective_function,
            bounds,
            constraints=constraints,
            seed=42,
            maxiter=200,
            popsize=15,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            print("   ✅ Optimization successful!")
            
            # Extract optimized parameters
            optimized_config = self._extract_optimized_configuration(
                result.x, n_layers, available_materials
            )
            
            # Validate optimized configuration
            validation_results = self._validate_optimized_configuration(
                optimized_config, frequencies, target_eps
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                optimized_config, frequencies, target_eps
            )
            
            optimization_result = {
                'success': True,
                'optimized_configuration': optimized_config,
                'validation_results': validation_results,
                'performance_metrics': performance_metrics,
                'optimization_result': result,
                'target_permittivity': target_eps,
                'frequencies': frequencies,
                'optimization_objectives': optimization_objectives
            }
            
            # Store in optimization history
            self.optimization_history.append(optimization_result)
            
            return optimization_result
            
        else:
            print(f"   ❌ Optimization failed: {result.message}")
            return {
                'success': False,
                'message': result.message,
                'optimization_objectives': optimization_objectives
            }
    
    def _process_target_permittivity(self, target_permittivity: Union[float, np.ndarray, Callable],
                                   frequencies: np.ndarray) -> np.ndarray:
        """Process target permittivity into array format."""
        
        if callable(target_permittivity):
            return target_permittivity(frequencies)
        elif isinstance(target_permittivity, (int, float)):
            return np.full_like(frequencies, target_permittivity)
        else:
            target_array = np.array(target_permittivity)
            if len(target_array) != len(frequencies):
                # Interpolate to match frequency grid
                freq_norm = np.linspace(0, 1, len(target_array))
                freq_new = np.linspace(0, 1, len(frequencies))
                interpolator = interp1d(freq_norm, target_array, kind='cubic', 
                                      bounds_error=False, fill_value='extrapolate')
                return interpolator(freq_new)
            return target_array
    
    def _evaluate_single_material_objective(self, params: np.ndarray, n_layers: int,
                                          available_materials: List[str], frequencies: np.ndarray,
                                          target_eps: np.ndarray, objectives: List[str]) -> float:
        """Evaluate objective function for single-material optimization."""
        try:
            # Extract parameters
            material_indices = params[0:n_layers]
            layer_thicknesses = params[n_layers:2*n_layers]
            layer_spacings = params[2*n_layers:3*n_layers]
            
            # Create layer configuration
            layer_config = []
            for i in range(n_layers):
                mat_idx = int(material_indices[i]) % len(available_materials)
                material = available_materials[mat_idx]
                thickness = layer_thicknesses[i]
                spacing = layer_spacings[i]
                
                layer_config.append({
                    'material': material,
                    'thickness': thickness,
                    'spacing': spacing
                })
            
            # Calculate effective permittivity
            effective_eps = self._calculate_effective_stack_permittivity(layer_config, frequencies)
            
            # Multi-objective scoring
            objective_scores = []
            
            if 'permittivity_control' in objectives:
                perm_score = self._evaluate_permittivity_control_objective(
                    effective_eps, target_eps
                )
                objective_scores.append(perm_score)
            
            if 'casimir_enhancement' in objectives:
                casimir_score = self._evaluate_casimir_enhancement_objective(layer_config)
                objective_scores.append(casimir_score)
            
            if 'manufacturability' in objectives:
                manuf_score = self._evaluate_manufacturability_objective(layer_config)
                objective_scores.append(manuf_score)
            
            if 'cost_efficiency' in objectives:
                cost_score = self._evaluate_cost_efficiency_objective(layer_config)
                objective_scores.append(cost_score)
            
            # Weighted combination
            weights = [1.0, 0.8, 0.5, 0.3][:len(objective_scores)]
            return np.average(objective_scores, weights=weights)
            
        except Exception as e:
            return 1000.0  # High penalty for invalid configurations
    
    def _calculate_effective_stack_permittivity(self, layer_config: List[Dict],
                                              frequencies: np.ndarray) -> np.ndarray:
        """Calculate effective permittivity of multilayer stack."""
        
        # Get individual layer permittivities
        layer_permittivities = []
        layer_weights = []
        
        total_thickness = sum(layer['thickness'] for layer in layer_config)
        
        for layer in layer_config:
            material = layer['material']
            thickness = layer['thickness']
            
            # Calculate material permittivity
            eps_layer = self.calculate_material_permittivity(material, frequencies)
            layer_permittivities.append(eps_layer)
            
            # Thickness-based weighting
            weight = thickness / total_thickness
            layer_weights.append(weight)
        
        # Effective medium approximation (Maxwell-Garnett could be more accurate)
        effective_eps = np.zeros_like(frequencies, dtype=complex)
        
        for eps_layer, weight in zip(layer_permittivities, layer_weights):
            effective_eps += weight * eps_layer
        
        return effective_eps
    
    def _evaluate_permittivity_control_objective(self, effective_eps: np.ndarray,
                                                target_eps: np.ndarray) -> float:
        """Evaluate permittivity control objective."""
        
        # Relative error in real part
        relative_errors = np.abs(effective_eps.real - target_eps.real) / np.abs(target_eps.real)
        
        # Objective components
        max_error = np.max(relative_errors)
        mean_error = np.mean(relative_errors)
        
        # Tolerance compliance penalty
        tolerance_violations = np.sum(relative_errors > self.tolerance_target)
        tolerance_penalty = tolerance_violations / len(relative_errors) * 10
        
        return max_error + 0.5 * mean_error + tolerance_penalty
    
    def _evaluate_casimir_enhancement_objective(self, layer_config: List[Dict]) -> float:
        """Evaluate Casimir force enhancement objective."""
        
        # Simplified Casimir enhancement calculation
        total_enhancement = 0
        
        for layer in layer_config:
            spacing = layer['spacing']
            material = layer['material']
            
            # Base Casimir energy ∝ 1/spacing^4
            if spacing > 0:
                base_energy = 1 / spacing**4
                
                # Material enhancement factor
                material_props = self.material_library[material]
                if material_props['type'] == 'metamaterial':
                    # Metamaterial can provide significant enhancement
                    eps_real = material_props['epsilon_real']
                    mu_real = material_props['mu_real']
                    material_factor = abs(eps_real * mu_real)
                else:
                    # Regular material enhancement
                    material_factor = 1.0
                
                layer_enhancement = base_energy * material_factor
                total_enhancement += layer_enhancement
        
        # Return negative (for minimization) of enhancement
        return -total_enhancement * 1e30  # Scale for numerical stability
    
    def _evaluate_manufacturability_objective(self, layer_config: List[Dict]) -> float:
        """Evaluate manufacturability objective."""
        
        penalties = []
        
        for layer in layer_config:
            thickness = layer['thickness']
            material = layer['material']
            
            # Thickness manufacturability
            if thickness < 10e-9:
                thickness_penalty = 1.0  # Very difficult
            elif thickness < 20e-9:
                thickness_penalty = 0.5  # Challenging
            else:
                thickness_penalty = 0.1  # Manageable
            
            # Material manufacturability
            material_props = self.material_library[material]
            if material_props['type'] == 'metamaterial':
                material_penalty = 0.8  # Complex to manufacture
            else:
                material_penalty = 0.2  # Standard materials
            
            penalties.append(thickness_penalty + material_penalty)
        
        return np.mean(penalties)
    
    def _evaluate_cost_efficiency_objective(self, layer_config: List[Dict]) -> float:
        """Evaluate cost efficiency objective."""
        
        total_cost = 0
        total_volume = 0
        
        for layer in layer_config:
            thickness = layer['thickness']
            material = layer['material']
            
            material_props = self.material_library[material]
            cost_factor = material_props['cost_factor']
            density = material_props['density']
            
            # Volume per unit area
            volume = thickness
            mass = volume * density
            layer_cost = mass * cost_factor
            
            total_cost += layer_cost
            total_volume += volume
        
        # Cost per unit volume
        return total_cost / total_volume if total_volume > 0 else 1000.0
    
    def _setup_optimization_constraints(self, n_layers: int, 
                                      available_materials: List[str]) -> List:
        """Setup optimization constraints."""
        constraints = []
        
        # Constraint: Total thickness reasonable for manufacturing
        def total_thickness_constraint(params):
            layer_thicknesses = params[n_layers:2*n_layers]
            total_thickness = np.sum(layer_thicknesses)
            return 1e-6 - total_thickness  # Max 1 μm total thickness
        
        constraints.append({
            'type': 'ineq',
            'fun': total_thickness_constraint
        })
        
        # Constraint: Thickness variations not too extreme
        def thickness_variation_constraint(params):
            layer_thicknesses = params[n_layers:2*n_layers]
            thickness_cv = np.std(layer_thicknesses) / np.mean(layer_thicknesses)
            return 2.0 - thickness_cv  # CV < 2.0
        
        constraints.append({
            'type': 'ineq',
            'fun': thickness_variation_constraint
        })
        
        return constraints
    
    def _extract_optimized_configuration(self, opt_params: np.ndarray, n_layers: int,
                                       available_materials: List[str]) -> Dict:
        """Extract optimized configuration from optimization parameters."""
        
        material_indices = opt_params[0:n_layers]
        layer_thicknesses = opt_params[n_layers:2*n_layers]
        layer_spacings = opt_params[2*n_layers:3*n_layers]
        
        # Extract layer configuration
        layers = []
        for i in range(n_layers):
            mat_idx = int(material_indices[i]) % len(available_materials)
            material = available_materials[mat_idx]
            thickness = layer_thicknesses[i]
            spacing = layer_spacings[i]
            
            layers.append({
                'layer_index': i,
                'material': material,
                'thickness': thickness,
                'spacing': spacing,
                'material_properties': self.material_library[material]
            })
        
        # Calculate summary statistics
        total_thickness = np.sum(layer_thicknesses)
        unique_materials = list(set(layer['material'] for layer in layers))
        
        return {
            'n_layers': n_layers,
            'layers': layers,
            'layer_materials': [layer['material'] for layer in layers],
            'layer_thicknesses': layer_thicknesses,
            'layer_spacings': layer_spacings,
            'total_thickness': total_thickness,
            'unique_materials': unique_materials,
            'material_count': len(unique_materials)
        }
    
    def _validate_optimized_configuration(self, config: Dict, frequencies: np.ndarray,
                                        target_eps: np.ndarray) -> Dict:
        """Validate optimized configuration."""
        
        # Calculate achieved permittivity
        achieved_eps = self._calculate_effective_stack_permittivity(
            config['layers'], frequencies
        )
        
        # Permittivity control validation
        relative_errors = np.abs(achieved_eps.real - target_eps.real) / np.abs(target_eps.real)
        
        max_error = np.max(relative_errors)
        mean_error = np.mean(relative_errors)
        tolerance_compliance = np.sum(relative_errors < self.tolerance_target) / len(relative_errors)
        
        # Frequency band analysis
        n_bands = 10
        freq_bands = np.linspace(frequencies[0], frequencies[-1], n_bands + 1)
        band_performance = {}
        
        for i in range(n_bands):
            band_start, band_end = freq_bands[i], freq_bands[i+1]
            band_mask = (frequencies >= band_start) & (frequencies < band_end)
            
            if np.any(band_mask):
                band_errors = relative_errors[band_mask]
                band_performance[f'band_{i+1}'] = {
                    'freq_range_THz': (band_start/1e12, band_end/1e12),
                    'max_error': np.max(band_errors),
                    'mean_error': np.mean(band_errors),
                    'compliance_rate': np.sum(band_errors < self.tolerance_target) / len(band_errors)
                }
        
        # Manufacturing validation
        manufacturing_score = self._assess_manufacturing_feasibility(config)
        
        # Overall validation
        validation_passed = (
            max_error < self.tolerance_target and
            tolerance_compliance > 0.95 and
            manufacturing_score > 0.7
        )
        
        return {
            'validation_passed': validation_passed,
            'achieved_permittivity': achieved_eps,
            'max_relative_error': max_error,
            'mean_relative_error': mean_error,
            'tolerance_compliance_rate': tolerance_compliance,
            'frequency_band_performance': band_performance,
            'manufacturing_feasibility_score': manufacturing_score,
            'tolerance_target': self.tolerance_target
        }
    
    def _assess_manufacturing_feasibility(self, config: Dict) -> float:
        """Assess manufacturing feasibility of configuration."""
        
        scores = []
        
        # Thickness feasibility
        layer_thicknesses = config['layer_thicknesses']
        min_thickness = np.min(layer_thicknesses)
        
        if min_thickness < 5e-9:
            thickness_score = 0.1  # Very difficult
        elif min_thickness < 10e-9:
            thickness_score = 0.5  # Challenging
        elif min_thickness < 50e-9:
            thickness_score = 0.8  # Good
        else:
            thickness_score = 1.0  # Excellent
        
        scores.append(thickness_score)
        
        # Material complexity
        material_count = config['material_count']
        if material_count <= 2:
            material_score = 1.0  # Simple
        elif material_count <= 3:
            material_score = 0.8  # Moderate
        elif material_count <= 4:
            material_score = 0.6  # Complex
        else:
            material_score = 0.4  # Very complex
        
        scores.append(material_score)
        
        # Layer count feasibility
        n_layers = config['n_layers']
        if n_layers <= 10:
            layer_score = 1.0
        elif n_layers <= 15:
            layer_score = 0.8
        elif n_layers <= 20:
            layer_score = 0.6
        else:
            layer_score = 0.4
        
        scores.append(layer_score)
        
        return np.mean(scores)
    
    def _calculate_performance_metrics(self, config: Dict, frequencies: np.ndarray,
                                     target_eps: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        # Electromagnetic performance
        achieved_eps = self._calculate_effective_stack_permittivity(
            config['layers'], frequencies
        )
        
        # Casimir enhancement calculation
        casimir_enhancement = self._calculate_casimir_enhancement_factor(config)
        
        # Manufacturing metrics
        manufacturing_complexity = 1.0 - self._assess_manufacturing_feasibility(config)
        
        # Cost analysis
        total_cost = self._calculate_total_cost(config)
        
        # Uncertainty analysis
        uncertainty_metrics = self._analyze_uncertainty_propagation(config)
        
        return {
            'electromagnetic_performance': {
                'achieved_permittivity': achieved_eps,
                'permittivity_accuracy': self._calculate_permittivity_accuracy(achieved_eps, target_eps),
                'frequency_response_quality': self._assess_frequency_response_quality(achieved_eps, frequencies)
            },
            'casimir_enhancement_factor': casimir_enhancement,
            'manufacturing_complexity': manufacturing_complexity,
            'estimated_total_cost': total_cost,
            'uncertainty_metrics': uncertainty_metrics,
            'overall_performance_score': self._calculate_overall_performance_score(config, achieved_eps, target_eps)
        }
    
    def _calculate_casimir_enhancement_factor(self, config: Dict) -> float:
        """Calculate Casimir force enhancement factor."""
        
        total_enhancement = 1.0  # Base enhancement
        
        for layer in config['layers']:
            spacing = layer['spacing']
            material = layer['material']
            thickness = layer['thickness']
            
            # Geometric enhancement from small spacing
            if spacing > 0:
                geometric_factor = (100e-9 / spacing)**2  # Reference 100 nm
            else:
                geometric_factor = 1.0
            
            # Material enhancement
            material_props = self.material_library[material]
            if material_props['type'] == 'metamaterial':
                # Strong enhancement from negative-index materials
                eps_real = material_props['epsilon_real']
                mu_real = material_props['mu_real']
                material_factor = abs(eps_real * mu_real) * 2  # Enhancement factor
            else:
                material_factor = 1.2  # Modest enhancement from regular materials
            
            # Layer contribution
            layer_enhancement = geometric_factor * material_factor
            total_enhancement *= layer_enhancement**(thickness / sum(l['thickness'] for l in config['layers']))
        
        return total_enhancement
    
    def _calculate_total_cost(self, config: Dict) -> float:
        """Calculate estimated total cost."""
        
        total_cost = 0
        area = 1e-4  # 1 cm² reference area
        
        for layer in config['layers']:
            thickness = layer['thickness']
            material = layer['material']
            
            material_props = self.material_library[material]
            cost_factor = material_props['cost_factor']
            density = material_props['density']
            
            volume = thickness * area
            mass = volume * density
            layer_cost = mass * cost_factor * 100  # Cost scaling factor
            
            total_cost += layer_cost
        
        return total_cost
    
    def _analyze_uncertainty_propagation(self, config: Dict) -> Dict:
        """Analyze uncertainty propagation through stack."""
        
        # Material uncertainties
        material_uncertainties = []
        for layer in config['layers']:
            material = layer['material']
            material_props = self.material_library[material]
            material_uncertainties.append(material_props['uncertainty'])
        
        # Propagated uncertainty (simplified RSS)
        combined_uncertainty = np.sqrt(np.sum(np.array(material_uncertainties)**2))
        
        # Thickness tolerance effects
        thickness_uncertainty = 1e-9 / np.mean(config['layer_thicknesses'])  # Relative
        
        # Total uncertainty
        total_uncertainty = np.sqrt(combined_uncertainty**2 + thickness_uncertainty**2)
        
        return {
            'material_uncertainty_rms': combined_uncertainty,
            'thickness_uncertainty_relative': thickness_uncertainty,
            'total_uncertainty_estimate': total_uncertainty,
            'uncertainty_breakdown': {
                'material_contributions': material_uncertainties,
                'thickness_contribution': thickness_uncertainty
            }
        }
    
    def _calculate_permittivity_accuracy(self, achieved_eps: np.ndarray, 
                                       target_eps: np.ndarray) -> float:
        """Calculate permittivity accuracy metric."""
        
        relative_errors = np.abs(achieved_eps.real - target_eps.real) / np.abs(target_eps.real)
        accuracy = 1.0 - np.mean(relative_errors)  # Higher is better
        return max(0, accuracy)
    
    def _assess_frequency_response_quality(self, achieved_eps: np.ndarray,
                                         frequencies: np.ndarray) -> float:
        """Assess quality of frequency response."""
        
        # Smoothness metric (lower variation is better)
        eps_derivative = np.gradient(achieved_eps.real)
        smoothness = 1.0 / (1.0 + np.std(eps_derivative))
        
        # Bandwidth coverage
        eps_range = np.max(achieved_eps.real) - np.min(achieved_eps.real)
        bandwidth_score = min(1.0, eps_range / 5.0)  # Normalized to 5.0 range
        
        return 0.7 * smoothness + 0.3 * bandwidth_score
    
    def _calculate_overall_performance_score(self, config: Dict, achieved_eps: np.ndarray,
                                           target_eps: np.ndarray) -> float:
        """Calculate overall performance score."""
        
        # Component scores
        accuracy_score = self._calculate_permittivity_accuracy(achieved_eps, target_eps)
        manufacturing_score = self._assess_manufacturing_feasibility(config)
        
        # Enhancement score (normalized)
        enhancement_factor = self._calculate_casimir_enhancement_factor(config)
        enhancement_score = min(1.0, enhancement_factor / 10.0)  # Normalized to 10x
        
        # Weighted combination
        weights = [0.5, 0.3, 0.2]  # Accuracy, manufacturing, enhancement
        scores = [accuracy_score, manufacturing_score, enhancement_score]
        
        return np.average(scores, weights=weights)


def demonstrate_permittivity_optimization():
    """Demonstrate permittivity optimization system."""
    print("🧪 PERMITTIVITY OPTIMIZATION SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = PermittivityOptimizer(
        frequency_range=(10e12, 100e12),
        tolerance_target=0.05
    )
    
    # Test 1: Constant target permittivity
    print(f"\n1️⃣  Constant Target Permittivity Optimization")
    result1 = optimizer.optimize_single_material_stack(
        target_permittivity=2.5,
        available_materials=['gold', 'silicon', 'silicon_dioxide'],
        n_layers=6,
        optimization_objectives=['permittivity_control', 'casimir_enhancement']
    )
    
    if result1['success']:
        validation = result1['validation_results']
        performance = result1['performance_metrics']
        
        print(f"   ✅ Optimization successful")
        print(f"      Max error: {validation['max_relative_error']*100:.2f}%")
        print(f"      Compliance: {validation['tolerance_compliance_rate']*100:.1f}%")
        print(f"      Enhancement: {performance['casimir_enhancement_factor']:.1f}x")
        print(f"      Overall score: {performance['overall_performance_score']:.3f}")
    
    # Test 2: Frequency-dependent target
    print(f"\n2️⃣  Frequency-Dependent Target Optimization")
    def target_function(frequencies):
        # Linear variation across frequency range
        freq_norm = (frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0])
        return 2.0 + 1.5 * freq_norm
    
    result2 = optimizer.optimize_single_material_stack(
        target_permittivity=target_function,
        available_materials=['silver', 'silicon', 'metamaterial_negative'],
        n_layers=8,
        optimization_objectives=['permittivity_control', 'casimir_enhancement', 'manufacturability']
    )
    
    if result2['success']:
        validation = result2['validation_results']
        performance = result2['performance_metrics']
        
        print(f"   ✅ Frequency-dependent optimization successful")
        print(f"      Max error: {validation['max_relative_error']*100:.2f}%")
        print(f"      Manufacturing score: {validation['manufacturing_feasibility_score']:.3f}")
        print(f"      Total cost estimate: ${performance['estimated_total_cost']:.2f}")
    
    print(f"\n✅ PERMITTIVITY OPTIMIZATION DEMONSTRATION COMPLETE")
    print(f"   Optimization history: {len(optimizer.optimization_history)} entries")


if __name__ == "__main__":
    demonstrate_permittivity_optimization()
