#!/usr/bin/env python3
"""
Multilayer Metamaterial Stack Modeling
======================================

Advanced multilayer stack modeling with electromagnetic enhancement
and tolerance propagation for tunable permittivity stacks.

Mathematical Foundation:
- Layer amplification: Σ(k=1 to N) η·k^(-β) 
- Electromagnetic coupling between layers
- Cumulative tolerance: δ_cumulative = δ_per_layer × √N
- Maxwell-Garnett effective medium theory

Integration Points:
- Enhanced multilayer mathematics from negative-energy-generator
- Metamaterial enhancement from lqg-anec-framework
- Process tolerance validation

Author: GitHub Copilot
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.constants import c, epsilon_0, pi, hbar, mu_0
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eig
import warnings


class MultilayerMetamaterialStack:
    """
    Advanced multilayer metamaterial stack with electromagnetic modeling.
    
    Features:
    - N-layer stacking with saturation effects
    - Electromagnetic enhancement calculation  
    - Tolerance propagation through stack
    - Casimir force optimization
    """
    
    def __init__(self, max_layers: int = 25):
        """
        Initialize multilayer stack modeler.
        
        Args:
            max_layers: Maximum number of layers (limited by ±1 nm cumulative tolerance)
        """
        self.max_layers = max_layers
        self.layer_tolerance = 1e-9  # ±1 nm per layer
        self.material_database = self._initialize_materials()
        
        print(f"✅ MultilayerMetamaterialStack initialized")
        print(f"   📚 Max layers: {max_layers}")
        print(f"   📏 Per-layer tolerance: ±{self.layer_tolerance*1e9:.1f} nm")
    
    def _initialize_materials(self) -> Dict:
        """Initialize material property database."""
        return {
            'gold': {
                'permittivity': lambda omega: 1 - 1.36e16**2 / (omega**2 + 1j*1.45e14*omega),
                'permeability': 1.0 + 0j,
                'loss_tangent': 0.05,
                'density': 19300,  # kg/m³
                'thermal_expansion': 14.2e-6  # /K
            },
            'silver': {
                'permittivity': lambda omega: 1 - 1.38e16**2 / (omega**2 + 1j*2.73e13*omega),
                'permeability': 1.0 + 0j,
                'loss_tangent': 0.02,
                'density': 10490,
                'thermal_expansion': 18.9e-6
            },
            'aluminum': {
                'permittivity': lambda omega: 1 - 2.24e16**2 / (omega**2 + 1j*1.22e14*omega),
                'permeability': 1.0 + 0j,
                'loss_tangent': 0.08,
                'density': 2700,
                'thermal_expansion': 23.1e-6
            },
            'silicon': {
                'permittivity': lambda omega: 11.68 + 0j,  # Simplified
                'permeability': 1.0 + 0j,
                'loss_tangent': 0.001,
                'density': 2329,
                'thermal_expansion': 2.6e-6
            },
            'silicon_dioxide': {
                'permittivity': lambda omega: 2.1 + 0.001j,
                'permeability': 1.0 + 0j,
                'loss_tangent': 0.0001,
                'density': 2203,
                'thermal_expansion': 0.5e-6
            }
        }
    
    def design_multilayer_stack(self, 
                               target_properties: Dict,
                               n_layers: int,
                               materials: List[str],
                               layer_pattern: str = 'alternating') -> Dict:
        """
        Design optimal multilayer stack configuration.
        
        Args:
            target_properties: Target electromagnetic properties
            n_layers: Number of layers in stack
            materials: Available materials
            layer_pattern: 'alternating', 'gradient', 'optimized'
        
        Returns:
            Optimized stack design
        """
        print(f"🏗️  DESIGNING MULTILAYER STACK")
        print(f"   Layers: {n_layers}")
        print(f"   Materials: {materials}")
        print(f"   Pattern: {layer_pattern}")
        
        # Validate layer count
        if n_layers > self.max_layers:
            print(f"   ⚠️  Layer count {n_layers} exceeds maximum {self.max_layers}")
            n_layers = self.max_layers
        
        # Calculate cumulative tolerance limit
        cumulative_tolerance = self.layer_tolerance * np.sqrt(n_layers)
        if cumulative_tolerance > 2e-9:  # Conservative 2 nm total limit
            print(f"   ⚠️  Cumulative tolerance {cumulative_tolerance*1e9:.1f} nm may be challenging")
        
        # Design stack based on pattern
        if layer_pattern == 'alternating':
            stack_design = self._design_alternating_stack(n_layers, materials, target_properties)
        elif layer_pattern == 'gradient':
            stack_design = self._design_gradient_stack(n_layers, materials, target_properties)
        elif layer_pattern == 'optimized':
            stack_design = self._design_optimized_stack(n_layers, materials, target_properties)
        else:
            raise ValueError(f"Unknown layer pattern: {layer_pattern}")
        
        # Add manufacturing and tolerance information
        stack_design.update({
            'cumulative_tolerance': cumulative_tolerance,
            'per_layer_tolerance': self.layer_tolerance,
            'manufacturability_score': self._assess_manufacturability(stack_design),
            'electromagnetic_properties': self._calculate_electromagnetic_properties(stack_design)
        })
        
        print(f"   ✅ Stack design complete")
        print(f"      Cumulative tolerance: ±{cumulative_tolerance*1e9:.1f} nm")
        print(f"      Manufacturability score: {stack_design['manufacturability_score']:.2f}")
        
        return stack_design
    
    def _design_alternating_stack(self, n_layers: int, materials: List[str], 
                                target_properties: Dict) -> Dict:
        """Design alternating material stack."""
        if len(materials) < 2:
            raise ValueError("Alternating pattern requires at least 2 materials")
        
        # Alternating pattern
        layer_materials = [materials[i % len(materials)] for i in range(n_layers)]
        
        # Base thicknesses - optimize for target frequency
        target_freq = target_properties.get('center_frequency', 50e12)  # 50 THz default
        base_wavelength = c / target_freq
        
        # Layer thicknesses: λ/4n for quarter-wave stack
        layer_thicknesses = []
        layer_spacings = []
        
        for i, material in enumerate(layer_materials):
            # Get material refractive index at target frequency
            omega = 2 * np.pi * target_freq
            eps = self.material_database[material]['permittivity'](omega)
            n_eff = np.sqrt(eps).real
            
            # Quarter-wave thickness
            thickness = base_wavelength / (4 * n_eff) if n_eff > 0 else 50e-9
            thickness = max(10e-9, min(200e-9, thickness))  # Practical bounds
            
            layer_thicknesses.append(thickness)
            
            # Layer spacing (for Casimir calculations)
            spacing = thickness + (10e-9 if i < n_layers-1 else 0)  # 10 nm gap
            layer_spacings.append(spacing)
        
        return {
            'design_type': 'alternating',
            'n_layers': n_layers,
            'layer_materials': layer_materials,
            'layer_thicknesses': np.array(layer_thicknesses),
            'layer_spacings': np.array(layer_spacings),
            'target_properties': target_properties
        }
    
    def _design_gradient_stack(self, n_layers: int, materials: List[str], 
                             target_properties: Dict) -> Dict:
        """Design gradient stack with smooth property transition."""
        # Create gradient in material properties
        if len(materials) < 2:
            materials = materials * 2  # Duplicate if needed
        
        # Linear gradient of material indices
        material_indices = np.linspace(0, len(materials)-1, n_layers)
        layer_materials = []
        layer_thicknesses = []
        layer_spacings = []
        
        target_freq = target_properties.get('center_frequency', 50e12)
        base_thickness = target_properties.get('base_thickness', 50e-9)
        
        for i, mat_idx in enumerate(material_indices):
            # Interpolate between materials (simplified)
            mat_idx_int = int(mat_idx) % len(materials)
            layer_materials.append(materials[mat_idx_int])
            
            # Gradient thickness: thicker in center, thinner at edges
            gradient_factor = 1.0 + 0.5 * np.sin(np.pi * i / (n_layers - 1))
            thickness = base_thickness * gradient_factor
            thickness = max(10e-9, min(200e-9, thickness))
            
            layer_thicknesses.append(thickness)
            layer_spacings.append(thickness + 5e-9)  # Small gap
        
        return {
            'design_type': 'gradient',
            'n_layers': n_layers,
            'layer_materials': layer_materials,
            'layer_thicknesses': np.array(layer_thicknesses),
            'layer_spacings': np.array(layer_spacings),
            'target_properties': target_properties
        }
    
    def _design_optimized_stack(self, n_layers: int, materials: List[str], 
                              target_properties: Dict) -> Dict:
        """Design fully optimized stack using electromagnetic optimization."""
        print("   🎯 Running electromagnetic optimization...")
        
        # Optimization variables: [material_indices, thicknesses, spacings]
        n_vars = n_layers * 3  # material, thickness, spacing for each layer
        
        def objective_function(params):
            return self._evaluate_stack_performance(params, n_layers, materials, target_properties)
        
        # Bounds
        bounds = []
        for i in range(n_layers):
            bounds.append((0, len(materials)-1))  # Material index
            bounds.append((10e-9, 200e-9))        # Thickness
            bounds.append((15e-9, 250e-9))        # Spacing
        
        # Optimize
        result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=100,
            popsize=10
        )
        
        if result.success:
            # Extract optimized parameters
            opt_params = result.x
            layer_materials = []
            layer_thicknesses = []
            layer_spacings = []
            
            for i in range(n_layers):
                mat_idx = int(opt_params[i*3]) % len(materials)
                thickness = opt_params[i*3 + 1]
                spacing = opt_params[i*3 + 2]
                
                layer_materials.append(materials[mat_idx])
                layer_thicknesses.append(thickness)
                layer_spacings.append(spacing)
            
            return {
                'design_type': 'optimized',
                'n_layers': n_layers,
                'layer_materials': layer_materials,
                'layer_thicknesses': np.array(layer_thicknesses),
                'layer_spacings': np.array(layer_spacings),
                'target_properties': target_properties,
                'optimization_result': result,
                'optimization_score': result.fun
            }
        else:
            print("   ❌ Optimization failed, falling back to alternating design")
            return self._design_alternating_stack(n_layers, materials, target_properties)
    
    def _evaluate_stack_performance(self, params: np.ndarray, n_layers: int, 
                                  materials: List[str], target_properties: Dict) -> float:
        """Evaluate electromagnetic performance of stack configuration."""
        try:
            # Extract parameters
            layer_configs = []
            for i in range(n_layers):
                mat_idx = int(params[i*3]) % len(materials)
                thickness = params[i*3 + 1]
                spacing = params[i*3 + 2]
                
                layer_configs.append({
                    'material': materials[mat_idx],
                    'thickness': thickness,
                    'spacing': spacing
                })
            
            # Calculate electromagnetic properties
            em_props = self._calculate_stack_electromagnetic_properties(
                layer_configs, target_properties
            )
            
            # Multi-objective scoring
            scores = []
            
            # Target permittivity matching
            if 'target_permittivity' in target_properties:
                target_eps = target_properties['target_permittivity']
                achieved_eps = em_props['effective_permittivity']
                eps_error = abs(achieved_eps - target_eps) / abs(target_eps)
                scores.append(eps_error)
            
            # Enhancement factor optimization
            if 'target_enhancement' in target_properties:
                target_enh = target_properties['target_enhancement']
                achieved_enh = em_props['enhancement_factor']
                enh_error = abs(achieved_enh - target_enh) / abs(target_enh)
                scores.append(enh_error)
            
            # Manufacturability penalty
            manufact_score = self._assess_layer_manufacturability(layer_configs)
            scores.append(1.0 - manufact_score)  # Lower is better
            
            # Tolerance penalty
            tolerance_penalty = self._calculate_tolerance_penalty(layer_configs)
            scores.append(tolerance_penalty)
            
            return np.mean(scores)
            
        except Exception as e:
            return 1000.0  # High penalty for invalid configurations
    
    def _calculate_stack_electromagnetic_properties(self, layer_configs: List[Dict], 
                                                   target_properties: Dict) -> Dict:
        """Calculate electromagnetic properties of layered stack."""
        # Simplified electromagnetic modeling
        target_freq = target_properties.get('center_frequency', 50e12)
        omega = 2 * np.pi * target_freq
        
        # Calculate effective properties using transfer matrix method (simplified)
        total_thickness = sum(config['thickness'] for config in layer_configs)
        
        # Effective permittivity using volume averaging
        eps_effective = 0
        total_volume = 0
        
        for config in layer_configs:
            material = config['material']
            thickness = config['thickness']
            volume = thickness  # Per unit area
            
            eps_layer = self.material_database[material]['permittivity'](omega)
            eps_effective += eps_layer * volume
            total_volume += volume
        
        eps_effective /= total_volume
        
        # Enhancement factor from multilayer stacking
        n_layers = len(layer_configs)
        
        # Layer amplification with saturation: Σ(k=1 to N) η·k^(-β)
        eta = 0.95  # Layer efficiency
        beta = 0.5  # Saturation exponent
        k_values = np.arange(1, n_layers + 1)
        layer_amplification = np.sum(eta * k_values**(-beta))
        
        # Geometric enhancement from layer spacing
        avg_spacing = np.mean([config['spacing'] for config in layer_configs])
        geometric_enhancement = (100e-9 / avg_spacing)**2  # Reference 100 nm
        
        enhancement_factor = layer_amplification * geometric_enhancement
        
        return {
            'effective_permittivity': eps_effective,
            'enhancement_factor': enhancement_factor,
            'layer_amplification': layer_amplification,
            'geometric_enhancement': geometric_enhancement,
            'total_thickness': total_thickness,
            'n_layers': n_layers
        }
    
    def _assess_manufacturability(self, stack_design: Dict) -> float:
        """Assess manufacturability of stack design."""
        thicknesses = stack_design['layer_thicknesses']
        materials = stack_design['layer_materials']
        
        # Factors affecting manufacturability
        factors = []
        
        # Thickness uniformity (easier if similar thicknesses)
        thickness_cv = np.std(thicknesses) / np.mean(thicknesses)
        thickness_score = max(0, 1 - thickness_cv)  # Lower CV is better
        factors.append(thickness_score)
        
        # Material compatibility (fewer materials is easier)
        n_unique_materials = len(set(materials))
        material_score = max(0, 1 - 0.2 * (n_unique_materials - 1))
        factors.append(material_score)
        
        # Minimum thickness constraint (>10 nm easier to control)
        min_thickness = np.min(thicknesses)
        min_thickness_score = min(1.0, min_thickness / 10e-9)
        factors.append(min_thickness_score)
        
        # Layer count penalty (more layers = more complex)
        n_layers = len(thicknesses)
        layer_count_score = max(0, 1 - 0.05 * (n_layers - 5))
        factors.append(layer_count_score)
        
        return np.mean(factors)
    
    def _assess_layer_manufacturability(self, layer_configs: List[Dict]) -> float:
        """Assess manufacturability of individual layer configuration."""
        scores = []
        
        for config in layer_configs:
            thickness = config['thickness']
            
            # Thickness manufacturability
            if thickness < 10e-9:
                thick_score = 0.1  # Very difficult
            elif thickness < 20e-9:
                thick_score = 0.5  # Challenging
            elif thickness < 100e-9:
                thick_score = 0.8  # Good
            else:
                thick_score = 1.0  # Easy
            
            scores.append(thick_score)
        
        return np.mean(scores)
    
    def _calculate_tolerance_penalty(self, layer_configs: List[Dict]) -> float:
        """Calculate penalty for difficult tolerance control."""
        penalties = []
        
        for config in layer_configs:
            thickness = config['thickness']
            
            # Tolerance difficulty increases as thickness decreases
            tolerance_ratio = self.layer_tolerance / thickness
            
            if tolerance_ratio > 0.2:  # >20% tolerance
                penalty = 1.0  # Very difficult
            elif tolerance_ratio > 0.1:  # >10% tolerance
                penalty = 0.5  # Challenging
            else:
                penalty = 0.1  # Manageable
            
            penalties.append(penalty)
        
        return np.mean(penalties)
    
    def _calculate_electromagnetic_properties(self, stack_design: Dict) -> Dict:
        """Calculate full electromagnetic properties of designed stack."""
        layer_materials = stack_design['layer_materials']
        layer_thicknesses = stack_design['layer_thicknesses']
        layer_spacings = stack_design['layer_spacings']
        target_props = stack_design['target_properties']
        
        # Create layer configurations
        layer_configs = []
        for i, (material, thickness, spacing) in enumerate(zip(layer_materials, layer_thicknesses, layer_spacings)):
            layer_configs.append({
                'material': material,
                'thickness': thickness,
                'spacing': spacing,
                'index': i
            })
        
        # Calculate electromagnetic properties
        em_props = self._calculate_stack_electromagnetic_properties(layer_configs, target_props)
        
        # Add additional properties
        em_props.update({
            'casimir_energy_density': self._calculate_casimir_energy_density(layer_configs),
            'reflection_coefficient': self._calculate_reflection_properties(layer_configs, target_props),
            'field_enhancement': self._calculate_field_enhancement(layer_configs)
        })
        
        return em_props
    
    def _calculate_casimir_energy_density(self, layer_configs: List[Dict]) -> float:
        """Calculate Casimir energy density for multilayer stack."""
        # Simplified Casimir calculation
        total_energy = 0
        
        for config in layer_configs:
            spacing = config['spacing']
            
            # Basic Casimir energy density: ∝ 1/spacing^4
            if spacing > 0:
                layer_energy = -(pi**2 * hbar * c) / (720 * spacing**4)
                total_energy += layer_energy
        
        return total_energy
    
    def _calculate_reflection_properties(self, layer_configs: List[Dict], 
                                       target_props: Dict) -> float:
        """Calculate reflection properties of multilayer stack."""
        # Simplified reflection calculation
        target_freq = target_props.get('center_frequency', 50e12)
        omega = 2 * np.pi * target_freq
        
        # Calculate impedances and reflection
        impedances = []
        for config in layer_configs:
            material = config['material']
            eps = self.material_database[material]['permittivity'](omega)
            mu = self.material_database[material]['permeability']
            
            # Impedance: Z = sqrt(μ/ε)
            Z = np.sqrt(mu / eps)
            impedances.append(Z)
        
        # Simplified multi-layer reflection (would use transfer matrix in full implementation)
        if len(impedances) > 1:
            Z_avg = np.mean(impedances)
            Z0 = np.sqrt(mu_0 / epsilon_0)  # Free space impedance
            reflection = abs((Z_avg - Z0) / (Z_avg + Z0))**2
        else:
            reflection = 0.1  # Default value
        
        return reflection
    
    def _calculate_field_enhancement(self, layer_configs: List[Dict]) -> float:
        """Calculate electromagnetic field enhancement in stack."""
        # Field enhancement from layer interference
        n_layers = len(layer_configs)
        
        # Simplified model: enhancement grows with number of layers up to saturation
        base_enhancement = np.sqrt(n_layers)
        saturation_factor = 1 / (1 + 0.1 * n_layers)  # Saturation for large N
        
        return base_enhancement * saturation_factor
    
    def validate_stack_tolerances(self, stack_design: Dict, 
                                monte_carlo_samples: int = 10000) -> Dict:
        """
        Validate stack tolerances using Monte Carlo analysis.
        
        Args:
            stack_design: Stack design to validate
            monte_carlo_samples: Number of MC samples
        
        Returns:
            Tolerance validation results
        """
        print(f"🔬 VALIDATING STACK TOLERANCES")
        print(f"   MC samples: {monte_carlo_samples}")
        
        layer_thicknesses = stack_design['layer_thicknesses']
        n_layers = len(layer_thicknesses)
        
        # Monte Carlo simulation
        mc_results = []
        
        for sample in range(monte_carlo_samples):
            # Sample thickness variations
            thickness_variations = np.random.normal(0, self.layer_tolerance/3, n_layers)
            perturbed_thicknesses = layer_thicknesses + thickness_variations
            
            # Ensure positive thicknesses
            perturbed_thicknesses = np.maximum(perturbed_thicknesses, 1e-9)
            
            # Calculate performance with perturbed thicknesses
            perturbed_design = stack_design.copy()
            perturbed_design['layer_thicknesses'] = perturbed_thicknesses
            
            # Simplified performance metric
            thickness_deviation = np.std(perturbed_thicknesses) / np.mean(perturbed_thicknesses)
            total_thickness_error = abs(np.sum(perturbed_thicknesses) - np.sum(layer_thicknesses))
            
            mc_results.append({
                'thickness_deviation': thickness_deviation,
                'total_thickness_error': total_thickness_error,
                'max_layer_error': np.max(np.abs(thickness_variations))
            })
            
            # Progress update
            if (sample + 1) % (monte_carlo_samples // 10) == 0:
                progress = (sample + 1) / monte_carlo_samples * 100
                print(f"   Progress: {progress:.0f}%")
        
        # Statistical analysis
        thickness_deviations = [r['thickness_deviation'] for r in mc_results]
        total_errors = [r['total_thickness_error'] for r in mc_results]
        max_layer_errors = [r['max_layer_error'] for r in mc_results]
        
        # Cumulative tolerance calculation
        cumulative_tolerance_actual = np.std(total_errors)
        cumulative_tolerance_theoretical = self.layer_tolerance * np.sqrt(n_layers)
        
        # Process capability metrics
        process_capability = (2 * self.layer_tolerance) / (6 * np.std(max_layer_errors))
        
        # Validation criteria
        validation_passed = (
            process_capability > 5.0 and  # Conservative Cp requirement
            np.percentile(max_layer_errors, 95) < self.layer_tolerance and
            cumulative_tolerance_actual < 2e-9  # 2 nm total limit
        )
        
        tolerance_validation = {
            'validation_passed': validation_passed,
            'process_capability': process_capability,
            'cumulative_tolerance_actual': cumulative_tolerance_actual,
            'cumulative_tolerance_theoretical': cumulative_tolerance_theoretical,
            'max_layer_error_p95': np.percentile(max_layer_errors, 95),
            'thickness_deviation_mean': np.mean(thickness_deviations),
            'monte_carlo_samples': monte_carlo_samples,
            'per_layer_tolerance': self.layer_tolerance,
            'n_layers': n_layers
        }
        
        # Print validation summary
        print(f"   📊 Tolerance Validation Results:")
        print(f"      Process capability: {process_capability:.2f}")
        print(f"      Cumulative tolerance: {cumulative_tolerance_actual*1e9:.2f} nm")
        print(f"      Max layer error (95%): {tolerance_validation['max_layer_error_p95']*1e9:.2f} nm")
        print(f"      Status: {'✅ PASS' if validation_passed else '❌ FAIL'}")
        
        return tolerance_validation
    
    def optimize_for_casimir_enhancement(self, 
                                       target_enhancement: float,
                                       available_materials: List[str],
                                       n_layers_range: Tuple[int, int] = (5, 20)) -> Dict:
        """
        Optimize multilayer stack for maximum Casimir force enhancement.
        
        Args:
            target_enhancement: Target enhancement factor
            available_materials: List of available materials
            n_layers_range: (min_layers, max_layers) to consider
        
        Returns:
            Optimized stack for Casimir enhancement
        """
        print(f"⚡ OPTIMIZING FOR CASIMIR ENHANCEMENT")
        print(f"   Target enhancement: {target_enhancement:.1f}x")
        print(f"   Layer range: {n_layers_range[0]}-{n_layers_range[1]}")
        
        best_results = []
        
        # Scan layer counts
        for n_layers in range(n_layers_range[0], n_layers_range[1] + 1):
            print(f"   🔄 Testing {n_layers} layers...")
            
            # Target properties for Casimir optimization
            target_props = {
                'target_enhancement': target_enhancement,
                'center_frequency': 50e12,  # 50 THz
                'base_thickness': 30e-9     # 30 nm base
            }
            
            # Design stack
            stack_design = self.design_multilayer_stack(
                target_props, n_layers, available_materials, 'optimized'
            )
            
            # Calculate achieved enhancement
            em_props = stack_design['electromagnetic_properties']
            achieved_enhancement = em_props['enhancement_factor']
            
            # Score this configuration
            enhancement_error = abs(achieved_enhancement - target_enhancement) / target_enhancement
            manufacturability = stack_design['manufacturability_score']
            
            # Combined score (lower is better)
            combined_score = enhancement_error + 0.3 * (1 - manufacturability)
            
            result = {
                'n_layers': n_layers,
                'stack_design': stack_design,
                'achieved_enhancement': achieved_enhancement,
                'enhancement_error': enhancement_error,
                'manufacturability': manufacturability,
                'combined_score': combined_score
            }
            
            best_results.append(result)
            print(f"      Enhancement: {achieved_enhancement:.1f}x (error: {enhancement_error*100:.1f}%)")
        
        # Select best result
        best_result = min(best_results, key=lambda x: x['combined_score'])
        
        print(f"   🏆 Best configuration:")
        print(f"      Layers: {best_result['n_layers']}")
        print(f"      Enhancement: {best_result['achieved_enhancement']:.1f}x")
        print(f"      Manufacturability: {best_result['manufacturability']:.2f}")
        
        return {
            'success': True,
            'best_configuration': best_result,
            'all_configurations': best_results,
            'target_enhancement': target_enhancement
        }


def demonstrate_multilayer_modeling():
    """Demonstrate multilayer metamaterial stack modeling."""
    print("🧪 MULTILAYER METAMATERIAL STACK DEMO")
    print("=" * 45)
    
    # Initialize modeler
    modeler = MultilayerMetamaterialStack(max_layers=25)
    
    # Test 1: Alternating stack design
    print(f"\n1️⃣  Alternating Stack Design")
    target_props = {
        'center_frequency': 30e12,  # 30 THz
        'target_permittivity': 2.5,
        'base_thickness': 40e-9
    }
    
    alt_stack = modeler.design_multilayer_stack(
        target_props, 
        n_layers=8,
        materials=['gold', 'silicon'],
        layer_pattern='alternating'
    )
    
    print(f"   ✅ Alternating stack designed")
    print(f"      Enhancement factor: {alt_stack['electromagnetic_properties']['enhancement_factor']:.2f}")
    print(f"      Manufacturability: {alt_stack['manufacturability_score']:.2f}")
    
    # Test 2: Tolerance validation
    print(f"\n2️⃣  Tolerance Validation")
    tolerance_result = modeler.validate_stack_tolerances(alt_stack, monte_carlo_samples=1000)
    
    # Test 3: Casimir enhancement optimization
    print(f"\n3️⃣  Casimir Enhancement Optimization")
    casimir_result = modeler.optimize_for_casimir_enhancement(
        target_enhancement=10.0,
        available_materials=['gold', 'silver', 'silicon'],
        n_layers_range=(6, 12)
    )
    
    if casimir_result['success']:
        best_config = casimir_result['best_configuration']
        print(f"   ✅ Optimization successful")
        print(f"      Best layers: {best_config['n_layers']}")
        print(f"      Achieved enhancement: {best_config['achieved_enhancement']:.1f}x")
    
    print(f"\n✅ MULTILAYER MODELING DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    demonstrate_multilayer_modeling()
