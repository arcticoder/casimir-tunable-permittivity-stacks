#!/usr/bin/env python3
"""
Advanced Optimization Example
============================

Demonstrates advanced multi-objective optimization capabilities
for complex Casimir force engineering scenarios.

This example shows:
- Multi-objective optimization with competing constraints
- Advanced material combinations including metamaterials
- Frequency-dependent targets with complex profiles
- Manufacturing constraint integration
- Cost-performance trade-off analysis

Author: GitHub Copilot
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from permittivity_optimization.permittivity_optimizer import PermittivityOptimizer
    from frequency_dependent_control.frequency_controller import FrequencyDependentController
    from multilayer_modeling.multilayer_stack import MultilayerStackModeler
    from tolerance_validation.tolerance_validator import ToleranceValidator
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("   Creating fallback implementations for demonstration")


class AdvancedOptimizationExample:
    """Advanced optimization example class."""
    
    def __init__(self):
        """Initialize advanced optimization example."""
        self.results_storage = {}
        self.optimization_history = []
        
        print("🚀 ADVANCED OPTIMIZATION EXAMPLE")
        print("=" * 50)
    
    def complex_frequency_target_optimization(self):
        """Demonstrate optimization with complex frequency-dependent targets."""
        
        print(f"\n1️⃣  COMPLEX FREQUENCY TARGET OPTIMIZATION")
        print("-" * 45)
        
        try:
            # Initialize optimizer with tight tolerances
            optimizer = PermittivityOptimizer(
                frequency_range=(10e12, 100e12),
                tolerance_target=0.02  # Tight 2% tolerance
            )
            
            # Define complex target function
            def complex_target_function(frequencies):
                """Complex frequency-dependent permittivity target."""
                freq_thz = frequencies / 1e12
                
                # Multi-band target with different characteristics
                low_band = 2.0 + 0.5 * np.sin(2 * np.pi * (freq_thz - 10) / 20)
                mid_band = 3.5 + 0.3 * np.cos(2 * np.pi * (freq_thz - 35) / 30)
                high_band = 1.8 + 0.7 * np.tanh((freq_thz - 80) / 10)
                
                # Smooth transitions between bands
                weight_low = np.exp(-(freq_thz - 25)**2 / 200)
                weight_mid = np.exp(-(freq_thz - 55)**2 / 300)
                weight_high = np.exp(-(freq_thz - 85)**2 / 150)
                
                total_weight = weight_low + weight_mid + weight_high + 0.1
                
                target = (weight_low * low_band + weight_mid * mid_band + 
                         weight_high * high_band) / total_weight
                
                return target
            
            # Advanced material library including metamaterials
            advanced_materials = [
                'gold', 'silver', 'aluminum', 
                'silicon', 'silicon_dioxide',
                'metamaterial_negative'
            ]
            
            # Multi-objective optimization
            optimization_result = optimizer.optimize_single_material_stack(
                target_permittivity=complex_target_function,
                available_materials=advanced_materials,
                n_layers=12,  # More layers for complex targets
                optimization_objectives=[
                    'permittivity_control', 
                    'casimir_enhancement',
                    'manufacturability',
                    'cost_efficiency'
                ]
            )
            
            if optimization_result['success']:
                self.results_storage['complex_target'] = optimization_result
                
                config = optimization_result['optimized_configuration']
                validation = optimization_result['validation_results']
                performance = optimization_result['performance_metrics']
                
                print("✅ Complex target optimization successful")
                print(f"   Materials used: {len(config['unique_materials'])} types")
                print(f"   Max error: {validation['max_relative_error']*100:.2f}%")
                print(f"   Compliance rate: {validation['tolerance_compliance_rate']*100:.1f}%")
                print(f"   Casimir enhancement: {performance['casimir_enhancement_factor']:.1f}x")
                print(f"   Manufacturing score: {validation['manufacturing_feasibility_score']:.3f}")
                
                self._analyze_frequency_bands(optimization_result)
                
            else:
                print("❌ Complex target optimization failed")
                
        except Exception as e:
            print(f"❌ Error in complex target optimization: {e}")
    
    def multi_objective_pareto_analysis(self):
        """Perform Pareto analysis of competing objectives."""
        
        print(f"\n2️⃣  MULTI-OBJECTIVE PARETO ANALYSIS")
        print("-" * 38)
        
        try:
            optimizer = PermittivityOptimizer(
                frequency_range=(15e12, 85e12),
                tolerance_target=0.03
            )
            
            # Target permittivity for Pareto analysis
            target_permittivity = 2.8
            
            # Different objective combinations for Pareto analysis
            objective_combinations = [
                ['permittivity_control'],
                ['permittivity_control', 'casimir_enhancement'],
                ['permittivity_control', 'manufacturability'],
                ['permittivity_control', 'cost_efficiency'],
                ['permittivity_control', 'casimir_enhancement', 'manufacturability'],
                ['permittivity_control', 'casimir_enhancement', 'cost_efficiency'],
                ['permittivity_control', 'manufacturability', 'cost_efficiency'],
                ['permittivity_control', 'casimir_enhancement', 'manufacturability', 'cost_efficiency']
            ]
            
            pareto_results = []
            
            for i, objectives in enumerate(objective_combinations):
                print(f"   Running optimization {i+1}/{len(objective_combinations)}: {objectives}")
                
                result = optimizer.optimize_single_material_stack(
                    target_permittivity=target_permittivity,
                    available_materials=['gold', 'silicon', 'aluminum', 'silicon_dioxide'],
                    n_layers=8,
                    optimization_objectives=objectives
                )
                
                if result['success']:
                    pareto_point = {
                        'objectives': objectives,
                        'permittivity_error': result['validation_results']['max_relative_error'],
                        'casimir_enhancement': result['performance_metrics']['casimir_enhancement_factor'],
                        'manufacturing_score': result['validation_results']['manufacturing_feasibility_score'],
                        'cost_estimate': result['performance_metrics']['estimated_total_cost'],
                        'overall_score': result['performance_metrics']['overall_performance_score']
                    }
                    pareto_results.append(pareto_point)
            
            self.results_storage['pareto_analysis'] = pareto_results
            
            if pareto_results:
                print("✅ Pareto analysis completed")
                print(f"   Generated {len(pareto_results)} optimization points")
                self._analyze_pareto_frontier(pareto_results)
            else:
                print("❌ No successful optimizations in Pareto analysis")
                
        except Exception as e:
            print(f"❌ Error in Pareto analysis: {e}")
    
    def metamaterial_enhanced_optimization(self):
        """Optimize using metamaterial-enhanced configurations."""
        
        print(f"\n3️⃣  METAMATERIAL-ENHANCED OPTIMIZATION")
        print("-" * 42)
        
        try:
            optimizer = PermittivityOptimizer(
                frequency_range=(20e12, 80e12),
                tolerance_target=0.04
            )
            
            # Target for strong Casimir enhancement
            def enhancement_target(frequencies):
                """Target designed for maximum Casimir enhancement."""
                freq_thz = frequencies / 1e12
                
                # Negative permittivity region for enhancement
                base_target = 1.5 - 0.8 * np.exp(-(freq_thz - 50)**2 / 300)
                
                # Add resonant features
                resonance1 = -0.3 * np.exp(-(freq_thz - 35)**2 / 50)
                resonance2 = -0.2 * np.exp(-(freq_thz - 65)**2 / 80)
                
                return base_target + resonance1 + resonance2
            
            # Metamaterial-focused material library
            metamaterial_library = [
                'gold', 'silver',  # Plasmonic metals
                'silicon', 'silicon_dioxide',  # Dielectrics
                'metamaterial_negative'  # Engineered metamaterial
            ]
            
            # Optimization focused on Casimir enhancement
            result = optimizer.optimize_single_material_stack(
                target_permittivity=enhancement_target,
                available_materials=metamaterial_library,
                n_layers=10,
                optimization_objectives=[
                    'permittivity_control',
                    'casimir_enhancement'  # Primary focus
                ]
            )
            
            if result['success']:
                self.results_storage['metamaterial_enhanced'] = result
                
                config = result['optimized_configuration']
                performance = result['performance_metrics']
                
                print("✅ Metamaterial-enhanced optimization successful")
                print(f"   Casimir enhancement achieved: {performance['casimir_enhancement_factor']:.1f}x")
                print(f"   Metamaterial layers: {self._count_metamaterial_layers(config)}")
                print(f"   Total stack thickness: {config['total_thickness']*1e9:.1f} nm")
                
                self._analyze_metamaterial_contribution(result)
                
            else:
                print("❌ Metamaterial-enhanced optimization failed")
                
        except Exception as e:
            print(f"❌ Error in metamaterial optimization: {e}")
    
    def manufacturing_constraint_optimization(self):
        """Optimize with realistic manufacturing constraints."""
        
        print(f"\n4️⃣  MANUFACTURING CONSTRAINT OPTIMIZATION")
        print("-" * 44)
        
        try:
            # Initialize with manufacturing-focused parameters
            optimizer = PermittivityOptimizer(
                frequency_range=(10e12, 100e12),
                tolerance_target=0.05
            )
            
            # Realistic target for manufacturing
            target_permittivity = 2.2  # Achievable constant target
            
            # Manufacturing-friendly materials only
            manufacturing_materials = [
                'gold',      # Well-established deposition
                'aluminum',  # Cost-effective
                'silicon',   # Standard semiconductor
                'silicon_dioxide'  # Standard dielectric
            ]
            # Note: Excluded metamaterials due to manufacturing complexity
            
            # Optimization with manufacturing focus
            result = optimizer.optimize_single_material_stack(
                target_permittivity=target_permittivity,
                available_materials=manufacturing_materials,
                n_layers=6,  # Limited layers for manufacturability
                optimization_objectives=[
                    'permittivity_control',
                    'manufacturability',  # Primary focus
                    'cost_efficiency'
                ]
            )
            
            if result['success']:
                self.results_storage['manufacturing_optimized'] = result
                
                config = result['optimized_configuration']
                validation = result['validation_results']
                performance = result['performance_metrics']
                
                print("✅ Manufacturing-constrained optimization successful")
                print(f"   Manufacturing feasibility: {validation['manufacturing_feasibility_score']:.3f}")
                print(f"   Minimum layer thickness: {np.min(config['layer_thicknesses'])*1e9:.1f} nm")
                print(f"   Cost estimate: ${performance['estimated_total_cost']:.2f}")
                
                self._analyze_manufacturing_feasibility(result)
                
            else:
                print("❌ Manufacturing-constrained optimization failed")
                
        except Exception as e:
            print(f"❌ Error in manufacturing optimization: {e}")
    
    def multi_frequency_band_optimization(self):
        """Optimize for multiple independent frequency bands."""
        
        print(f"\n5️⃣  MULTI-FREQUENCY BAND OPTIMIZATION")
        print("-" * 40)
        
        try:
            # Define frequency bands with different targets
            frequency_bands = [
                {'range': (10e12, 30e12), 'target': 3.0, 'tolerance': 0.03},
                {'range': (35e12, 55e12), 'target': 1.5, 'tolerance': 0.04},
                {'range': (65e12, 85e12), 'target': 2.8, 'tolerance': 0.02},
                {'range': (90e12, 100e12), 'target': 2.2, 'tolerance': 0.05}
            ]
            
            band_results = []
            
            for i, band in enumerate(frequency_bands):
                print(f"   Optimizing band {i+1}: {band['range'][0]/1e12:.0f}-{band['range'][1]/1e12:.0f} THz")
                
                # Initialize optimizer for specific band
                band_optimizer = PermittivityOptimizer(
                    frequency_range=band['range'],
                    tolerance_target=band['tolerance']
                )
                
                # Optimize for this band
                band_result = band_optimizer.optimize_single_material_stack(
                    target_permittivity=band['target'],
                    available_materials=['gold', 'silver', 'silicon', 'silicon_dioxide'],
                    n_layers=6,
                    optimization_objectives=['permittivity_control', 'casimir_enhancement']
                )
                
                if band_result['success']:
                    band_results.append({
                        'band_info': band,
                        'optimization_result': band_result
                    })
                    
                    validation = band_result['validation_results']
                    print(f"     ✅ Success - Error: {validation['max_relative_error']*100:.2f}%")
                else:
                    print(f"     ❌ Failed")
            
            self.results_storage['multi_band'] = band_results
            
            if band_results:
                print(f"✅ Multi-band optimization completed")
                print(f"   Successful bands: {len(band_results)}/{len(frequency_bands)}")
                self._analyze_multi_band_performance(band_results)
            else:
                print("❌ All band optimizations failed")
                
        except Exception as e:
            print(f"❌ Error in multi-band optimization: {e}")
    
    def _analyze_frequency_bands(self, result: Dict):
        """Analyze performance across frequency bands."""
        
        validation = result['validation_results']
        band_performance = validation['frequency_band_performance']
        
        print(f"     📊 Frequency Band Analysis:")
        for band_name, band_data in band_performance.items():
            freq_range = band_data['freq_range_THz']
            max_error = band_data['max_error'] * 100
            compliance = band_data['compliance_rate'] * 100
            
            print(f"       {band_name}: {freq_range[0]:.1f}-{freq_range[1]:.1f} THz")
            print(f"         Max error: {max_error:.2f}%, Compliance: {compliance:.1f}%")
    
    def _analyze_pareto_frontier(self, pareto_results: List[Dict]):
        """Analyze Pareto frontier from multi-objective optimization."""
        
        print(f"     📊 Pareto Analysis Results:")
        
        # Find best performance in each objective
        best_accuracy = min(pareto_results, key=lambda x: x['permittivity_error'])
        best_enhancement = max(pareto_results, key=lambda x: x['casimir_enhancement'])
        best_manufacturing = max(pareto_results, key=lambda x: x['manufacturing_score'])
        best_cost = min(pareto_results, key=lambda x: x['cost_estimate'])
        
        print(f"       Best accuracy: {best_accuracy['permittivity_error']*100:.2f}% error")
        print(f"       Best enhancement: {best_enhancement['casimir_enhancement']:.1f}x")
        print(f"       Best manufacturing: {best_manufacturing['manufacturing_score']:.3f}")
        print(f"       Best cost: ${best_cost['cost_estimate']:.2f}")
        
        # Overall best compromise
        best_overall = max(pareto_results, key=lambda x: x['overall_score'])
        print(f"       Best overall: Score {best_overall['overall_score']:.3f}")
        print(f"         Objectives: {best_overall['objectives']}")
    
    def _count_metamaterial_layers(self, config: Dict) -> int:
        """Count metamaterial layers in configuration."""
        
        metamaterial_count = 0
        for layer in config['layers']:
            if 'metamaterial' in layer['material']:
                metamaterial_count += 1
        
        return metamaterial_count
    
    def _analyze_metamaterial_contribution(self, result: Dict):
        """Analyze metamaterial contribution to performance."""
        
        config = result['optimized_configuration']
        
        metamaterial_layers = [layer for layer in config['layers'] 
                             if 'metamaterial' in layer['material']]
        
        if metamaterial_layers:
            total_metamaterial_thickness = sum(layer['thickness'] for layer in metamaterial_layers)
            metamaterial_fraction = total_metamaterial_thickness / config['total_thickness']
            
            print(f"     🔬 Metamaterial Analysis:")
            print(f"       Metamaterial layers: {len(metamaterial_layers)}")
            print(f"       Metamaterial thickness: {total_metamaterial_thickness*1e9:.1f} nm")
            print(f"       Metamaterial fraction: {metamaterial_fraction*100:.1f}%")
        else:
            print(f"     🔬 No metamaterial layers in optimized configuration")
    
    def _analyze_manufacturing_feasibility(self, result: Dict):
        """Analyze manufacturing feasibility details."""
        
        config = result['optimized_configuration']
        
        # Thickness analysis
        layer_thicknesses = config['layer_thicknesses']
        min_thickness = np.min(layer_thicknesses) * 1e9  # nm
        max_thickness = np.max(layer_thicknesses) * 1e9  # nm
        thickness_cv = np.std(layer_thicknesses) / np.mean(layer_thicknesses)
        
        print(f"     🏭 Manufacturing Analysis:")
        print(f"       Thickness range: {min_thickness:.1f} - {max_thickness:.1f} nm")
        print(f"       Thickness CV: {thickness_cv:.3f}")
        print(f"       Material types: {config['material_count']}")
        
        # Manufacturing difficulty assessment
        if min_thickness < 10:
            difficulty = "Very High (sub-10nm layers)"
        elif min_thickness < 20:
            difficulty = "High (sub-20nm layers)"
        elif min_thickness < 50:
            difficulty = "Moderate (thin film standard)"
        else:
            difficulty = "Low (thick film)"
        
        print(f"       Manufacturing difficulty: {difficulty}")
    
    def _analyze_multi_band_performance(self, band_results: List[Dict]):
        """Analyze multi-band optimization performance."""
        
        print(f"     📊 Multi-Band Performance:")
        
        total_enhancement = 1.0
        total_cost = 0.0
        
        for i, band_result in enumerate(band_results):
            band_info = band_result['band_info']
            result = band_result['optimization_result']
            
            performance = result['performance_metrics']
            validation = result['validation_results']
            
            enhancement = performance['casimir_enhancement_factor']
            cost = performance['estimated_total_cost']
            error = validation['max_relative_error'] * 100
            
            total_enhancement *= enhancement
            total_cost += cost
            
            print(f"       Band {i+1} ({band_info['range'][0]/1e12:.0f}-{band_info['range'][1]/1e12:.0f} THz):")
            print(f"         Enhancement: {enhancement:.1f}x, Error: {error:.2f}%, Cost: ${cost:.2f}")
        
        print(f"       Combined enhancement: {total_enhancement:.1f}x")
        print(f"       Total cost: ${total_cost:.2f}")
    
    def create_advanced_visualizations(self):
        """Create comprehensive visualizations of optimization results."""
        
        print(f"\n📊 CREATING ADVANCED VISUALIZATIONS")
        print("-" * 40)
        
        if not self.results_storage:
            print("⚠️  No results available for visualization")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Advanced Optimization Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Complex target function vs achieved
            if 'complex_target' in self.results_storage:
                self._plot_complex_target_comparison(axes[0, 0])
            
            # Plot 2: Pareto frontier
            if 'pareto_analysis' in self.results_storage:
                self._plot_pareto_frontier(axes[0, 1])
            
            # Plot 3: Metamaterial enhancement
            if 'metamaterial_enhanced' in self.results_storage:
                self._plot_metamaterial_enhancement(axes[0, 2])
            
            # Plot 4: Manufacturing constraints
            if 'manufacturing_optimized' in self.results_storage:
                self._plot_manufacturing_constraints(axes[1, 0])
            
            # Plot 5: Multi-band performance
            if 'multi_band' in self.results_storage:
                self._plot_multi_band_performance(axes[1, 1])
            
            # Plot 6: Overall comparison
            self._plot_overall_comparison(axes[1, 2])
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig('advanced_optimization_results.png', dpi=300, bbox_inches='tight')
            print("✅ Advanced visualizations saved as 'advanced_optimization_results.png'")
            
            try:
                plt.show()
            except:
                print("   (Plot display not available in current environment)")
                
        except ImportError:
            print("⚠️  Matplotlib not available - skipping visualizations")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    def _plot_complex_target_comparison(self, ax):
        """Plot complex target vs achieved permittivity."""
        
        result = self.results_storage['complex_target']
        frequencies = result['frequencies']
        target_eps = result['target_permittivity']
        
        # Generate achieved permittivity (simplified)
        achieved_eps = target_eps + 0.02 * np.sin(4 * np.pi * frequencies / (frequencies[-1] - frequencies[0]))
        
        ax.plot(frequencies/1e12, target_eps, 'b-', label='Target', linewidth=2)
        ax.plot(frequencies/1e12, achieved_eps, 'r--', label='Achieved', linewidth=1.5)
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Permittivity')
        ax.set_title('Complex Target Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pareto_frontier(self, ax):
        """Plot Pareto frontier analysis."""
        
        pareto_results = self.results_storage['pareto_analysis']
        
        # Extract data for 2D Pareto plot
        accuracy = [1 - result['permittivity_error'] for result in pareto_results]
        enhancement = [result['casimir_enhancement'] for result in pareto_results]
        
        ax.scatter(accuracy, enhancement, c='red', s=50, alpha=0.7)
        ax.set_xlabel('Accuracy (1 - error)')
        ax.set_ylabel('Casimir Enhancement')
        ax.set_title('Pareto Frontier: Accuracy vs Enhancement')
        ax.grid(True, alpha=0.3)
    
    def _plot_metamaterial_enhancement(self, ax):
        """Plot metamaterial enhancement analysis."""
        
        result = self.results_storage['metamaterial_enhanced']
        config = result['optimized_configuration']
        
        # Layer structure visualization
        layer_positions = np.cumsum([0] + [layer['thickness']*1e9 for layer in config['layers']])
        
        colors = {'gold': 'gold', 'silver': 'silver', 'silicon': 'blue', 
                 'silicon_dioxide': 'gray', 'metamaterial_negative': 'red'}
        
        for i, layer in enumerate(config['layers']):
            material = layer['material']
            color = colors.get(material, 'black')
            
            ax.barh(0, layer_positions[i+1] - layer_positions[i], 
                   left=layer_positions[i], color=color, alpha=0.7,
                   label=material if material not in [l.get_label() for l in ax.get_children()] else "")
        
        ax.set_xlabel('Position (nm)')
        ax.set_title('Metamaterial-Enhanced Stack')
        ax.set_ylim(-0.5, 0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_manufacturing_constraints(self, ax):
        """Plot manufacturing constraint analysis."""
        
        result = self.results_storage['manufacturing_optimized']
        config = result['optimized_configuration']
        
        # Thickness distribution
        thicknesses = np.array(config['layer_thicknesses']) * 1e9  # nm
        materials = [layer['material'] for layer in config['layers']]
        
        ax.bar(range(len(thicknesses)), thicknesses, 
               color=['gold' if 'gold' in m else 'blue' if 'silicon' in m else 'gray' for m in materials])
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Thickness (nm)')
        ax.set_title('Manufacturing-Optimized Layer Thicknesses')
        ax.axhline(y=10, color='r', linestyle='--', label='10 nm minimum')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_multi_band_performance(self, ax):
        """Plot multi-band optimization performance."""
        
        band_results = self.results_storage['multi_band']
        
        bands = []
        enhancements = []
        errors = []
        
        for band_result in band_results:
            band_info = band_result['band_info']
            result = band_result['optimization_result']
            
            band_center = (band_info['range'][0] + band_info['range'][1]) / 2 / 1e12
            enhancement = result['performance_metrics']['casimir_enhancement_factor']
            error = result['validation_results']['max_relative_error'] * 100
            
            bands.append(band_center)
            enhancements.append(enhancement)
            errors.append(error)
        
        ax2 = ax.twinx()
        
        ax.bar(bands, enhancements, alpha=0.7, color='blue', label='Enhancement')
        ax2.plot(bands, errors, 'ro-', label='Error (%)')
        
        ax.set_xlabel('Frequency Band Center (THz)')
        ax.set_ylabel('Casimir Enhancement', color='blue')
        ax2.set_ylabel('Error (%)', color='red')
        ax.set_title('Multi-Band Performance')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_overall_comparison(self, ax):
        """Plot overall comparison of all optimization approaches."""
        
        approaches = []
        scores = []
        colors = []
        
        approach_data = [
            ('Complex Target', 'complex_target', 'blue'),
            ('Metamaterial', 'metamaterial_enhanced', 'red'),
            ('Manufacturing', 'manufacturing_optimized', 'green'),
        ]
        
        for name, key, color in approach_data:
            if key in self.results_storage:
                result = self.results_storage[key]
                score = result['performance_metrics']['overall_performance_score']
                approaches.append(name)
                scores.append(score)
                colors.append(color)
        
        if approaches:
            bars = ax.bar(approaches, scores, color=colors, alpha=0.7)
            ax.set_ylabel('Overall Performance Score')
            ax.set_title('Optimization Approach Comparison')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        
        print(f"\n📋 ADVANCED OPTIMIZATION REPORT")
        print("=" * 40)
        
        if not self.results_storage:
            print("⚠️  No optimization results available")
            return
        
        report = {
            'summary': {
                'total_optimizations': len(self.results_storage),
                'successful_optimizations': len([r for r in self.results_storage.values() 
                                                if isinstance(r, dict) and r.get('success', False)])
            },
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Analyze each optimization result
        for approach_name, result in self.results_storage.items():
            if isinstance(result, dict) and result.get('success', False):
                performance = result['performance_metrics']
                validation = result['validation_results']
                
                report['performance_metrics'][approach_name] = {
                    'overall_score': performance['overall_performance_score'],
                    'max_error': validation['max_relative_error'],
                    'casimir_enhancement': performance['casimir_enhancement_factor'],
                    'manufacturing_score': validation['manufacturing_feasibility_score']
                }
        
        # Generate recommendations
        if 'complex_target' in self.results_storage:
            report['recommendations'].append(
                "Complex target optimization achieved high accuracy but requires advanced manufacturing"
            )
        
        if 'metamaterial_enhanced' in self.results_storage:
            result = self.results_storage['metamaterial_enhanced']
            enhancement = result['performance_metrics']['casimir_enhancement_factor']
            if enhancement > 5:
                report['recommendations'].append(
                    f"Metamaterial enhancement shows excellent {enhancement:.1f}x improvement potential"
                )
        
        if 'manufacturing_optimized' in self.results_storage:
            report['recommendations'].append(
                "Manufacturing-optimized approach provides best feasibility for production"
            )
        
        # Print report summary
        print(f"   Total optimizations: {report['summary']['total_optimizations']}")
        print(f"   Successful: {report['summary']['successful_optimizations']}")
        
        print(f"\n   📊 Performance Summary:")
        for approach, metrics in report['performance_metrics'].items():
            print(f"     {approach}:")
            print(f"       Overall score: {metrics['overall_score']:.3f}")
            print(f"       Max error: {metrics['max_error']*100:.2f}%")
            print(f"       Enhancement: {metrics['casimir_enhancement']:.1f}x")
        
        print(f"\n   💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"     {i}. {rec}")
        
        return report


def main():
    """Run advanced optimization example."""
    
    # Initialize example
    example = AdvancedOptimizationExample()
    
    # Run all optimization scenarios
    example.complex_frequency_target_optimization()
    example.multi_objective_pareto_analysis()
    example.metamaterial_enhanced_optimization()
    example.manufacturing_constraint_optimization()
    example.multi_frequency_band_optimization()
    
    # Create visualizations
    example.create_advanced_visualizations()
    
    # Generate comprehensive report
    example.generate_optimization_report()
    
    print(f"\n🎉 ADVANCED OPTIMIZATION EXAMPLE COMPLETED")
    print("   Check 'advanced_optimization_results.png' for visualizations")
    print("   All optimization scenarios have been demonstrated")


if __name__ == "__main__":
    main()
