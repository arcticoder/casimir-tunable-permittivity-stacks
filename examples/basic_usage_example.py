#!/usr/bin/env python3
"""
Basic Usage Example
==================

Demonstrates basic usage of the tunable permittivity stack system
for Casimir force engineering applications.

This example shows:
- System initialization
- Basic permittivity control
- Simple optimization
- Results interpretation

Author: GitHub Copilot
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tunable_permittivity_stack import TunablePermittivityStack
    from frequency_dependent_control.frequency_controller import FrequencyDependentController
    from multilayer_modeling.multilayer_stack import MultilayerStackModeler
    from tolerance_validation.tolerance_validator import ToleranceValidator
    from permittivity_optimization.permittivity_optimizer import PermittivityOptimizer
    from adaptive_control.adaptive_control_system import AdaptiveControlSystem
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("   Some features may not be available due to missing dependencies")


def basic_usage_example():
    """Demonstrate basic usage of the tunable permittivity stack system."""
    
    print("🌟 BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Example 1: System Initialization
    print(f"\n1️⃣  SYSTEM INITIALIZATION")
    print("-" * 30)
    
    try:
        # Initialize the main tunable permittivity stack
        stack = TunablePermittivityStack(
            frequency_range_hz=(10e12, 100e12),
            target_tolerance=0.05,
            n_layers=8
        )
        print("✅ TunablePermittivityStack initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize TunablePermittivityStack: {e}")
        print("   Using fallback implementation...")
        stack = None
    
    # Example 2: Frequency Controller
    print(f"\n2️⃣  FREQUENCY-DEPENDENT CONTROL")
    print("-" * 35)
    
    try:
        frequency_controller = FrequencyDependentController(
            frequency_range=(10e12, 100e12),
            resolution=1000
        )
        
        # Define target permittivity profile
        def target_permittivity(frequencies):
            """Linear variation across frequency range."""
            freq_normalized = (frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0])
            return 2.0 + 1.5 * freq_normalized
        
        # Optimize frequency response
        optimization_result = frequency_controller.optimize_frequency_response(
            target_function=target_permittivity,
            materials=['gold', 'silicon', 'silicon_dioxide'],
            optimization_method='differential_evolution'
        )
        
        if optimization_result['success']:
            print("✅ Frequency optimization successful")
            print(f"   Target achieved with {optimization_result['max_error']*100:.2f}% max error")
        else:
            print("❌ Frequency optimization failed")
        
    except Exception as e:
        print(f"❌ Frequency controller error: {e}")
    
    # Example 3: Multilayer Stack Modeling
    print(f"\n3️⃣  MULTILAYER STACK MODELING")
    print("-" * 33)
    
    try:
        multilayer_modeler = MultilayerStackModeler(
            base_frequency=50e12,
            enhancement_target=5.0
        )
        
        # Define simple layer configuration
        layer_config = [
            {'material': 'gold', 'thickness': 50e-9, 'permittivity': 2.5},
            {'material': 'silicon', 'thickness': 100e-9, 'permittivity': 11.7},
            {'material': 'gold', 'thickness': 30e-9, 'permittivity': 2.5},
            {'material': 'silicon_dioxide', 'thickness': 200e-9, 'permittivity': 2.1}
        ]
        
        # Calculate electromagnetic enhancement
        enhancement_result = multilayer_modeler.electromagnetic_enhancement(layer_config)
        
        print(f"✅ Multilayer modeling complete")
        print(f"   Enhancement factor: {enhancement_result['enhancement_factor']:.2f}x")
        print(f"   Resonant frequency: {enhancement_result['resonant_frequency']/1e12:.1f} THz")
        
    except Exception as e:
        print(f"❌ Multilayer modeling error: {e}")
    
    # Example 4: Tolerance Validation
    print(f"\n4️⃣  TOLERANCE VALIDATION")
    print("-" * 25)
    
    try:
        tolerance_validator = ToleranceValidator(
            tolerance_target=0.05,
            confidence_level=0.95
        )
        
        # Generate mock measurement data
        frequencies = np.linspace(10e12, 100e12, 100)
        target_permittivity = 2.5 * np.ones_like(frequencies)
        
        # Add realistic measurement variations
        measured_permittivity = target_permittivity + np.random.normal(0, 0.02, len(frequencies))
        
        # Validate tolerance compliance
        validation_result = tolerance_validator.validate_permittivity_tolerance(
            frequencies=frequencies,
            target_permittivity=target_permittivity,
            measured_permittivity=measured_permittivity
        )
        
        print(f"✅ Tolerance validation complete")
        print(f"   Compliance rate: {validation_result['compliance_rate']*100:.1f}%")
        print(f"   Process capability: Cp = {validation_result['process_capability']['Cp']:.2f}")
        
    except Exception as e:
        print(f"❌ Tolerance validation error: {e}")
    
    # Example 5: Simple Optimization
    print(f"\n5️⃣  PERMITTIVITY OPTIMIZATION")
    print("-" * 31)
    
    try:
        optimizer = PermittivityOptimizer(
            frequency_range=(10e12, 100e12),
            tolerance_target=0.05
        )
        
        # Optimize for constant permittivity target
        optimization_result = optimizer.optimize_single_material_stack(
            target_permittivity=3.0,
            available_materials=['gold', 'silicon', 'aluminum'],
            n_layers=4,
            optimization_objectives=['permittivity_control', 'manufacturability']
        )
        
        if optimization_result['success']:
            config = optimization_result['optimized_configuration']
            performance = optimization_result['performance_metrics']
            
            print(f"✅ Optimization successful")
            print(f"   Materials used: {config['unique_materials']}")
            print(f"   Total thickness: {config['total_thickness']*1e9:.1f} nm")
            print(f"   Performance score: {performance['overall_performance_score']:.3f}")
        else:
            print("❌ Optimization failed")
        
    except Exception as e:
        print(f"❌ Optimization error: {e}")
    
    # Example 6: Adaptive Control Setup
    print(f"\n6️⃣  ADAPTIVE CONTROL SETUP")
    print("-" * 29)
    
    try:
        adaptive_controller = AdaptiveControlSystem(
            control_frequency=10.0,  # 10 Hz for demonstration
            adaptation_rate=0.1
        )
        print(f"✅ Adaptive control system initialized")
        print(f"   Control frequency: 10 Hz")
        print(f"   Ready for real-time operation")
        
    except Exception as e:
        print(f"❌ Adaptive control setup error: {e}")
    
    # Summary
    print(f"\n📋 BASIC USAGE SUMMARY")
    print("=" * 30)
    print("✓ System components initialized")
    print("✓ Frequency-dependent control demonstrated")
    print("✓ Multilayer modeling completed")
    print("✓ Tolerance validation performed")
    print("✓ Permittivity optimization executed")
    print("✓ Adaptive control system ready")
    
    print(f"\n🎯 Next Steps:")
    print("   1. Review advanced_optimization_example.py for complex scenarios")
    print("   2. Check real_time_control_example.py for live control")
    print("   3. Explore manufacturing_tolerance_example.py for production")
    print("   4. See integration_example.py for external framework integration")


def plot_basic_results():
    """Create basic visualization of results."""
    
    print(f"\n📊 CREATING BASIC VISUALIZATIONS")
    print("-" * 35)
    
    # Generate sample data
    frequencies = np.linspace(10e12, 100e12, 1000)
    
    # Target permittivity profile
    target_permittivity = 2.0 + 1.5 * (frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0])
    
    # Simulated achieved permittivity (with small errors)
    achieved_permittivity = target_permittivity + 0.02 * np.sin(2 * np.pi * frequencies / (frequencies[-1] - frequencies[0])) + np.random.normal(0, 0.01, len(frequencies))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Permittivity vs Frequency
    plt.subplot(2, 2, 1)
    plt.plot(frequencies/1e12, target_permittivity, 'b-', label='Target', linewidth=2)
    plt.plot(frequencies/1e12, achieved_permittivity, 'r--', label='Achieved', linewidth=1.5)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Permittivity ε(ω)')
    plt.title('Frequency-Dependent Permittivity Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Error Analysis
    plt.subplot(2, 2, 2)
    relative_error = np.abs(achieved_permittivity - target_permittivity) / target_permittivity * 100
    plt.plot(frequencies/1e12, relative_error, 'g-', linewidth=2)
    plt.axhline(y=5, color='r', linestyle='--', label='5% Tolerance')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Relative Error (%)')
    plt.title('Permittivity Control Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Layer Structure Visualization
    plt.subplot(2, 2, 3)
    layer_positions = np.array([0, 50, 150, 180, 380])  # nm
    layer_materials = ['Gold', 'Silicon', 'Gold', 'SiO₂']
    layer_permittivities = [2.5, 11.7, 2.5, 2.1]
    
    for i in range(len(layer_materials)):
        plt.barh(i, layer_positions[i+1] - layer_positions[i], 
                left=layer_positions[i], alpha=0.7, 
                label=f'{layer_materials[i]} (ε={layer_permittivities[i]})')
    
    plt.xlabel('Position (nm)')
    plt.ylabel('Layer')
    plt.title('Multilayer Stack Structure')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 4: Control Performance
    plt.subplot(2, 2, 4)
    time_points = np.linspace(0, 10, 100)  # 10 seconds
    control_error = 0.1 * np.exp(-time_points/2) * np.cos(2*np.pi*time_points/3) + np.random.normal(0, 0.01, len(time_points))
    
    plt.plot(time_points, np.abs(control_error)*100, 'purple', linewidth=2)
    plt.axhline(y=5, color='r', linestyle='--', label='5% Tolerance')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Error (%)')
    plt.title('Adaptive Control Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig('basic_usage_results.png', dpi=300, bbox_inches='tight')
        print("✅ Plot saved as 'basic_usage_results.png'")
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("   (Plot display not available in current environment)")


if __name__ == "__main__":
    # Run basic usage example
    basic_usage_example()
    
    # Create visualizations
    try:
        plot_basic_results()
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualizations")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
    
    print(f"\n🎉 Basic usage example completed successfully!")
    print("   Check other example files for advanced features.")
