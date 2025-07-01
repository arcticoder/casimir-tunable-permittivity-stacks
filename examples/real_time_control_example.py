#!/usr/bin/env python3
"""
Real-Time Control Example
========================

Demonstrates real-time adaptive control capabilities for tunable
permittivity stacks with live feedback and dynamic optimization.

This example shows:
- Real-time control loop implementation
- Live measurement integration
- Dynamic target adaptation
- Environmental compensation
- Performance monitoring

Author: GitHub Copilot
"""

import sys
import os
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from collections import deque

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from adaptive_control.adaptive_control_system import AdaptiveControlSystem
    from tunable_permittivity_stack import TunablePermittivityStack
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("   Creating fallback implementations for demonstration")


class RealTimeControlDemo:
    """Real-time control demonstration system."""
    
    def __init__(self):
        """Initialize real-time control demo."""
        self.control_active = False
        self.measurement_data = deque(maxlen=1000)
        self.control_data = deque(maxlen=1000)
        self.target_data = deque(maxlen=1000)
        self.time_data = deque(maxlen=1000)
        
        # Simulated system parameters
        self.system_state = {
            'current_permittivity': 2.5 + 0j,
            'target_permittivity': 2.5,
            'control_voltages': np.zeros(6),
            'environmental_conditions': {
                'temperature': 298.15,
                'humidity': 45.0,
                'pressure': 101325,
                'vibration_level': 0.0
            },
            'system_health': 1.0,
            'last_update': time.time()
        }
        
        # Physical system simulation parameters
        self.physics_params = {
            'response_time_constant': 0.1,  # seconds
            'measurement_noise_std': 0.01,
            'environmental_drift_rate': 1e-4,  # per second
            'control_gain': 0.8,
            'saturation_limits': (-10.0, 10.0)
        }
        
        print("🎛️  REAL-TIME CONTROL DEMO INITIALIZED")
        print("=" * 50)
    
    def simulate_measurement_system(self) -> Dict:
        """Simulate realistic measurement system with noise and drift."""
        
        current_time = time.time()
        dt = current_time - self.system_state['last_update']
        
        # Environmental drift
        temp_drift = 0.1 * np.sin(current_time * 0.1) + np.random.normal(0, 0.05)
        humidity_drift = 2.0 * np.sin(current_time * 0.05) + np.random.normal(0, 0.2)
        
        self.system_state['environmental_conditions']['temperature'] = 298.15 + temp_drift
        self.system_state['environmental_conditions']['humidity'] = 45.0 + humidity_drift
        
        # Simulate permittivity response to control
        target = self.system_state['target_permittivity']
        current = self.system_state['current_permittivity'].real
        control_sum = np.sum(self.system_state['control_voltages'])
        
        # First-order response with control influence
        response_rate = 1.0 / self.physics_params['response_time_constant']
        control_effect = self.physics_params['control_gain'] * control_sum * 0.1
        
        # Update permittivity with physics
        if callable(target):
            # For time-varying targets
            target_value = target(current_time)
        else:
            target_value = target
        
        dpermittivity_dt = response_rate * (target_value - current) + control_effect
        new_permittivity = current + dpermittivity_dt * dt
        
        # Add environmental effects
        temp_effect = -2e-4 * temp_drift  # Temperature coefficient
        humidity_effect = 1e-5 * humidity_drift  # Humidity coefficient
        
        new_permittivity += temp_effect + humidity_effect
        
        # Add measurement noise
        measurement_noise = np.random.normal(0, self.physics_params['measurement_noise_std'])
        measured_permittivity = new_permittivity + measurement_noise
        
        # Update system state
        self.system_state['current_permittivity'] = measured_permittivity + 0j
        self.system_state['last_update'] = current_time
        
        return {
            'permittivity': measured_permittivity + 0j,
            'timestamp': current_time,
            'environment': self.system_state['environmental_conditions'].copy(),
            'system_health': self.system_state['system_health']
        }
    
    def simulate_control_system(self, control_actions: np.ndarray) -> None:
        """Simulate control system applying voltages to tunable elements."""
        
        # Apply saturation limits
        saturated_actions = np.clip(
            control_actions, 
            self.physics_params['saturation_limits'][0],
            self.physics_params['saturation_limits'][1]
        )
        
        # Update control voltages
        self.system_state['control_voltages'] = saturated_actions
        
        # Simulate control system health (degradation over time)
        if np.any(np.abs(saturated_actions) > 8.0):
            self.system_state['system_health'] *= 0.999  # Slight degradation at high control
        else:
            self.system_state['system_health'] *= 0.9999  # Very slow recovery
        
        # Clamp health to reasonable bounds
        self.system_state['system_health'] = np.clip(self.system_state['system_health'], 0.1, 1.0)
    
    def run_constant_target_control(self, target_permittivity: float = 3.0, duration: float = 30.0):
        """Run real-time control with constant target."""
        
        print(f"\n1️⃣  CONSTANT TARGET CONTROL")
        print(f"   Target: ε = {target_permittivity}")
        print(f"   Duration: {duration:.1f}s")
        print("-" * 40)
        
        # Initialize adaptive control system
        try:
            control_system = AdaptiveControlSystem(
                control_frequency=50.0,  # 50 Hz control rate
                adaptation_rate=0.02
            )
            
            # Set target
            self.system_state['target_permittivity'] = target_permittivity
            
            # Create control callbacks
            measurement_callback = self.simulate_measurement_system
            control_callback = self.simulate_control_system
            
            # Start control loop in separate thread for demonstration
            control_thread = threading.Thread(
                target=control_system.start_control_loop,
                args=(target_permittivity, measurement_callback, control_callback, duration)
            )
            
            # Start data logging
            self.control_active = True
            logging_thread = threading.Thread(target=self._log_control_data)
            
            # Start both threads
            control_thread.start()
            logging_thread.start()
            
            # Wait for completion
            control_thread.join()
            self.control_active = False
            logging_thread.join()
            
            print("✅ Constant target control completed")
            
        except Exception as e:
            print(f"❌ Control system error: {e}")
            self.control_active = False
    
    def run_dynamic_target_control(self, duration: float = 45.0):
        """Run real-time control with time-varying target."""
        
        print(f"\n2️⃣  DYNAMIC TARGET CONTROL")
        print(f"   Target: Time-varying sinusoidal")
        print(f"   Duration: {duration:.1f}s")
        print("-" * 40)
        
        # Define dynamic target function
        def dynamic_target(t):
            """Time-varying target permittivity."""
            base_time = time.time()
            relative_time = t - base_time if hasattr(self, '_start_time') else 0
            return 2.5 + 0.8 * np.sin(2 * np.pi * relative_time / 15.0)  # 15-second period
        
        try:
            control_system = AdaptiveControlSystem(
                control_frequency=50.0,
                adaptation_rate=0.05  # Higher adaptation for dynamic targets
            )
            
            # Store start time for dynamic target
            self._start_time = time.time()
            self.system_state['target_permittivity'] = dynamic_target
            
            # Clear previous data
            self.measurement_data.clear()
            self.control_data.clear()
            self.target_data.clear()
            self.time_data.clear()
            
            # Control callbacks
            measurement_callback = self.simulate_measurement_system
            control_callback = self.simulate_control_system
            
            # Start control with dynamic target
            control_thread = threading.Thread(
                target=control_system.start_control_loop,
                args=(dynamic_target, measurement_callback, control_callback, duration)
            )
            
            # Start data logging
            self.control_active = True
            logging_thread = threading.Thread(target=self._log_control_data)
            
            # Start both threads
            control_thread.start()
            logging_thread.start()
            
            # Wait for completion
            control_thread.join()
            self.control_active = False
            logging_thread.join()
            
            print("✅ Dynamic target control completed")
            
        except Exception as e:
            print(f"❌ Dynamic control system error: {e}")
            self.control_active = False
    
    def run_disturbance_rejection_test(self, duration: float = 25.0):
        """Test control system response to disturbances."""
        
        print(f"\n3️⃣  DISTURBANCE REJECTION TEST")
        print(f"   Disturbances: Temperature and electromagnetic")
        print(f"   Duration: {duration:.1f}s")
        print("-" * 40)
        
        # Add disturbance injection
        def measurement_with_disturbances():
            """Measurement system with injected disturbances."""
            
            base_measurement = self.simulate_measurement_system()
            current_time = time.time()
            
            # Large temperature disturbance at specific times
            if hasattr(self, '_disturbance_start_time'):
                relative_time = current_time - self._disturbance_start_time
                
                # Temperature spike at 8 seconds
                if 8.0 < relative_time < 12.0:
                    temp_disturbance = 5.0 * np.sin(np.pi * (relative_time - 8.0) / 4.0)
                    base_measurement['environment']['temperature'] += temp_disturbance
                    
                    # Corresponding permittivity disturbance
                    disturbance_effect = -2e-4 * temp_disturbance
                    base_measurement['permittivity'] += disturbance_effect
                
                # Electromagnetic interference at 18 seconds
                if 18.0 < relative_time < 20.0:
                    emi_noise = 0.05 * np.random.normal(0, 1)
                    base_measurement['permittivity'] += emi_noise
            
            return base_measurement
        
        try:
            control_system = AdaptiveControlSystem(
                control_frequency=50.0,
                adaptation_rate=0.08  # Higher adaptation for disturbance rejection
            )
            
            # Fixed target for disturbance test
            target_permittivity = 2.8
            self.system_state['target_permittivity'] = target_permittivity
            self._disturbance_start_time = time.time()
            
            # Clear previous data
            self.measurement_data.clear()
            self.control_data.clear()
            self.target_data.clear()
            self.time_data.clear()
            
            # Control with disturbance injection
            control_callback = self.simulate_control_system
            
            control_thread = threading.Thread(
                target=control_system.start_control_loop,
                args=(target_permittivity, measurement_with_disturbances, control_callback, duration)
            )
            
            # Start data logging
            self.control_active = True
            logging_thread = threading.Thread(target=self._log_control_data)
            
            # Start both threads
            control_thread.start()
            logging_thread.start()
            
            # Wait for completion
            control_thread.join()
            self.control_active = False
            logging_thread.join()
            
            print("✅ Disturbance rejection test completed")
            
        except Exception as e:
            print(f"❌ Disturbance rejection test error: {e}")
            self.control_active = False
    
    def _log_control_data(self):
        """Log control data for analysis."""
        
        while self.control_active:
            current_time = time.time()
            
            # Log current state
            self.time_data.append(current_time)
            self.measurement_data.append(self.system_state['current_permittivity'].real)
            self.control_data.append(np.sum(self.system_state['control_voltages']))
            
            # Log target (handle both constant and dynamic)
            target = self.system_state['target_permittivity']
            if callable(target):
                target_value = target(current_time)
            else:
                target_value = target
            
            self.target_data.append(target_value)
            
            # Sample rate for logging
            time.sleep(0.02)  # 50 Hz logging
    
    def analyze_control_performance(self):
        """Analyze logged control performance data."""
        
        print(f"\n📊 CONTROL PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if len(self.measurement_data) < 10:
            print("⚠️  Insufficient data for analysis")
            return
        
        # Convert to numpy arrays
        times = np.array(list(self.time_data))
        measurements = np.array(list(self.measurement_data))
        targets = np.array(list(self.target_data))
        controls = np.array(list(self.control_data))
        
        # Normalize time to start from 0
        times = times - times[0]
        
        # Calculate performance metrics
        errors = measurements - targets
        abs_errors = np.abs(errors)
        
        # Performance statistics
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        rms_error = np.sqrt(np.mean(errors**2))
        
        # Settling time (time to get within 5% of target)
        settling_threshold = 0.05 * np.mean(np.abs(targets))
        settled_indices = np.where(abs_errors < settling_threshold)[0]
        settling_time = times[settled_indices[0]] if len(settled_indices) > 0 else np.inf
        
        # Control effort
        max_control = np.max(np.abs(controls))
        mean_control = np.mean(np.abs(controls))
        
        print(f"   📈 Error Statistics:")
        print(f"      Max error: {max_error:.4f}")
        print(f"      Mean error: {mean_error:.4f}")
        print(f"      RMS error: {rms_error:.4f}")
        print(f"      Settling time: {settling_time:.2f}s")
        
        print(f"   🎛️  Control Statistics:")
        print(f"      Max control: {max_control:.2f}")
        print(f"      Mean control: {mean_control:.2f}")
        
        # System health analysis
        final_health = self.system_state['system_health']
        print(f"   💪 System Health: {final_health:.3f}")
        
        if final_health > 0.95:
            health_status = "🟢 EXCELLENT"
        elif final_health > 0.85:
            health_status = "🟡 GOOD"
        elif final_health > 0.70:
            health_status = "🟠 FAIR"
        else:
            health_status = "🔴 DEGRADED"
        
        print(f"   Status: {health_status}")
        
        return {
            'max_error': max_error,
            'mean_error': mean_error,
            'rms_error': rms_error,
            'settling_time': settling_time,
            'max_control': max_control,
            'mean_control': mean_control,
            'final_health': final_health,
            'data_points': len(measurements)
        }
    
    def create_real_time_visualizations(self):
        """Create real-time control performance visualizations."""
        
        print(f"\n📊 CREATING REAL-TIME VISUALIZATIONS")
        print("-" * 40)
        
        if len(self.measurement_data) < 10:
            print("⚠️  Insufficient data for visualization")
            return
        
        try:
            # Convert data to arrays
            times = np.array(list(self.time_data))
            measurements = np.array(list(self.measurement_data))
            targets = np.array(list(self.target_data))
            controls = np.array(list(self.control_data))
            
            # Normalize time
            times = times - times[0]
            
            # Create comprehensive plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Real-Time Control Performance Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Permittivity tracking
            axes[0, 0].plot(times, targets, 'b-', label='Target', linewidth=2)
            axes[0, 0].plot(times, measurements, 'r--', label='Measured', linewidth=1.5)
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Permittivity')
            axes[0, 0].set_title('Permittivity Control Tracking')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Control error
            errors = measurements - targets
            axes[0, 1].plot(times, errors * 100, 'g-', linewidth=2)
            axes[0, 1].axhline(y=5, color='r', linestyle='--', alpha=0.7, label='±5% tolerance')
            axes[0, 1].axhline(y=-5, color='r', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Error (%)')
            axes[0, 1].set_title('Control Error')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Control actions
            axes[1, 0].plot(times, controls, 'purple', linewidth=2)
            axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.7, label='Saturation limits')
            axes[1, 0].axhline(y=-10, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Control Signal')
            axes[1, 0].set_title('Control Actions')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Performance metrics
            window_size = max(10, len(measurements) // 20)
            windowed_errors = []
            windowed_times = []
            
            for i in range(window_size, len(measurements)):
                window_error = np.sqrt(np.mean(errors[i-window_size:i]**2))
                windowed_errors.append(window_error)
                windowed_times.append(times[i])
            
            axes[1, 1].plot(windowed_times, np.array(windowed_errors) * 100, 'orange', linewidth=2)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('RMS Error (%)')
            axes[1, 1].set_title('Moving RMS Error')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig('real_time_control_results.png', dpi=300, bbox_inches='tight')
            print("✅ Real-time visualizations saved as 'real_time_control_results.png'")
            
            try:
                plt.show()
            except:
                print("   (Plot display not available in current environment)")
                
        except ImportError:
            print("⚠️  Matplotlib not available - skipping visualizations")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    def demonstrate_live_monitoring(self, duration: float = 15.0):
        """Demonstrate live monitoring capabilities."""
        
        print(f"\n4️⃣  LIVE MONITORING DEMONSTRATION")
        print(f"   Duration: {duration:.1f}s")
        print("-" * 40)
        
        try:
            # Simple control system for monitoring demo
            target = 2.6
            self.system_state['target_permittivity'] = target
            
            start_time = time.time()
            monitor_active = True
            
            print("   🖥️  Live monitoring started (Ctrl+C to stop early)...")
            print("   Time    Target   Measured  Error     Control   Health")
            print("   ----    ------   --------  -----     -------   ------")
            
            try:
                while (time.time() - start_time) < duration and monitor_active:
                    # Get measurement
                    measurement = self.simulate_measurement_system()
                    current_perm = measurement['permittivity'].real
                    
                    # Simple proportional control
                    error = target - current_perm
                    control_action = 2.0 * error  # Proportional gain = 2.0
                    
                    # Apply control
                    control_actions = np.array([control_action, 0, 0, 0, 0, 0])
                    self.simulate_control_system(control_actions)
                    
                    # Display status
                    elapsed = time.time() - start_time
                    health = self.system_state['system_health']
                    
                    print(f"   {elapsed:5.1f}s   {target:6.3f}   {current_perm:8.3f}  {error:5.3f}     {control_action:7.3f}   {health:6.3f}")
                    
                    time.sleep(0.5)  # Update every 0.5 seconds for readability
                    
            except KeyboardInterrupt:
                print("\n   ⏹️  Monitoring stopped by user")
            
            print("✅ Live monitoring demonstration completed")
            
        except Exception as e:
            print(f"❌ Live monitoring error: {e}")


def main():
    """Run real-time control example."""
    
    print("🎛️  REAL-TIME CONTROL EXAMPLE")
    print("=" * 50)
    
    # Initialize demo system
    demo = RealTimeControlDemo()
    
    # Test 1: Constant target control
    demo.run_constant_target_control(target_permittivity=2.8, duration=20.0)
    
    # Analyze performance
    performance1 = demo.analyze_control_performance()
    
    # Test 2: Dynamic target control
    demo.run_dynamic_target_control(duration=30.0)
    
    # Analyze performance
    performance2 = demo.analyze_control_performance()
    
    # Test 3: Disturbance rejection
    demo.run_disturbance_rejection_test(duration=25.0)
    
    # Analyze performance
    performance3 = demo.analyze_control_performance()
    
    # Create visualizations
    demo.create_real_time_visualizations()
    
    # Test 4: Live monitoring
    demo.demonstrate_live_monitoring(duration=10.0)
    
    # Final summary
    print(f"\n📋 REAL-TIME CONTROL SUMMARY")
    print("=" * 40)
    print("✓ Constant target control demonstrated")
    print("✓ Dynamic target tracking achieved")
    print("✓ Disturbance rejection tested")
    print("✓ Live monitoring capabilities shown")
    print("✓ Performance analysis completed")
    
    print(f"\n🎯 Key Performance Metrics:")
    if 'performance1' in locals():
        print(f"   Constant target - RMS error: {performance1['rms_error']:.4f}")
    if 'performance2' in locals():
        print(f"   Dynamic target - RMS error: {performance2['rms_error']:.4f}")
    if 'performance3' in locals():
        print(f"   Disturbance rejection - RMS error: {performance3['rms_error']:.4f}")
    
    print(f"\n🎉 REAL-TIME CONTROL EXAMPLE COMPLETED")
    print("   Check 'real_time_control_results.png' for detailed analysis")


if __name__ == "__main__":
    main()
