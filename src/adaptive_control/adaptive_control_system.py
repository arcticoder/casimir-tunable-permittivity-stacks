#!/usr/bin/env python3
"""
Adaptive Control System
=======================

Real-time adaptive control for tunable permittivity stacks with dynamic
feedback, environmental compensation, and autonomous optimization.

Mathematical Foundation:
- PID control with adaptive gain scheduling
- Kalman filtering for state estimation
- Model predictive control (MPC) for optimization
- Environmental compensation algorithms

Features:
- Real-time frequency response adjustment
- Manufacturing tolerance compensation
- Environmental drift correction
- Autonomous optimization loops

Author: GitHub Copilot
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.constants import c, epsilon_0, pi, hbar
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import warnings
import time


class AdaptiveControlSystem:
    """
    Adaptive control system for real-time tunable permittivity management.
    
    Features:
    - Real-time feedback control
    - Environmental compensation
    - Manufacturing tolerance adaptation
    - Autonomous optimization
    """
    
    def __init__(self, control_frequency: float = 1000.0,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive control system.
        
        Args:
            control_frequency: Control loop frequency [Hz]
            adaptation_rate: Adaptation learning rate
        """
        self.control_frequency = control_frequency
        self.adaptation_rate = adaptation_rate
        self.control_period = 1.0 / control_frequency
        
        # Control system state
        self.system_state = {
            'permittivity_target': None,
            'permittivity_measured': None,
            'control_voltages': np.zeros(10),  # Max 10 control channels
            'environmental_parameters': {},
            'system_health': 1.0,
            'control_active': False,
            'last_update_time': 0
        }
        
        # PID controller parameters
        self.pid_controllers = {}
        self.initialize_pid_controllers()
        
        # Kalman filter for state estimation
        self.kalman_filter = self.initialize_kalman_filter()
        
        # Environmental compensation models
        self.environmental_models = self.initialize_environmental_models()
        
        # Adaptive parameters
        self.adaptive_parameters = {
            'learning_rate': adaptation_rate,
            'adaptation_memory': 1000,  # Number of samples to remember
            'performance_history': [],
            'parameter_history': [],
            'optimization_schedule': 'continuous'
        }
        
        # Control history for analysis
        self.control_history = {
            'timestamps': [],
            'permittivity_targets': [],
            'permittivity_measured': [],
            'control_actions': [],
            'errors': [],
            'environmental_data': []
        }
        
        print(f"✅ AdaptiveControlSystem initialized")
        print(f"   🔄 Control frequency: {control_frequency} Hz")
        print(f"   📈 Adaptation rate: {adaptation_rate}")
    
    def initialize_pid_controllers(self) -> None:
        """Initialize PID controllers for each frequency band."""
        
        # Frequency bands for control
        frequency_bands = [
            (10e12, 20e12),   # Band 1: 10-20 THz
            (20e12, 35e12),   # Band 2: 20-35 THz
            (35e12, 50e12),   # Band 3: 35-50 THz
            (50e12, 70e12),   # Band 4: 50-70 THz
            (70e12, 85e12),   # Band 5: 70-85 THz
            (85e12, 100e12)   # Band 6: 85-100 THz
        ]
        
        for i, (f_min, f_max) in enumerate(frequency_bands):
            controller_id = f"band_{i+1}"
            
            # Adaptive PID parameters
            self.pid_controllers[controller_id] = {
                'frequency_range': (f_min, f_max),
                'kp': 1.0,  # Proportional gain
                'ki': 0.1,  # Integral gain
                'kd': 0.05, # Derivative gain
                'integral_error': 0.0,
                'previous_error': 0.0,
                'error_history': [],
                'gain_adaptation_active': True,
                'performance_metric': 0.0
            }
        
        print(f"   🎛️  Initialized {len(frequency_bands)} PID controllers")
    
    def initialize_kalman_filter(self) -> Dict:
        """Initialize Kalman filter for state estimation."""
        
        # State vector: [ε_real, ε_imag, ε_dot_real, ε_dot_imag]
        n_states = 4
        n_observations = 2  # Real and imaginary permittivity measurements
        
        kalman_params = {
            'n_states': n_states,
            'n_observations': n_observations,
            
            # State estimate and covariance
            'x_est': np.zeros(n_states),
            'P_est': np.eye(n_states) * 1.0,
            
            # Process model (simple kinematic model)
            'F': np.array([
                [1, 0, self.control_period, 0],
                [0, 1, 0, self.control_period],
                [0, 0, 0.95, 0],  # Some damping on derivatives
                [0, 0, 0, 0.95]
            ]),
            
            # Observation model
            'H': np.array([
                [1, 0, 0, 0],  # Observe real permittivity
                [0, 1, 0, 0]   # Observe imaginary permittivity
            ]),
            
            # Process noise covariance
            'Q': np.eye(n_states) * 0.01,
            
            # Measurement noise covariance
            'R': np.eye(n_observations) * 0.05,
            
            # Filter status
            'initialized': False,
            'prediction_count': 0,
            'update_count': 0
        }
        
        return kalman_params
    
    def initialize_environmental_models(self) -> Dict:
        """Initialize environmental compensation models."""
        
        models = {
            'temperature_compensation': {
                'coefficient': -2e-4,  # Permittivity change per K
                'reference_temperature': 298.15,  # K
                'measurement_history': [],
                'model_active': True
            },
            
            'humidity_compensation': {
                'coefficient': 1e-5,   # Permittivity change per %RH
                'reference_humidity': 45.0,  # %RH
                'measurement_history': [],
                'model_active': True
            },
            
            'pressure_compensation': {
                'coefficient': 5e-7,   # Permittivity change per Pa
                'reference_pressure': 101325,  # Pa
                'measurement_history': [],
                'model_active': True
            },
            
            'electromagnetic_interference': {
                'detection_threshold': 1e-3,
                'filtering_active': True,
                'interference_history': [],
                'adaptive_filtering': True
            }
        }
        
        return models
    
    def start_control_loop(self, target_permittivity: Union[float, np.ndarray, Callable],
                          measurement_callback: Callable,
                          control_callback: Callable,
                          duration: float = np.inf) -> None:
        """
        Start the adaptive control loop.
        
        Args:
            target_permittivity: Target permittivity profile
            measurement_callback: Function to get current measurements
            control_callback: Function to apply control actions
            duration: Control duration [s], infinite by default
        """
        print(f"🚀 STARTING ADAPTIVE CONTROL LOOP")
        print(f"   ⏱️  Control frequency: {self.control_frequency} Hz")
        print(f"   ⏰ Duration: {'∞' if duration == np.inf else f'{duration:.1f}s'}")
        
        self.system_state['control_active'] = True
        self.system_state['permittivity_target'] = target_permittivity
        
        start_time = time.time()
        loop_count = 0
        
        try:
            while (time.time() - start_time) < duration and self.system_state['control_active']:
                loop_start = time.time()
                
                # Execute control cycle
                self.execute_control_cycle(measurement_callback, control_callback)
                
                # Adaptive parameter updates
                if loop_count % 10 == 0:  # Every 10 cycles
                    self.update_adaptive_parameters()
                
                # Environmental compensation
                if loop_count % 50 == 0:  # Every 50 cycles
                    self.update_environmental_compensation()
                
                # Control timing
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.control_period - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                loop_count += 1
                
                # Performance reporting (every second)
                if loop_count % int(self.control_frequency) == 0:
                    self.report_control_performance()
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Control loop interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Control loop error: {e}")
        
        finally:
            self.system_state['control_active'] = False
            print(f"🏁 Control loop stopped after {loop_count} cycles")
            self.generate_control_summary()
    
    def execute_control_cycle(self, measurement_callback: Callable,
                            control_callback: Callable) -> None:
        """Execute single control cycle."""
        
        current_time = time.time()
        
        # Get current measurements
        try:
            measurements = measurement_callback()
            self.system_state['permittivity_measured'] = measurements['permittivity']
            self.system_state['environmental_parameters'] = measurements.get('environment', {})
        except Exception as e:
            print(f"⚠️  Measurement error: {e}")
            return
        
        # State estimation using Kalman filter
        estimated_state = self.update_kalman_filter(measurements)
        
        # Calculate control errors
        control_errors = self.calculate_control_errors(estimated_state)
        
        # Generate control actions
        control_actions = self.generate_control_actions(control_errors)
        
        # Apply environmental compensation
        compensated_actions = self.apply_environmental_compensation(control_actions)
        
        # Apply control actions
        try:
            control_callback(compensated_actions)
            self.system_state['control_voltages'] = compensated_actions
        except Exception as e:
            print(f"⚠️  Control application error: {e}")
            return
        
        # Update control history
        self.update_control_history(current_time, control_errors, compensated_actions)
        
        self.system_state['last_update_time'] = current_time
    
    def update_kalman_filter(self, measurements: Dict) -> np.ndarray:
        """Update Kalman filter with new measurements."""
        
        kf = self.kalman_filter
        
        if not kf['initialized']:
            # Initialize with first measurement
            permittivity = measurements['permittivity']
            if isinstance(permittivity, complex):
                kf['x_est'][0] = permittivity.real
                kf['x_est'][1] = permittivity.imag
            else:
                kf['x_est'][0] = float(permittivity)
                kf['x_est'][1] = 0.0
            
            kf['initialized'] = True
            return kf['x_est']
        
        # Prediction step
        x_pred = kf['F'] @ kf['x_est']
        P_pred = kf['F'] @ kf['P_est'] @ kf['F'].T + kf['Q']
        
        # Update step
        permittivity = measurements['permittivity']
        if isinstance(permittivity, complex):
            z = np.array([permittivity.real, permittivity.imag])
        else:
            z = np.array([float(permittivity), 0.0])
        
        # Innovation
        y = z - kf['H'] @ x_pred
        S = kf['H'] @ P_pred @ kf['H'].T + kf['R']
        
        # Kalman gain
        K = P_pred @ kf['H'].T @ np.linalg.inv(S)
        
        # Update estimates
        kf['x_est'] = x_pred + K @ y
        kf['P_est'] = (np.eye(kf['n_states']) - K @ kf['H']) @ P_pred
        
        kf['prediction_count'] += 1
        kf['update_count'] += 1
        
        return kf['x_est']
    
    def calculate_control_errors(self, state: np.ndarray) -> Dict:
        """Calculate control errors for each frequency band."""
        
        errors = {}
        
        # Current permittivity from state estimate
        current_permittivity = complex(state[0], state[1])
        
        # Target permittivity
        target = self.system_state['permittivity_target']
        if callable(target):
            # For frequency-dependent targets, use average error
            frequencies = np.linspace(10e12, 100e12, 100)
            target_values = target(frequencies)
            target_permittivity = np.mean(target_values)
        elif isinstance(target, (int, float)):
            target_permittivity = target
        else:
            target_permittivity = np.mean(target)
        
        # Calculate errors for each PID controller
        for controller_id, controller in self.pid_controllers.items():
            error = target_permittivity - current_permittivity.real
            
            errors[controller_id] = {
                'proportional_error': error,
                'integral_error': controller['integral_error'] + error * self.control_period,
                'derivative_error': (error - controller['previous_error']) / self.control_period,
                'total_error': error
            }
            
            # Update controller state
            controller['previous_error'] = error
            controller['integral_error'] = errors[controller_id]['integral_error']
            controller['error_history'].append(error)
            
            # Limit error history
            if len(controller['error_history']) > 100:
                controller['error_history'] = controller['error_history'][-100:]
        
        return errors
    
    def generate_control_actions(self, errors: Dict) -> np.ndarray:
        """Generate control actions using PID controllers."""
        
        control_actions = np.zeros(10)  # 10 control channels
        
        for i, (controller_id, controller) in enumerate(self.pid_controllers.items()):
            if i >= len(control_actions):
                break
            
            if controller_id in errors:
                error_data = errors[controller_id]
                
                # PID calculation
                proportional_term = controller['kp'] * error_data['proportional_error']
                integral_term = controller['ki'] * error_data['integral_error']
                derivative_term = controller['kd'] * error_data['derivative_error']
                
                # Total control action
                control_action = proportional_term + integral_term + derivative_term
                
                # Saturation limits
                control_actions[i] = np.clip(control_action, -10.0, 10.0)
                
                # Update controller performance
                controller['performance_metric'] = abs(error_data['total_error'])
        
        return control_actions
    
    def apply_environmental_compensation(self, control_actions: np.ndarray) -> np.ndarray:
        """Apply environmental compensation to control actions."""
        
        compensated_actions = control_actions.copy()
        
        env_params = self.system_state['environmental_parameters']
        
        # Temperature compensation
        if 'temperature' in env_params and self.environmental_models['temperature_compensation']['model_active']:
            temp_model = self.environmental_models['temperature_compensation']
            temp_delta = env_params['temperature'] - temp_model['reference_temperature']
            temp_compensation = temp_model['coefficient'] * temp_delta
            
            # Apply to all channels
            compensated_actions += temp_compensation
        
        # Humidity compensation
        if 'humidity' in env_params and self.environmental_models['humidity_compensation']['model_active']:
            humidity_model = self.environmental_models['humidity_compensation']
            humidity_delta = env_params['humidity'] - humidity_model['reference_humidity']
            humidity_compensation = humidity_model['coefficient'] * humidity_delta
            
            compensated_actions += humidity_compensation
        
        # Pressure compensation
        if 'pressure' in env_params and self.environmental_models['pressure_compensation']['model_active']:
            pressure_model = self.environmental_models['pressure_compensation']
            pressure_delta = env_params['pressure'] - pressure_model['reference_pressure']
            pressure_compensation = pressure_model['coefficient'] * pressure_delta
            
            compensated_actions += pressure_compensation
        
        # EMI filtering
        if self.environmental_models['electromagnetic_interference']['filtering_active']:
            # Simple low-pass filtering for EMI rejection
            compensated_actions = self.apply_emi_filtering(compensated_actions)
        
        return compensated_actions
    
    def apply_emi_filtering(self, control_actions: np.ndarray) -> np.ndarray:
        """Apply EMI filtering to control actions."""
        
        # Simple exponential smoothing for EMI rejection
        alpha = 0.8  # Smoothing factor
        
        if not hasattr(self, '_previous_filtered_actions'):
            self._previous_filtered_actions = control_actions.copy()
        
        filtered_actions = alpha * control_actions + (1 - alpha) * self._previous_filtered_actions
        self._previous_filtered_actions = filtered_actions.copy()
        
        return filtered_actions
    
    def update_adaptive_parameters(self) -> None:
        """Update adaptive parameters based on performance."""
        
        # Analyze recent performance
        if len(self.control_history['errors']) < 10:
            return
        
        recent_errors = self.control_history['errors'][-10:]
        recent_performance = [np.mean(np.abs(error_dict.values())) for error_dict in recent_errors[-5:]]
        
        if len(recent_performance) < 2:
            return
        
        # Performance trend analysis
        performance_trend = recent_performance[-1] - recent_performance[0]
        
        # Adapt PID gains based on performance
        for controller_id, controller in self.pid_controllers.items():
            if controller['gain_adaptation_active']:
                
                # Get controller-specific performance
                controller_errors = [error_dict.get(controller_id, {}).get('total_error', 0) 
                                   for error_dict in recent_errors]
                controller_performance = np.mean(np.abs(controller_errors))
                
                # Adaptive gain adjustment
                if controller_performance > 0.1:  # High error
                    # Increase proportional gain, reduce integral gain
                    controller['kp'] *= 1.05
                    controller['ki'] *= 0.98
                elif controller_performance < 0.01:  # Very low error
                    # Reduce gains to prevent overcontrol
                    controller['kp'] *= 0.99
                    controller['ki'] *= 0.99
                
                # Limit gain ranges
                controller['kp'] = np.clip(controller['kp'], 0.1, 5.0)
                controller['ki'] = np.clip(controller['ki'], 0.01, 1.0)
                controller['kd'] = np.clip(controller['kd'], 0.001, 0.5)
    
    def update_environmental_compensation(self) -> None:
        """Update environmental compensation models."""
        
        env_params = self.system_state['environmental_parameters']
        
        # Update temperature model
        if 'temperature' in env_params:
            temp_model = self.environmental_models['temperature_compensation']
            temp_model['measurement_history'].append(env_params['temperature'])
            
            # Limit history
            if len(temp_model['measurement_history']) > 100:
                temp_model['measurement_history'] = temp_model['measurement_history'][-100:]
        
        # Update humidity model
        if 'humidity' in env_params:
            humidity_model = self.environmental_models['humidity_compensation']
            humidity_model['measurement_history'].append(env_params['humidity'])
            
            if len(humidity_model['measurement_history']) > 100:
                humidity_model['measurement_history'] = humidity_model['measurement_history'][-100:]
        
        # Update pressure model
        if 'pressure' in env_params:
            pressure_model = self.environmental_models['pressure_compensation']
            pressure_model['measurement_history'].append(env_params['pressure'])
            
            if len(pressure_model['measurement_history']) > 100:
                pressure_model['measurement_history'] = pressure_model['measurement_history'][-100:]
        
        # Adaptive model parameter updates could be implemented here
        # For now, using fixed coefficients
    
    def update_control_history(self, timestamp: float, errors: Dict, 
                             actions: np.ndarray) -> None:
        """Update control history for analysis."""
        
        self.control_history['timestamps'].append(timestamp)
        self.control_history['permittivity_targets'].append(self.system_state['permittivity_target'])
        self.control_history['permittivity_measured'].append(self.system_state['permittivity_measured'])
        self.control_history['control_actions'].append(actions.copy())
        self.control_history['errors'].append(errors.copy())
        self.control_history['environmental_data'].append(
            self.system_state['environmental_parameters'].copy()
        )
        
        # Limit history size
        max_history = 10000
        for key in self.control_history:
            if len(self.control_history[key]) > max_history:
                self.control_history[key] = self.control_history[key][-max_history:]
    
    def report_control_performance(self) -> None:
        """Report current control performance."""
        
        if len(self.control_history['errors']) < 5:
            return
        
        # Calculate recent performance metrics
        recent_errors = self.control_history['errors'][-5:]
        
        # Average error across all controllers
        avg_errors = []
        for error_dict in recent_errors:
            controller_errors = [abs(error_data.get('total_error', 0)) 
                               for error_data in error_dict.values()]
            if controller_errors:
                avg_errors.append(np.mean(controller_errors))
        
        if avg_errors:
            current_performance = np.mean(avg_errors)
            
            # System health assessment
            if current_performance < 0.01:
                self.system_state['system_health'] = 1.0
                status = "🟢 EXCELLENT"
            elif current_performance < 0.05:
                self.system_state['system_health'] = 0.8
                status = "🟡 GOOD"
            elif current_performance < 0.1:
                self.system_state['system_health'] = 0.6
                status = "🟠 FAIR"
            else:
                self.system_state['system_health'] = 0.4
                status = "🔴 POOR"
            
            print(f"📊 Control Performance: {status} (Error: {current_performance:.4f})")
    
    def stop_control_loop(self) -> None:
        """Stop the control loop."""
        self.system_state['control_active'] = False
        print(f"⏹️  Control loop stop requested")
    
    def generate_control_summary(self) -> Dict:
        """Generate comprehensive control summary."""
        
        if not self.control_history['timestamps']:
            return {'summary': 'No control data available'}
        
        # Time analysis
        start_time = self.control_history['timestamps'][0]
        end_time = self.control_history['timestamps'][-1]
        total_duration = end_time - start_time
        n_samples = len(self.control_history['timestamps'])
        
        # Error analysis
        all_errors = []
        for error_dict in self.control_history['errors']:
            for error_data in error_dict.values():
                all_errors.append(abs(error_data.get('total_error', 0)))
        
        error_stats = {
            'mean_error': np.mean(all_errors) if all_errors else 0,
            'max_error': np.max(all_errors) if all_errors else 0,
            'min_error': np.min(all_errors) if all_errors else 0,
            'std_error': np.std(all_errors) if all_errors else 0
        }
        
        # Control action analysis
        all_actions = np.array(self.control_history['control_actions'])
        if len(all_actions) > 0:
            action_stats = {
                'mean_action': np.mean(np.abs(all_actions)),
                'max_action': np.max(np.abs(all_actions)),
                'action_std': np.std(all_actions, axis=0).tolist()
            }
        else:
            action_stats = {'message': 'No control actions recorded'}
        
        # PID controller analysis
        controller_analysis = {}
        for controller_id, controller in self.pid_controllers.items():
            controller_analysis[controller_id] = {
                'final_gains': {
                    'kp': controller['kp'],
                    'ki': controller['ki'],
                    'kd': controller['kd']
                },
                'performance_metric': controller['performance_metric'],
                'error_history_length': len(controller['error_history'])
            }
        
        summary = {
            'control_session_summary': {
                'duration_seconds': total_duration,
                'total_samples': n_samples,
                'average_control_rate': n_samples / total_duration if total_duration > 0 else 0,
                'final_system_health': self.system_state['system_health']
            },
            'error_statistics': error_stats,
            'control_action_statistics': action_stats,
            'controller_analysis': controller_analysis,
            'kalman_filter_status': {
                'predictions': self.kalman_filter['prediction_count'],
                'updates': self.kalman_filter['update_count'],
                'final_state': self.kalman_filter['x_est'].tolist()
            }
        }
        
        print(f"\n📋 CONTROL SESSION SUMMARY")
        print(f"   ⏱️  Duration: {total_duration:.1f}s ({n_samples} samples)")
        print(f"   📊 Mean error: {error_stats['mean_error']:.4f}")
        print(f"   🎯 Max error: {error_stats['max_error']:.4f}")
        print(f"   💪 System health: {self.system_state['system_health']:.2f}")
        
        return summary


def demonstrate_adaptive_control():
    """Demonstrate adaptive control system."""
    print("🎛️  ADAPTIVE CONTROL SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize control system
    control_system = AdaptiveControlSystem(
        control_frequency=100.0,  # 100 Hz for demo
        adaptation_rate=0.05
    )
    
    # Mock measurement function
    def mock_measurement():
        """Mock measurement function with noise and drift."""
        base_permittivity = 2.5 + 0.1 * np.sin(time.time() * 0.5)  # Slow drift
        noise = np.random.normal(0, 0.01)  # Measurement noise
        
        return {
            'permittivity': base_permittivity + noise,
            'environment': {
                'temperature': 298.15 + np.random.normal(0, 0.5),
                'humidity': 45.0 + np.random.normal(0, 2.0),
                'pressure': 101325 + np.random.normal(0, 100)
            }
        }
    
    # Mock control function
    def mock_control(actions):
        """Mock control function."""
        # In real system, this would apply voltages to tunable elements
        print(f"   🎛️  Control actions: {np.array2string(actions[:3], precision=3, suppress_small=True)}...")
    
    # Test constant target
    print(f"\n1️⃣  Constant Target Control Test")
    target_permittivity = 2.5
    
    # Run control loop for short duration
    control_system.start_control_loop(
        target_permittivity=target_permittivity,
        measurement_callback=mock_measurement,
        control_callback=mock_control,
        duration=5.0  # 5 seconds for demo
    )
    
    # Test frequency-dependent target
    print(f"\n2️⃣  Frequency-Dependent Target Control Test")
    def target_function(frequencies):
        return 2.0 + 0.5 * np.sin(2 * np.pi * frequencies / (frequencies[-1] - frequencies[0]))
    
    control_system.start_control_loop(
        target_permittivity=target_function,
        measurement_callback=mock_measurement,
        control_callback=mock_control,
        duration=3.0  # 3 seconds for demo
    )
    
    print(f"\n✅ ADAPTIVE CONTROL DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    demonstrate_adaptive_control()
