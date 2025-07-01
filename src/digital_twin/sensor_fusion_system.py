#!/usr/bin/env python3
"""
Sensor Fusion System Module
===========================

Implements advanced EWMA (Exponentially Weighted Moving Average) adaptive filtering
with validated sensor fusion mathematics for multi-sensor data integration.

Mathematical Foundation:
- EWMA filter: x̂_k = α*z_k + (1-α)*x̂_{k-1}
- Adaptive α: α_k = f(innovation, uncertainty, sensor_quality)
- Multi-sensor fusion: x̂ = Σᵢ wᵢ*x̂ᵢ / Σᵢ wᵢ
- Validated uncertainty propagation: σ²_fused = f(σᵢ², corr_matrix)

Author: GitHub Copilot
"""

import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import warnings
from abc import ABC, abstractmethod
from enum import Enum


class SensorType(Enum):
    """Enumeration of sensor types."""
    CAPACITIVE = "capacitive"
    OPTICAL = "optical"
    THERMAL = "thermal"
    FORCE = "force"
    POSITION = "position"
    VOLTAGE = "voltage"
    CURRENT = "current"


@dataclass
class SensorConfiguration:
    """Configuration for individual sensor."""
    sensor_id: str
    sensor_type: SensorType
    measurement_noise_std: float
    bias_drift_rate: float = 1e-6  # Bias drift per second
    sampling_frequency: float = 1e3  # Hz
    dynamic_range: Tuple[float, float] = (-1e6, 1e6)
    resolution: float = 1e-9
    calibration_coefficients: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    
    # Quality metrics
    accuracy_class: float = 0.01  # 1% accuracy
    stability_factor: float = 0.99  # Long-term stability
    reliability_factor: float = 0.995  # Reliability score


@dataclass
class FusionConfiguration:
    """Configuration for sensor fusion system."""
    # EWMA parameters
    initial_alpha: float = 0.1
    alpha_adaptation_rate: float = 0.01
    alpha_bounds: Tuple[float, float] = (0.001, 0.5)
    
    # Fusion parameters
    correlation_threshold: float = 0.95
    outlier_threshold: float = 3.0  # Sigma threshold
    trust_decay_rate: float = 0.1
    
    # Validation parameters
    chi_square_confidence: float = 0.95
    innovation_validation: bool = True
    cross_validation_window: int = 50
    
    # Performance parameters
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    update_frequency: float = 1e3  # Hz


class SensorModel:
    """
    Advanced sensor model with calibration and uncertainty quantification.
    
    Models:
    - Measurement noise (Gaussian + non-Gaussian components)
    - Bias drift (time-varying)
    - Scale factor errors
    - Cross-coupling effects
    """
    
    def __init__(self, config: SensorConfiguration):
        self.config = config
        self.sensor_id = config.sensor_id
        self.sensor_type = config.sensor_type
        
        # Current state
        self.current_bias = 0.0
        self.current_scale_factor = config.calibration_coefficients[0]
        self.current_offset = config.calibration_coefficients[1]
        
        # Historical data
        self.measurement_history = []
        self.timestamp_history = []
        self.quality_history = []
        
        # Statistics
        self.running_mean = 0.0
        self.running_variance = config.measurement_noise_std**2
        self.measurement_count = 0
        
        # Quality indicators
        self.current_snr = np.inf
        self.health_status = 1.0  # 1.0 = healthy, 0.0 = failed
        
        print(f"📡 SENSOR MODEL INITIALIZED: {config.sensor_id} ({config.sensor_type.value})")
    
    def simulate_measurement(self, true_value: float, timestamp: float) -> Dict:
        """
        Simulate sensor measurement with realistic noise and drift.
        
        measurement = scale_factor * true_value + offset + bias_drift + noise
        """
        # Update bias drift
        time_elapsed = timestamp - (self.timestamp_history[-1] if self.timestamp_history else 0.0)
        drift_increment = self.config.bias_drift_rate * time_elapsed * np.random.randn()
        self.current_bias += drift_increment
        
        # Apply sensor model
        scaled_value = self.current_scale_factor * true_value
        biased_value = scaled_value + self.current_offset + self.current_bias
        
        # Add measurement noise (mixed Gaussian + occasional outliers)
        noise_std = self.config.measurement_noise_std
        
        # 95% Gaussian noise + 5% outliers
        if np.random.rand() < 0.95:
            noise = np.random.normal(0, noise_std)
        else:
            # Outlier noise (5x standard deviation)
            noise = np.random.normal(0, 5 * noise_std)
        
        measurement = biased_value + noise
        
        # Apply dynamic range limits
        measurement = np.clip(measurement, 
                            self.config.dynamic_range[0], 
                            self.config.dynamic_range[1])
        
        # Quantization (resolution limit)
        if self.config.resolution > 0:
            measurement = np.round(measurement / self.config.resolution) * self.config.resolution
        
        # Update statistics
        self._update_statistics(measurement, timestamp)
        
        # Compute measurement quality
        quality_metrics = self._compute_quality_metrics(measurement, true_value)
        
        return {
            'measurement': measurement,
            'timestamp': timestamp,
            'true_value': true_value,
            'noise_contribution': noise,
            'bias_contribution': self.current_bias,
            'quality_metrics': quality_metrics,
            'sensor_health': self.health_status
        }
    
    def _update_statistics(self, measurement: float, timestamp: float):
        """Update running statistics."""
        self.measurement_count += 1
        
        # Update running mean and variance (Welford's method)
        delta = measurement - self.running_mean
        self.running_mean += delta / self.measurement_count
        self.running_variance += delta * (measurement - self.running_mean)
        
        # Store history (limited window)
        max_history = 1000
        self.measurement_history.append(measurement)
        self.timestamp_history.append(timestamp)
        
        if len(self.measurement_history) > max_history:
            self.measurement_history.pop(0)
            self.timestamp_history.pop(0)
    
    def _compute_quality_metrics(self, measurement: float, true_value: float) -> Dict:
        """Compute measurement quality metrics."""
        # Signal-to-noise ratio estimate
        if len(self.measurement_history) > 10:
            recent_measurements = np.array(self.measurement_history[-10:])
            signal_power = np.mean(recent_measurements)**2
            noise_power = np.var(recent_measurements)
            self.current_snr = signal_power / (noise_power + 1e-12)
        
        # Accuracy (error relative to true value)
        absolute_error = abs(measurement - true_value)
        relative_error = absolute_error / (abs(true_value) + 1e-12)
        
        # Consistency (variation from expected)
        expected_std = self.config.measurement_noise_std
        actual_std = np.sqrt(self.running_variance / (self.measurement_count + 1e-12))
        consistency = 1.0 - abs(actual_std - expected_std) / expected_std
        
        # Overall quality score (0-1)
        accuracy_score = max(0, 1 - relative_error / self.config.accuracy_class)
        snr_score = min(1, self.current_snr / 100.0)  # Normalize SNR
        consistency_score = max(0, consistency)
        
        overall_quality = 0.4 * accuracy_score + 0.3 * snr_score + 0.3 * consistency_score
        
        return {
            'snr': self.current_snr,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'consistency': consistency_score,
            'overall_quality': overall_quality
        }
    
    def calibrate_sensor(self, reference_measurements: List[Tuple[float, float]]) -> Dict:
        """
        Calibrate sensor using reference measurements.
        
        Args:
            reference_measurements: List of (true_value, measured_value) pairs
        """
        try:
            if len(reference_measurements) < 2:
                return {'success': False, 'error': 'Insufficient calibration data'}
            
            # Extract data
            true_values = np.array([pair[0] for pair in reference_measurements])
            measured_values = np.array([pair[1] for pair in reference_measurements])
            
            # Linear regression: measured = scale * true + offset
            A = np.vstack([true_values, np.ones(len(true_values))]).T
            calibration_coeffs, residuals, rank, s = la.lstsq(A, measured_values, rcond=None)
            
            # Update calibration
            self.current_scale_factor = calibration_coeffs[0]
            self.current_offset = calibration_coeffs[1]
            
            # Compute calibration quality
            predicted_values = calibration_coeffs[0] * true_values + calibration_coeffs[1]
            calibration_error = np.sqrt(np.mean((measured_values - predicted_values)**2))
            correlation = np.corrcoef(true_values, measured_values)[0, 1]
            
            return {
                'success': True,
                'scale_factor': self.current_scale_factor,
                'offset': self.current_offset,
                'calibration_error': calibration_error,
                'correlation': correlation
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class EWMAFilter:
    """
    Advanced Exponentially Weighted Moving Average filter with adaptive parameters.
    
    Features:
    - Adaptive smoothing parameter α
    - Innovation-based adaptation
    - Uncertainty quantification
    - Outlier detection and handling
    """
    
    def __init__(self, initial_alpha: float = 0.1):
        self.alpha = initial_alpha
        self.initial_alpha = initial_alpha
        
        # Filter state
        self.estimate = None
        self.variance = None
        self.innovation_variance = 1.0
        
        # Adaptation parameters
        self.adaptation_gain = 0.01
        self.alpha_min = 0.001
        self.alpha_max = 0.5
        
        # History for validation
        self.estimate_history = []
        self.innovation_history = []
        self.alpha_history = []
        
        print(f"🔄 EWMA FILTER INITIALIZED: α₀ = {initial_alpha:.3f}")
    
    def update(self, measurement: float, measurement_variance: float, 
               sensor_quality: float = 1.0) -> Dict:
        """
        Update EWMA filter with new measurement.
        
        x̂_k = α_k * z_k + (1-α_k) * x̂_{k-1}
        """
        try:
            # Initialize on first measurement
            if self.estimate is None:
                self.estimate = measurement
                self.variance = measurement_variance
                innovation = 0.0
                adapted_alpha = self.alpha
            else:
                # Compute innovation
                innovation = measurement - self.estimate
                
                # Adapt smoothing parameter based on innovation and quality
                adapted_alpha = self._adapt_alpha(innovation, measurement_variance, sensor_quality)
                
                # Update estimate
                self.estimate = adapted_alpha * measurement + (1 - adapted_alpha) * self.estimate
                
                # Update variance estimate
                self.variance = (adapted_alpha * measurement_variance + 
                               (1 - adapted_alpha)**2 * self.variance)
            
            # Store history
            self.estimate_history.append(self.estimate)
            self.innovation_history.append(innovation)
            self.alpha_history.append(adapted_alpha)
            
            # Limit history size
            max_history = 1000
            if len(self.estimate_history) > max_history:
                self.estimate_history.pop(0)
                self.innovation_history.pop(0)
                self.alpha_history.pop(0)
            
            # Update current alpha
            self.alpha = adapted_alpha
            
            return {
                'success': True,
                'estimate': self.estimate,
                'variance': self.variance,
                'innovation': innovation,
                'alpha_used': adapted_alpha,
                'confidence_interval': self._compute_confidence_interval()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _adapt_alpha(self, innovation: float, measurement_variance: float, 
                    sensor_quality: float) -> float:
        """
        Adapt smoothing parameter α based on innovation and sensor quality.
        
        Higher innovation → higher α (more responsive)
        Higher sensor quality → α closer to optimal value
        """
        # Normalized innovation (z-score)
        innovation_std = np.sqrt(self.innovation_variance + 1e-12)
        normalized_innovation = abs(innovation) / innovation_std
        
        # Base adaptation based on innovation
        if normalized_innovation > 2.0:  # Large innovation
            alpha_innovation = min(self.alpha_max, self.alpha * 1.5)
        elif normalized_innovation < 0.5:  # Small innovation
            alpha_innovation = max(self.alpha_min, self.alpha * 0.8)
        else:
            alpha_innovation = self.alpha
        
        # Quality-based modification
        quality_factor = sensor_quality  # Quality ∈ [0,1]
        alpha_quality = alpha_innovation * quality_factor + self.initial_alpha * (1 - quality_factor)
        
        # Variance-based adaptation (higher variance → lower α for stability)
        variance_factor = 1.0 / (1.0 + measurement_variance / self.innovation_variance)
        alpha_final = alpha_quality * variance_factor
        
        # Apply bounds
        alpha_final = np.clip(alpha_final, self.alpha_min, self.alpha_max)
        
        # Update innovation variance estimate
        if len(self.innovation_history) > 10:
            recent_innovations = np.array(self.innovation_history[-10:])
            self.innovation_variance = np.var(recent_innovations)
        
        return alpha_final
    
    def _compute_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for current estimate."""
        if self.variance is None or self.variance <= 0:
            return (self.estimate, self.estimate)
        
        std_dev = np.sqrt(self.variance)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = self.estimate - z_score * std_dev
        upper = self.estimate + z_score * std_dev
        
        return (lower, upper)
    
    def detect_outliers(self, threshold: float = 3.0) -> List[int]:
        """Detect outliers in innovation sequence."""
        if len(self.innovation_history) < 10:
            return []
        
        innovations = np.array(self.innovation_history)
        innovation_std = np.std(innovations)
        
        outlier_indices = []
        for i, innovation in enumerate(innovations):
            if abs(innovation) > threshold * innovation_std:
                outlier_indices.append(i)
        
        return outlier_indices
    
    def reset_filter(self):
        """Reset filter to initial state."""
        self.estimate = None
        self.variance = None
        self.alpha = self.initial_alpha
        self.estimate_history.clear()
        self.innovation_history.clear()
        self.alpha_history.clear()


class MultiSensorFusion:
    """
    Advanced multi-sensor fusion system with EWMA filtering.
    
    Features:
    - Weighted fusion based on sensor quality
    - Cross-correlation analysis
    - Fault detection and isolation
    - Uncertainty propagation
    """
    
    def __init__(self, sensors: List[SensorModel], config: FusionConfiguration):
        self.sensors = sensors
        self.config = config
        self.n_sensors = len(sensors)
        
        # EWMA filters for each sensor
        self.filters = {}
        for sensor in sensors:
            self.filters[sensor.sensor_id] = EWMAFilter(config.initial_alpha)
        
        # Fusion state
        self.fused_estimate = None
        self.fused_variance = None
        self.sensor_weights = np.ones(self.n_sensors) / self.n_sensors  # Equal initial weights
        
        # Cross-correlation matrix
        self.correlation_matrix = np.eye(self.n_sensors)
        
        # Trust factors for each sensor
        self.trust_factors = np.ones(self.n_sensors)
        
        # Fusion history
        self.fusion_history = []
        self.weight_history = []
        
        print(f"🔗 MULTI-SENSOR FUSION INITIALIZED")
        print(f"   Number of sensors: {self.n_sensors}")
        print(f"   Sensor types: {[s.sensor_type.value for s in sensors]}")
    
    def fuse_measurements(self, measurements: Dict[str, Dict]) -> Dict:
        """
        Fuse measurements from multiple sensors.
        
        measurements: {sensor_id: measurement_data}
        """
        try:
            # Extract measurements and update individual filters
            sensor_estimates = {}
            sensor_variances = {}
            sensor_qualities = {}
            
            for sensor_id, measurement_data in measurements.items():
                if sensor_id in self.filters:
                    # Update EWMA filter
                    measurement = measurement_data['measurement']
                    noise_var = measurement_data.get('measurement_variance', 
                                                   self.sensors[0].config.measurement_noise_std**2)
                    quality = measurement_data.get('quality_metrics', {}).get('overall_quality', 1.0)
                    
                    filter_result = self.filters[sensor_id].update(measurement, noise_var, quality)
                    
                    if filter_result['success']:
                        sensor_estimates[sensor_id] = filter_result['estimate']
                        sensor_variances[sensor_id] = filter_result['variance']
                        sensor_qualities[sensor_id] = quality
            
            if len(sensor_estimates) < 2:
                return {'success': False, 'error': 'Insufficient sensor data for fusion'}
            
            # Update cross-correlation matrix
            self._update_correlation_matrix(sensor_estimates)
            
            # Update sensor weights based on quality and correlation
            self._update_sensor_weights(sensor_qualities, sensor_variances)
            
            # Perform weighted fusion
            fusion_result = self._weighted_fusion(sensor_estimates, sensor_variances)
            
            # Validate fusion result
            validation_result = self._validate_fusion(sensor_estimates, fusion_result)
            
            # Store history
            self.fusion_history.append({
                'timestamp': measurements[list(measurements.keys())[0]].get('timestamp', 0.0),
                'fused_estimate': fusion_result['estimate'],
                'fused_variance': fusion_result['variance'],
                'sensor_estimates': sensor_estimates.copy(),
                'weights': self.sensor_weights.copy()
            })
            
            # Limit history size
            if len(self.fusion_history) > 1000:
                self.fusion_history.pop(0)
            
            return {
                'success': True,
                'fused_estimate': fusion_result['estimate'],
                'fused_variance': fusion_result['variance'],
                'sensor_estimates': sensor_estimates,
                'sensor_weights': dict(zip(sensor_estimates.keys(), 
                                         self.sensor_weights[:len(sensor_estimates)])),
                'validation': validation_result,
                'correlation_matrix': self.correlation_matrix
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_correlation_matrix(self, sensor_estimates: Dict[str, float]):
        """Update cross-correlation matrix using recent estimates."""
        if len(self.fusion_history) < 10:
            return  # Need sufficient history
        
        # Extract recent estimates for correlation analysis
        n_recent = min(50, len(self.fusion_history))
        recent_data = self.fusion_history[-n_recent:]
        
        sensor_ids = list(sensor_estimates.keys())
        n_active = len(sensor_ids)
        
        if n_active < 2:
            return
        
        # Build data matrix
        data_matrix = np.zeros((n_recent, n_active))
        for i, sensor_id in enumerate(sensor_ids):
            for j, record in enumerate(recent_data):
                if sensor_id in record['sensor_estimates']:
                    data_matrix[j, i] = record['sensor_estimates'][sensor_id]
        
        # Compute correlation matrix
        try:
            correlation_submatrix = np.corrcoef(data_matrix, rowvar=False)
            
            # Update full correlation matrix
            for i, sensor_id_i in enumerate(sensor_ids):
                for j, sensor_id_j in enumerate(sensor_ids):
                    # Find indices in full sensor list
                    idx_i = next((k for k, s in enumerate(self.sensors) if s.sensor_id == sensor_id_i), i)
                    idx_j = next((k for k, s in enumerate(self.sensors) if s.sensor_id == sensor_id_j), j)
                    
                    if idx_i < self.n_sensors and idx_j < self.n_sensors:
                        self.correlation_matrix[idx_i, idx_j] = correlation_submatrix[i, j]
        
        except Exception as e:
            warnings.warn(f"Correlation matrix update failed: {e}")
    
    def _update_sensor_weights(self, sensor_qualities: Dict[str, float], 
                              sensor_variances: Dict[str, float]):
        """Update sensor weights based on quality and uncertainty."""
        sensor_ids = list(sensor_qualities.keys())
        n_active = len(sensor_ids)
        
        if n_active == 0:
            return
        
        weights = np.zeros(n_active)
        
        for i, sensor_id in enumerate(sensor_ids):
            quality = sensor_qualities[sensor_id]
            variance = sensor_variances[sensor_id]
            
            # Weight inversely proportional to variance and proportional to quality
            precision = 1.0 / (variance + 1e-12)  # Inverse variance weighting
            quality_factor = quality**2  # Square quality for emphasis
            
            weights[i] = precision * quality_factor
        
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum > 1e-12:
            weights /= weight_sum
        else:
            weights = np.ones(n_active) / n_active
        
        # Update weights in full array (for consistent indexing)
        self.sensor_weights[:n_active] = weights
        if n_active < self.n_sensors:
            self.sensor_weights[n_active:] = 0.0
    
    def _weighted_fusion(self, sensor_estimates: Dict[str, float], 
                        sensor_variances: Dict[str, float]) -> Dict:
        """Perform weighted fusion of sensor estimates."""
        sensor_ids = list(sensor_estimates.keys())
        estimates = np.array([sensor_estimates[sid] for sid in sensor_ids])
        variances = np.array([sensor_variances[sid] for sid in sensor_ids])
        weights = self.sensor_weights[:len(sensor_ids)]
        
        # Weighted average
        fused_estimate = np.sum(weights * estimates)
        
        # Fused variance (accounting for correlations)
        fused_variance = 0.0
        for i in range(len(sensor_ids)):
            for j in range(len(sensor_ids)):
                correlation = self.correlation_matrix[i, j] if i < self.n_sensors and j < self.n_sensors else 0.0
                covariance = correlation * np.sqrt(variances[i] * variances[j])
                fused_variance += weights[i] * weights[j] * (variances[i] if i == j else covariance)
        
        # Ensure positive variance
        fused_variance = max(fused_variance, 1e-12)
        
        self.fused_estimate = fused_estimate
        self.fused_variance = fused_variance
        
        return {
            'estimate': fused_estimate,
            'variance': fused_variance,
            'standard_deviation': np.sqrt(fused_variance)
        }
    
    def _validate_fusion(self, sensor_estimates: Dict[str, float], 
                        fusion_result: Dict) -> Dict:
        """Validate fusion result using chi-square test and consistency checks."""
        try:
            sensor_ids = list(sensor_estimates.keys())
            estimates = np.array([sensor_estimates[sid] for sid in sensor_ids])
            fused_estimate = fusion_result['estimate']
            
            # Chi-square goodness of fit test
            residuals = estimates - fused_estimate
            chi_square_stat = np.sum(residuals**2 / (fusion_result['variance'] + 1e-12))
            
            degrees_of_freedom = len(estimates) - 1
            if degrees_of_freedom > 0:
                p_value = 1 - stats.chi2.cdf(chi_square_stat, degrees_of_freedom)
                chi_square_valid = p_value > (1 - self.config.chi_square_confidence)
            else:
                chi_square_valid = True
                p_value = 1.0
            
            # Consistency check (all estimates within reasonable bounds)
            std_dev = np.sqrt(fusion_result['variance'])
            outlier_threshold = self.config.outlier_threshold
            
            outliers = []
            consistent_estimates = []
            
            for i, (sensor_id, estimate) in enumerate(sensor_estimates.items()):
                z_score = abs(estimate - fused_estimate) / (std_dev + 1e-12)
                if z_score > outlier_threshold:
                    outliers.append(sensor_id)
                else:
                    consistent_estimates.append(sensor_id)
            
            consistency_ratio = len(consistent_estimates) / len(sensor_estimates)
            
            return {
                'chi_square_valid': chi_square_valid,
                'chi_square_statistic': chi_square_stat,
                'chi_square_p_value': p_value,
                'consistency_ratio': consistency_ratio,
                'outlier_sensors': outliers,
                'consistent_sensors': consistent_estimates,
                'overall_valid': chi_square_valid and consistency_ratio > 0.5
            }
            
        except Exception as e:
            return {
                'chi_square_valid': False,
                'overall_valid': False,
                'error': str(e)
            }
    
    def detect_sensor_faults(self) -> Dict:
        """Detect and isolate faulty sensors."""
        try:
            if len(self.fusion_history) < self.config.cross_validation_window:
                return {'success': False, 'error': 'Insufficient history for fault detection'}
            
            # Analyze recent fusion results
            recent_history = self.fusion_history[-self.config.cross_validation_window:]
            
            fault_scores = {}
            for sensor in self.sensors:
                sensor_id = sensor.sensor_id
                
                # Extract sensor estimates from history
                sensor_estimates = []
                fused_estimates = []
                
                for record in recent_history:
                    if sensor_id in record['sensor_estimates']:
                        sensor_estimates.append(record['sensor_estimates'][sensor_id])
                        fused_estimates.append(record['fused_estimate'])
                
                if len(sensor_estimates) < 10:
                    continue
                
                sensor_estimates = np.array(sensor_estimates)
                fused_estimates = np.array(fused_estimates)
                
                # Compute fault metrics
                residuals = sensor_estimates - fused_estimates
                residual_std = np.std(residuals)
                mean_residual = np.mean(residuals)
                
                # Normalized metrics
                residual_score = residual_std / (sensor.config.measurement_noise_std + 1e-12)
                bias_score = abs(mean_residual) / (sensor.config.measurement_noise_std + 1e-12)
                
                # Combined fault score
                fault_score = 0.6 * residual_score + 0.4 * bias_score
                fault_scores[sensor_id] = fault_score
            
            # Identify faulty sensors
            fault_threshold = 3.0
            faulty_sensors = [sid for sid, score in fault_scores.items() if score > fault_threshold]
            healthy_sensors = [sid for sid, score in fault_scores.items() if score <= fault_threshold]
            
            return {
                'success': True,
                'fault_scores': fault_scores,
                'faulty_sensors': faulty_sensors,
                'healthy_sensors': healthy_sensors,
                'fault_threshold': fault_threshold
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def demonstrate_sensor_fusion():
    """Demonstrate sensor fusion system capabilities."""
    
    print("🔗 SENSOR FUSION SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Create multiple sensors
    sensors = [
        SensorModel(SensorConfiguration(
            sensor_id="permittivity_capacitive",
            sensor_type=SensorType.CAPACITIVE,
            measurement_noise_std=0.05,
            bias_drift_rate=1e-7,
            accuracy_class=0.02,
            sampling_frequency=2e3
        )),
        SensorModel(SensorConfiguration(
            sensor_id="permittivity_optical",
            sensor_type=SensorType.OPTICAL,
            measurement_noise_std=0.02,
            bias_drift_rate=5e-8,
            accuracy_class=0.01,
            sampling_frequency=1e3
        )),
        SensorModel(SensorConfiguration(
            sensor_id="temperature_thermal",
            sensor_type=SensorType.THERMAL,
            measurement_noise_std=0.5,  # K
            bias_drift_rate=1e-6,
            accuracy_class=0.005,
            sampling_frequency=100
        )),
        SensorModel(SensorConfiguration(
            sensor_id="force_piezo",
            sensor_type=SensorType.FORCE,
            measurement_noise_std=1e-14,  # N
            bias_drift_rate=1e-16,
            accuracy_class=0.03,
            sampling_frequency=5e3
        )),
    ]
    
    print(f"\n📡 Configured sensors:")
    for sensor in sensors:
        print(f"   {sensor.sensor_id}: {sensor.sensor_type.value} (σ={sensor.config.measurement_noise_std:.2e})")
    
    # Initialize fusion system
    fusion_config = FusionConfiguration(
        initial_alpha=0.15,
        alpha_adaptation_rate=0.02,
        correlation_threshold=0.9,
        outlier_threshold=2.5
    )
    
    fusion_system = MultiSensorFusion(sensors, fusion_config)
    
    # Simulate measurement sequence
    print(f"\n🧪 Simulating measurement fusion...")
    
    true_values = {
        "permittivity_capacitive": 4.5,
        "permittivity_optical": 4.5,
        "temperature_thermal": 325.0,
        "force_piezo": 1.2e-13
    }
    
    fusion_results = []
    
    for time_step in range(100):
        timestamp = time_step * 0.001  # 1 ms steps
        
        # Simulate measurements from all sensors
        measurements = {}
        for sensor in sensors:
            true_value = true_values[sensor.sensor_id]
            
            # Add some time-varying effects
            if "permittivity" in sensor.sensor_id:
                true_value += 0.2 * np.sin(2 * np.pi * 0.1 * timestamp)  # 0.1 Hz variation
            elif "temperature" in sensor.sensor_id:
                true_value += 5 * np.sin(2 * np.pi * 0.05 * timestamp)  # 0.05 Hz variation
            
            measurement_data = sensor.simulate_measurement(true_value, timestamp)
            measurements[sensor.sensor_id] = measurement_data
        
        # Perform fusion
        fusion_result = fusion_system.fuse_measurements(measurements)
        
        if fusion_result['success']:
            fusion_results.append(fusion_result)
            
            # Print progress every 20 steps
            if time_step % 20 == 0:
                print(f"   Step {time_step}: Fusion successful")
                print(f"      Permittivity estimates: {[f'{v:.3f}' for k, v in fusion_result['sensor_estimates'].items() if 'permittivity' in k]}")
                weights = fusion_result['sensor_weights']
                print(f"      Sensor weights: {[f'{k}:{v:.3f}' for k, v in weights.items()]}")
    
    # Analyze results
    print(f"\n📊 Fusion Analysis:")
    print(f"   Total fusion steps: {len(fusion_results)}")
    
    if fusion_results:
        # Extract permittivity results for analysis
        permittivity_results = []
        for result in fusion_results:
            perm_estimates = [v for k, v in result['sensor_estimates'].items() if 'permittivity' in k]
            if perm_estimates:
                permittivity_results.extend(perm_estimates)
        
        if permittivity_results:
            print(f"   Permittivity statistics:")
            print(f"      Mean: {np.mean(permittivity_results):.4f}")
            print(f"      Std: {np.std(permittivity_results):.4f}")
            print(f"      Range: [{np.min(permittivity_results):.4f}, {np.max(permittivity_results):.4f}]")
        
        # Validation statistics
        valid_fusions = [r for r in fusion_results if r['validation']['overall_valid']]
        validation_rate = len(valid_fusions) / len(fusion_results)
        print(f"   Validation success rate: {validation_rate:.1%}")
        
        # Average consistency
        consistency_scores = [r['validation']['consistency_ratio'] for r in fusion_results]
        avg_consistency = np.mean(consistency_scores)
        print(f"   Average consistency: {avg_consistency:.3f}")
    
    # Test fault detection
    print(f"\n🔍 Testing fault detection...")
    fault_detection = fusion_system.detect_sensor_faults()
    
    if fault_detection['success']:
        print(f"   Fault detection successful")
        print(f"   Healthy sensors: {fault_detection['healthy_sensors']}")
        if fault_detection['faulty_sensors']:
            print(f"   Faulty sensors detected: {fault_detection['faulty_sensors']}")
        else:
            print(f"   No faulty sensors detected")
    else:
        print(f"   Fault detection failed: {fault_detection['error']}")
    
    # Test individual EWMA filter performance
    print(f"\n🔄 EWMA Filter Performance:")
    for sensor_id, ewma_filter in fusion_system.filters.items():
        if ewma_filter.estimate is not None:
            recent_alphas = ewma_filter.alpha_history[-10:] if len(ewma_filter.alpha_history) >= 10 else ewma_filter.alpha_history
            if recent_alphas:
                print(f"   {sensor_id}:")
                print(f"      Current estimate: {ewma_filter.estimate:.4f}")
                print(f"      Current α: {ewma_filter.alpha:.4f}")
                print(f"      Average α (recent): {np.mean(recent_alphas):.4f}")
                
                outliers = ewma_filter.detect_outliers()
                print(f"      Outliers detected: {len(outliers)}")
    
    print(f"\n✅ Sensor fusion demonstration completed!")
    
    return fusion_system, fusion_results


if __name__ == "__main__":
    demonstrate_sensor_fusion()
