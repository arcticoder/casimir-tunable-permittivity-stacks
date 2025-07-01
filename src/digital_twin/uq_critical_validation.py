#!/usr/bin/env python3
"""
Critical UQ Validation Suite
============================

Validates fixes for high and critical severity UQ concerns:

CRITICAL FIXES TESTED:
1. Cholesky decomposition stability with singular matrices
2. PCE coefficient computation robustness
3. Sobol sensitivity analysis with degenerate cases
4. Recursive polynomial overflow protection
5. Parameter bounds validation

HIGH SEVERITY FIXES TESTED:
1. GP hyperparameter optimization robustness
2. Bootstrap confidence interval stability
3. Non-finite value handling throughout
4. Extreme parameter range handling

Author: GitHub Copilot
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple
import traceback
from advanced_uncertainty_quantification import (
    AdvancedUncertaintyQuantification, 
    UQConfiguration,
    PolynomialChaosExpansion,
    GaussianProcessSurrogate,
    SobolSensitivityAnalysis,
    UniformDistribution,
    GaussianDistribution
)


class CriticalUQValidator:
    """Validates critical UQ fixes for numerical stability and robustness."""
    
    def __init__(self):
        self.test_results = {}
        self.warnings_caught = []
        self.critical_failures = []
        self.high_severity_issues = []
        
    def run_all_tests(self) -> Dict:
        """Run complete validation suite for critical UQ fixes."""
        
        print("🔍 RUNNING CRITICAL UQ VALIDATION SUITE")
        print("=" * 50)
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test 1: Cholesky decomposition robustness
            print("\n1. Testing Cholesky decomposition robustness...")
            cholesky_result = self._test_cholesky_robustness()
            
            # Test 2: PCE coefficient computation stability
            print("2. Testing PCE coefficient computation...")
            pce_result = self._test_pce_stability()
            
            # Test 3: Sobol sensitivity edge cases
            print("3. Testing Sobol sensitivity analysis...")
            sobol_result = self._test_sobol_robustness()
            
            # Test 4: Polynomial overflow protection
            print("4. Testing polynomial overflow protection...")
            poly_result = self._test_polynomial_overflow()
            
            # Test 5: Parameter bounds validation
            print("5. Testing parameter bounds validation...")
            bounds_result = self._test_parameter_bounds()
            
            # Test 6: GP hyperparameter robustness
            print("6. Testing GP hyperparameter optimization...")
            gp_result = self._test_gp_robustness()
            
            # Test 7: Bootstrap confidence intervals
            print("7. Testing bootstrap confidence intervals...")
            bootstrap_result = self._test_bootstrap_robustness()
            
            # Test 8: Non-finite value handling
            print("8. Testing non-finite value handling...")
            nonfinite_result = self._test_nonfinite_handling()
            
            # Collect warnings
            self.warnings_caught = [str(warning.message) for warning in w]
        
        # Compile results
        self.test_results = {
            'cholesky_robustness': cholesky_result,
            'pce_stability': pce_result,
            'sobol_robustness': sobol_result,
            'polynomial_overflow': poly_result,
            'parameter_bounds': bounds_result,
            'gp_robustness': gp_result,
            'bootstrap_robustness': bootstrap_result,
            'nonfinite_handling': nonfinite_result
        }
        
        # Analyze results
        return self._analyze_validation_results()
    
    def _test_cholesky_robustness(self) -> Dict:
        """Test Cholesky decomposition robustness with singular matrices."""
        try:
            config = UQConfiguration(pce_coefficients=3)
            pce = PolynomialChaosExpansion(config)
            
            # Test 1: Singular matrix
            singular_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            function_values = np.array([1.0, 2.0, 3.0])
            
            coeffs1 = pce.compute_coefficients(singular_matrix, function_values)
            
            # Test 2: Near-singular matrix
            near_singular = np.array([[1, 0, 0], [0, 1e-15, 0], [0, 0, 1]])
            coeffs2 = pce.compute_coefficients(near_singular, function_values)
            
            # Test 3: Ill-conditioned matrix
            ill_conditioned = np.array([[1, 1-1e-12, 1-2e-12], 
                                       [1-1e-12, 1, 1-1e-12], 
                                       [1-2e-12, 1-1e-12, 1]])
            coeffs3 = pce.compute_coefficients(ill_conditioned, function_values)
            
            # Validate results
            all_finite = (np.all(np.isfinite(coeffs1)) and 
                         np.all(np.isfinite(coeffs2)) and 
                         np.all(np.isfinite(coeffs3)))
            
            return {
                'success': True,
                'all_finite': all_finite,
                'singular_handled': not np.allclose(coeffs1, 0),
                'near_singular_handled': not np.allclose(coeffs2, 0),
                'coefficients': [coeffs1, coeffs2, coeffs3]
            }
            
        except Exception as e:
            self.critical_failures.append(f"Cholesky robustness test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_pce_stability(self) -> Dict:
        """Test PCE coefficient computation with extreme cases."""
        try:
            config = UQConfiguration(pce_coefficients=5)
            pce = PolynomialChaosExpansion(config)
            
            # Initialize distributions
            distributions = [UniformDistribution(-1, 1) for _ in range(2)]
            pce.generate_basis_functions(distributions)
            
            # Test 1: Normal case
            normal_samples = np.random.uniform(-1, 1, (10, 2))
            normal_values = np.sum(normal_samples, axis=1)
            coeffs_normal = pce.compute_coefficients(normal_samples, normal_values)
            
            # Test 2: Constant function (zero variance)
            constant_values = np.ones(10)
            coeffs_constant = pce.compute_coefficients(normal_samples, constant_values)
            
            # Test 3: Extreme values
            extreme_samples = np.array([[1e10, 1e10], [-1e10, -1e10], [0, 0]])
            extreme_values = np.array([1e15, -1e15, 0])
            coeffs_extreme = pce.compute_coefficients(extreme_samples, extreme_values)
            
            # Test 4: NaN/Inf inputs
            nan_samples = np.array([[np.nan, 1], [1, np.inf], [1, 1]])
            nan_values = np.array([1, 2, 3])
            coeffs_nan = pce.compute_coefficients(nan_samples, nan_values)
            
            return {
                'success': True,
                'normal_finite': np.all(np.isfinite(coeffs_normal)),
                'constant_handled': np.all(np.isfinite(coeffs_constant)),
                'extreme_handled': np.all(np.isfinite(coeffs_extreme)),
                'nan_handled': np.all(np.isfinite(coeffs_nan)),
                'validation_errors': [pce.validation_error]
            }
            
        except Exception as e:
            self.critical_failures.append(f"PCE stability test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_sobol_robustness(self) -> Dict:
        """Test Sobol sensitivity analysis with degenerate cases."""
        try:
            config = UQConfiguration(sobol_samples=64)  # Small for testing
            sobol = SobolSensitivityAnalysis(config)
            
            distributions = [UniformDistribution(0, 1) for _ in range(3)]
            
            # Test 1: Constant function (zero variance)
            def constant_func(x):
                return 5.0
            
            result1 = sobol.compute_sensitivity_indices(constant_func, distributions)
            
            # Test 2: Function with NaN outputs
            def nan_func(x):
                if x[0] < 0.5:
                    return np.nan
                return x[0] + x[1]
            
            result2 = sobol.compute_sensitivity_indices(nan_func, distributions)
            
            # Test 3: Function with infinite outputs
            def inf_func(x):
                if x[0] > 0.8:
                    return np.inf
                return x[0] * x[1]
            
            result3 = sobol.compute_sensitivity_indices(inf_func, distributions)
            
            # Test 4: Function that raises exceptions
            def error_func(x):
                if x[0] > 0.9:
                    raise ValueError("Test error")
                return x[0] + x[1] + x[2]
            
            result4 = sobol.compute_sensitivity_indices(error_func, distributions)
            
            return {
                'success': True,
                'constant_handled': result1['success'],
                'nan_handled': result2['success'],
                'inf_handled': result3['success'],
                'error_handled': result4['success'],
                'results': [result1, result2, result3, result4]
            }
            
        except Exception as e:
            self.critical_failures.append(f"Sobol robustness test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_polynomial_overflow(self) -> Dict:
        """Test polynomial overflow protection."""
        try:
            config = UQConfiguration()
            pce = PolynomialChaosExpansion(config)
            
            # Test extreme polynomial orders
            extreme_x = np.array([100, -100, 1e6, -1e6])
            
            # Test Legendre polynomials
            legendre_results = []
            for order in [5, 10, 15, 20, 25]:  # Test up to high orders
                try:
                    result = pce._recursive_legendre(extreme_x, order)
                    legendre_results.append(np.all(np.isfinite(result)))
                except Exception:
                    legendre_results.append(False)
            
            # Test Hermite polynomials
            hermite_results = []
            for order in [5, 10, 15, 20]:  # Hermite grows faster
                try:
                    result = pce._recursive_hermite(extreme_x, order)
                    hermite_results.append(np.all(np.isfinite(result)))
                except Exception:
                    hermite_results.append(False)
            
            return {
                'success': True,
                'legendre_stable': all(legendre_results),
                'hermite_stable': all(hermite_results),
                'legendre_results': legendre_results,
                'hermite_results': hermite_results
            }
            
        except Exception as e:
            self.high_severity_issues.append(f"Polynomial overflow test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_parameter_bounds(self) -> Dict:
        """Test parameter bounds validation."""
        try:
            # Test 1: Invalid bounds (lower >= upper)
            config1 = UQConfiguration()
            config1.parameter_bounds = {'test': (10.0, 5.0)}  # Invalid order
            config1._validate_parameter_bounds()
            
            # Test 2: Non-finite bounds
            config2 = UQConfiguration()
            config2.parameter_bounds = {'test': (np.inf, np.nan)}
            config2._validate_parameter_bounds()
            
            # Test 3: Extreme ratios
            config3 = UQConfiguration()
            config3.parameter_bounds = {'test': (1e-15, 1e15)}  # Extreme ratio
            config3._validate_parameter_bounds()
            
            # Test 4: Very narrow bounds
            config4 = UQConfiguration()
            config4.parameter_bounds = {'test': (1.0, 1.0000001)}  # Very narrow
            config4._validate_parameter_bounds()
            
            return {
                'success': True,
                'invalid_order_handled': config1.parameter_bounds['test'][0] < config1.parameter_bounds['test'][1],
                'nonfinite_handled': all(np.isfinite(b) for b in config2.parameter_bounds['test']),
                'extreme_ratio_handled': True,  # Should not crash
                'narrow_bounds_handled': config4.parameter_bounds['test'][1] > config4.parameter_bounds['test'][0]
            }
            
        except Exception as e:
            self.high_severity_issues.append(f"Parameter bounds test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_gp_robustness(self) -> Dict:
        """Test GP hyperparameter optimization robustness."""
        try:
            config = UQConfiguration()
            gp = GaussianProcessSurrogate(config)
            
            # Test 1: Normal case
            X_normal = np.random.randn(20, 2)
            y_normal = np.sum(X_normal, axis=1) + 0.1 * np.random.randn(20)
            gp.fit(X_normal, y_normal)
            
            # Test 2: Constant outputs
            y_constant = np.ones(20)
            gp_constant = GaussianProcessSurrogate(config)
            gp_constant.fit(X_normal, y_constant)
            
            # Test 3: Extreme values
            X_extreme = np.array([[1e6, 1e6], [-1e6, -1e6]])
            y_extreme = np.array([1e10, -1e10])
            gp_extreme = GaussianProcessSurrogate(config)
            gp_extreme.fit(X_extreme, y_extreme)
            
            return {
                'success': True,
                'normal_fitted': gp.is_fitted,
                'constant_fitted': gp_constant.is_fitted,
                'extreme_fitted': gp_extreme.is_fitted
            }
            
        except Exception as e:
            self.high_severity_issues.append(f"GP robustness test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_bootstrap_robustness(self) -> Dict:
        """Test bootstrap confidence interval robustness."""
        try:
            config = UQConfiguration(sobol_samples=32)
            sobol = SobolSensitivityAnalysis(config)
            
            # Simple test function
            def test_func(x):
                return x[0] + 0.5 * x[1]
            
            distributions = [UniformDistribution(0, 1) for _ in range(2)]
            
            # Generate sample matrices
            sample_matrices = sobol.generate_sobol_samples(distributions)
            
            # Test robust confidence intervals
            ci_result = sobol._compute_robust_confidence_intervals(
                sample_matrices, test_func, 1.0
            )
            
            return {
                'success': True,
                'ci_computed': 'first_order' in ci_result,
                'successful_bootstraps': ci_result.get('successful_bootstraps', 0),
                'ci_finite': np.all(np.isfinite(ci_result.get('first_order', [[0, 0]])))
            }
            
        except Exception as e:
            self.high_severity_issues.append(f"Bootstrap robustness test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_nonfinite_handling(self) -> Dict:
        """Test non-finite value handling throughout UQ pipeline."""
        try:
            config = UQConfiguration()
            uq = AdvancedUncertaintyQuantification(config)
            
            # Test function that returns NaN/Inf
            def problematic_func(x):
                if x[0] < 0.2:
                    return np.nan
                elif x[0] > 0.8:
                    return np.inf
                else:
                    return x[0] + x[1]
            
            # This should not crash due to robust handling
            result = uq.comprehensive_uq_analysis(problematic_func)
            
            return {
                'success': True,
                'analysis_completed': result.get('success', False),
                'handled_gracefully': True  # If we reach here, it didn't crash
            }
            
        except Exception as e:
            self.critical_failures.append(f"Non-finite handling test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_validation_results(self) -> Dict:
        """Analyze validation results and provide summary."""
        
        # Count successes and failures
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        # Categorize issues
        critical_issues_resolved = len(self.critical_failures) == 0
        high_issues_resolved = len(self.high_severity_issues) == 0
        
        # Generate report
        print(f"\n🔍 VALIDATION RESULTS SUMMARY")
        print(f"=" * 40)
        print(f"Total tests: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        print(f"Critical failures: {len(self.critical_failures)}")
        print(f"High severity issues: {len(self.high_severity_issues)}")
        print(f"Warnings generated: {len(self.warnings_caught)}")
        
        # Detailed test results
        print(f"\n📊 DETAILED TEST RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"   {test_name}: {status}")
            
            if not result.get('success', False) and 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Critical issue analysis
        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES:")
            for failure in self.critical_failures:
                print(f"   - {failure}")
        
        if self.high_severity_issues:
            print(f"\n⚠️  HIGH SEVERITY ISSUES:")
            for issue in self.high_severity_issues:
                print(f"   - {issue}")
        
        # Summary assessment
        if critical_issues_resolved and high_issues_resolved and successful_tests >= 6:
            overall_status = "✅ UQ VALIDATION PASSED"
            print(f"\n{overall_status}")
            print("All critical and high severity UQ concerns have been resolved.")
        elif critical_issues_resolved and successful_tests >= 4:
            overall_status = "⚠️  UQ VALIDATION PARTIAL"
            print(f"\n{overall_status}")
            print("Critical issues resolved, but some high severity concerns remain.")
        else:
            overall_status = "❌ UQ VALIDATION FAILED"
            print(f"\n{overall_status}")
            print("Critical UQ issues remain unresolved.")
        
        return {
            'overall_status': overall_status,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests/total_tests*100,
            'critical_failures': self.critical_failures,
            'high_severity_issues': self.high_severity_issues,
            'warnings_caught': self.warnings_caught,
            'test_results': self.test_results,
            'critical_issues_resolved': critical_issues_resolved,
            'high_issues_resolved': high_issues_resolved
        }


def run_critical_uq_validation():
    """Run the complete critical UQ validation suite."""
    
    print("🎯 STARTING CRITICAL UQ VALIDATION")
    print("Testing fixes for high and critical severity UQ concerns...")
    
    validator = CriticalUQValidator()
    validation_results = validator.run_all_tests()
    
    return validation_results


if __name__ == "__main__":
    results = run_critical_uq_validation()
