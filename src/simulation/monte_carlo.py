import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..core.metrics import EVMSMetrics
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing
import logging
from scipy.stats import norm

@dataclass
class SimulationResults:
    percentiles: Dict[str, Dict[float, float]]
    probabilities: Dict[str, Dict[str, float]]
    sensitivity: Dict[str, float]
    
class MonteCarloSimulator:
    """Monte Carlo simulation for EVMS forecasting"""
    
    def __init__(self, 
                 base_metrics: EVMSMetrics,
                 time_points: np.ndarray,
                 n_simulations: int = 1000,
                 timeout: int = 300):  # 5 minutes timeout
        self.base_metrics = base_metrics
        self.time_points = time_points
        self.n_simulations = n_simulations
        self.timeout = timeout
        
        # Define simulation parameters
        self.param_ranges = {
            'performance_factor': (0.8, 1.2),
            'cost_performance': (0.9, 1.3),
            'schedule_impact': (0.9, 1.1)
        }
        
    def _run_single_simulation(self, sim_id: int) -> Dict[str, np.ndarray]:
        """Run a single simulation iteration"""
        try:
            # Generate random parameters
            params = {
                'performance_factor': np.random.uniform(*self.param_ranges['performance_factor']),
                'cost_performance': np.random.uniform(*self.param_ranges['cost_performance']),
                'schedule_impact': np.random.uniform(*self.param_ranges['schedule_impact'])
            }
            
            # Calculate metrics with randomized parameters
            pv = self.base_metrics.calculate_planned_value(
                self.time_points * params['schedule_impact']
            )
            ev = self.base_metrics.calculate_earned_value(
                pv, params['performance_factor']
            )
            ac = self.base_metrics.calculate_actual_cost(
                ev, params['cost_performance']
            )
            
            # Calculate derived metrics
            variances = self.base_metrics.calculate_variances(ev, ac, pv)
            indices = self.base_metrics.calculate_indices(ev, ac, pv)
            eac = self.base_metrics.calculate_eac(ac, ev, indices['cpi'])
            
            return {
                'EAC': eac[-1],
                'CPI': indices['cpi'][-1],
                'SPI': indices['spi'][-1],
                'CV': variances['cv'][-1],
                'SV': variances['sv'][-1]
            }
        except Exception as e:
            logging.error(f"Error in simulation {sim_id}: {str(e)}")
            return None
    
    def run_simulation(self) -> SimulationResults:
        """Run Monte Carlo simulation with parallel processing"""
        try:
            # Use multiple processes for faster simulation
            n_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
            logging.info(f"Starting Monte Carlo simulation with {n_processes} processes")
            
            results = {
                'EAC': [],
                'CPI': [],
                'SPI': [],
                'CV': [],
                'SV': []
            }
            
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                future_to_sim = {
                    executor.submit(self._run_single_simulation, i): i 
                    for i in range(self.n_simulations)
                }
                
                completed = 0
                for future in as_completed(future_to_sim, timeout=self.timeout):
                    sim_id = future_to_sim[future]
                    try:
                        sim_result = future.result()
                        if sim_result is not None:
                            for metric, value in sim_result.items():
                                results[metric].append(value)
                        completed += 1
                        if completed % 100 == 0:
                            logging.info(f"Completed {completed}/{self.n_simulations} simulations")
                    except Exception as e:
                        logging.error(f"Simulation {sim_id} failed: {str(e)}")
                
            # Check if we have enough valid results
            min_required = self.n_simulations // 2
            valid_results = len(results['EAC'])
            if valid_results < min_required:
                raise RuntimeError(
                    f"Too many failed simulations. Only {valid_results} valid results "
                    f"out of {self.n_simulations} (minimum required: {min_required})"
                )
            
            # Calculate percentiles
            percentiles = {
                metric: {
                    p: np.percentile(values, p*100)
                    for p in [0.1, 0.25, 0.5, 0.75, 0.9]
                }
                for metric, values in results.items()
            }
            
            # Calculate probabilities of specific events
            probabilities = {
                'EAC': {
                    'over_budget': np.mean(np.array(results['EAC']) > self.base_metrics.bac),
                    'under_budget': np.mean(np.array(results['EAC']) < self.base_metrics.bac)
                },
                'CPI': {
                    'below_threshold': np.mean(np.array(results['CPI']) < 0.95),
                    'above_threshold': np.mean(np.array(results['CPI']) > 1.05)
                },
                'SPI': {
                    'schedule_delay': np.mean(np.array(results['SPI']) < 0.95),
                    'ahead_schedule': np.mean(np.array(results['SPI']) > 1.05)
                }
            }
            
            # Calculate sensitivity analysis
            sensitivity = self._calculate_sensitivity(results)
            
            logging.info("Monte Carlo simulation completed successfully")
            return SimulationResults(
                percentiles=percentiles,
                probabilities=probabilities,
                sensitivity=sensitivity
            )
            
        except TimeoutError:
            logging.error(f"Monte Carlo simulation timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            logging.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise
    
    def _calculate_sensitivity(self, results: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate sensitivity coefficients for each metric"""
        sensitivity = {}
        
        # Convert results to numpy arrays for easier calculation
        baseline = {
            metric: np.mean(values) for metric, values in results.items()
        }
        
        # Calculate standard deviations
        std_devs = {
            metric: np.std(values) for metric, values in results.items()
        }
        
        # Calculate sensitivity coefficients (normalized standard deviation)
        for metric, std_dev in std_devs.items():
            sensitivity[metric] = std_dev / abs(baseline[metric]) if baseline[metric] != 0 else 0
            
        return sensitivity
    
    def generate_confidence_bounds(self, 
                                 results: SimulationResults,
                                 confidence_level: float = 0.95
                                 ) -> Dict[str, Tuple[float, float]]:
        """Generate confidence bounds for each metric"""
        bounds = {}
        
        for metric, percentiles in results.percentiles.items():
            lower_bound = percentiles[0.25]  # 25th percentile
            upper_bound = percentiles[0.75]  # 75th percentile
            
            # Adjust bounds based on confidence level
            z_score = norm.ppf((1 + confidence_level) / 2)
            margin = (upper_bound - lower_bound) * z_score / 1.349  # Assuming normal distribution
            
            mean = percentiles[0.5]  # median
            bounds[metric] = (mean - margin, mean + margin)
            
        return bounds
