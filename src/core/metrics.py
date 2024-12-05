import numpy as np
from scipy.special import erf
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EVMSMetrics:
    """Core EVMS metrics calculator following DCMA guidelines"""
    
    def __init__(self, bac: float, duration: int):
        self.bac = bac  # Budget at Completion
        self.duration = duration
        
    def calculate_planned_value(self, time_points: List[float], 
                              distribution: str = 'normal') -> np.ndarray:
        """Calculate time-phased Planned Value (PV)"""
        if distribution == 'normal':
            # Generate S-curve using cumulative normal distribution
            mu = self.duration / 2
            sigma = self.duration / 6
            pv = self.bac * (0.5 * (1 + erf((time_points - mu) / (sigma * np.sqrt(2)))))
            return pv
            
    def calculate_earned_value(self, pv: np.ndarray, 
                             performance_factor: float = 1.0,
                             variance: float = 0.1) -> np.ndarray:
        """Calculate Earned Value (EV) with realistic variations"""
        noise = np.random.normal(0, variance, len(pv))
        ev = pv * performance_factor + noise * pv
        return np.clip(ev, 0, self.bac)
        
    def calculate_actual_cost(self, ev: np.ndarray,
                            cost_performance: float = 1.0,
                            variance: float = 0.15) -> np.ndarray:
        """Calculate Actual Cost (AC) with realistic variations"""
        noise = np.random.normal(0, variance, len(ev))
        ac = ev * cost_performance + noise * ev
        return np.maximum(ac, 0)
        
    def calculate_variances(self, ev: np.ndarray, ac: np.ndarray, 
                          pv: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Cost and Schedule Variances"""
        return {
            'cv': ev - ac,  # Cost Variance
            'sv': ev - pv   # Schedule Variance
        }
        
    def calculate_indices(self, ev: np.ndarray, ac: np.ndarray, 
                         pv: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Performance Indices"""
        return {
            'cpi': np.divide(ev, ac, out=np.zeros_like(ev), where=ac!=0),  # Cost Performance Index
            'spi': np.divide(ev, pv, out=np.zeros_like(ev), where=pv!=0)   # Schedule Performance Index
        }
        
    def calculate_eac(self, ac: np.ndarray, ev: np.ndarray, 
                     cpi: np.ndarray) -> np.ndarray:
        """Calculate Estimate at Completion (EAC)"""
        return ac + (self.bac - ev) / cpi
        
    def calculate_tcpi(self, ev: np.ndarray, ac: np.ndarray) -> np.ndarray:
        """Calculate To Complete Performance Index (TCPI)"""
        return (self.bac - ev) / (self.bac - ac)
        
    def validate_compliance(self, cpi: np.ndarray, spi: np.ndarray,
                          threshold: float = 0.1) -> Dict[str, bool]:
        """Validate metrics against DCMA thresholds"""
        return {
            'cpi_compliant': np.abs(1 - cpi.mean()) <= threshold,
            'spi_compliant': np.abs(1 - spi.mean()) <= threshold
        }
