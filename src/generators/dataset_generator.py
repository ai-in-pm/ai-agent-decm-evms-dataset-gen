import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
from ..core.metrics import EVMSMetrics

class DatasetGenerator:
    """Generate realistic EVMS datasets based on DCMA guidelines"""
    
    def __init__(self, 
                 contract_value: float,
                 duration_months: int,
                 risk_level: str = 'medium',
                 contract_type: str = 'fixed_price',
                 complexity: str = 'medium',
                 start_date: Optional[datetime] = None):
        
        self.contract_value = contract_value
        self.duration_months = duration_months
        self.risk_level = risk_level
        self.contract_type = contract_type
        self.complexity = complexity
        self.start_date = start_date or datetime.now()
        
        # Initialize EVMS metrics calculator
        self.metrics = EVMSMetrics(bac=contract_value, duration=duration_months)
        
        # Risk factors based on project parameters
        self._risk_factors = {
            'low': {'perf': 0.05, 'cost': 0.08},
            'medium': {'perf': 0.10, 'cost': 0.15},
            'high': {'perf': 0.20, 'cost': 0.25}
        }
        
    def _generate_time_points(self) -> np.ndarray:
        """Generate monthly time points for the project duration"""
        return np.linspace(0, self.duration_months, self.duration_months + 1)
        
    def _apply_risk_factors(self) -> Dict[str, float]:
        """Calculate performance and cost factors based on risk level"""
        base_factors = self._risk_factors[self.risk_level]
        
        # Adjust for contract type
        if self.contract_type == 'cost_plus':
            cost_multiplier = 1.2
        elif self.contract_type == 'time_materials':
            cost_multiplier = 1.1
        else:  # fixed_price
            cost_multiplier = 1.0
            
        # Adjust for complexity
        if self.complexity == 'high':
            perf_multiplier = 0.9
        elif self.complexity == 'low':
            perf_multiplier = 1.1
        else:  # medium
            perf_multiplier = 1.0
            
        return {
            'performance': 1.0 + np.random.normal(0, base_factors['perf']) * perf_multiplier,
            'cost': 1.0 + np.random.normal(0, base_factors['cost']) * cost_multiplier
        }
        
    def generate(self) -> pd.DataFrame:
        """Generate a complete EVMS dataset"""
        time_points = self._generate_time_points()
        factors = self._apply_risk_factors()
        
        # Generate core metrics
        pv = self.metrics.calculate_planned_value(time_points)
        ev = self.metrics.calculate_earned_value(pv, factors['performance'])
        ac = self.metrics.calculate_actual_cost(ev, factors['cost'])
        
        # Calculate derived metrics
        variances = self.metrics.calculate_variances(ev, ac, pv)
        indices = self.metrics.calculate_indices(ev, ac, pv)
        eac = self.metrics.calculate_eac(ac, ev, indices['cpi'])
        tcpi = self.metrics.calculate_tcpi(ev, ac)
        
        # Create DataFrame
        dates = [self.start_date + timedelta(days=30*i) for i in range(len(time_points))]
        
        df = pd.DataFrame({
            'Date': dates,
            'PV': pv,
            'EV': ev,
            'AC': ac,
            'CV': variances['cv'],
            'SV': variances['sv'],
            'CPI': indices['cpi'],
            'SPI': indices['spi'],
            'EAC': eac,
            'TCPI': tcpi,
            'BAC': self.contract_value
        })
        
        # Add DCMA-specific metrics
        df['VAC'] = self.contract_value - df['EAC']
        df['MR_Utilization'] = self._calculate_mr_utilization(df)
        df['CPI_DCMA'] = self._calculate_cpi_dcma(df)
        
        return df
        
    def _calculate_mr_utilization(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Management Reserve Utilization"""
        mr_initial = self.contract_value * 0.05  # 5% initial management reserve
        return mr_initial * (1 - df.index / len(df))
        
    def _calculate_cpi_dcma(self, df: pd.DataFrame) -> pd.Series:
        """Calculate DCMA-specific CPI with additional compliance factors"""
        base_cpi = df['CPI']
        compliance_factor = np.where(base_cpi < 0.95, 0.95, 
                                   np.where(base_cpi > 1.05, 1.05, base_cpi))
        return compliance_factor
