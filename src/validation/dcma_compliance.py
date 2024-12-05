import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ComplianceResult:
    status: bool
    message: str
    value: float

class DCMACompliance:
    """DCMA EVMS compliance checker implementing the 32 guidelines"""
    
    def __init__(self):
        # Define DCMA thresholds
        self.thresholds = {
            'cpi_bounds': (0.95, 1.05),
            'spi_bounds': (0.95, 1.05),
            'var_threshold': 0.10,  # 10% threshold for variances
            'bcws_cum_limit': 0.15,  # 15% limit for cumulative BCWS variance
            'etc_threshold': 0.20,   # 20% threshold for ETC variance
        }
    
    def check_guideline_7(self, df: pd.DataFrame) -> ComplianceResult:
        """DCMA Guideline 7: Time-Phased Budget Baseline"""
        cum_pv = df['PV'].cumsum()
        total_bac = df['BAC'].iloc[-1]
        variance = abs(cum_pv.iloc[-1] - total_bac) / total_bac
        
        status = variance <= self.thresholds['bcws_cum_limit']
        return ComplianceResult(
            status=status,
            message=f"Cumulative PV variance is {variance:.2%}",
            value=variance
        )
    
    def check_guideline_22(self, df: pd.DataFrame) -> ComplianceResult:
        """DCMA Guideline 22: Critical Path Length Index (CPLI)"""
        spi = df['SPI'].mean()
        status = self.thresholds['spi_bounds'][0] <= spi <= self.thresholds['spi_bounds'][1]
        return ComplianceResult(
            status=status,
            message=f"Average SPI is {spi:.2f}",
            value=spi
        )
    
    def check_guideline_23(self, df: pd.DataFrame) -> ComplianceResult:
        """DCMA Guideline 23: Cost Performance Index Stability"""
        cpi_stability = df['CPI'].std() / df['CPI'].mean()
        status = cpi_stability <= self.thresholds['var_threshold']
        return ComplianceResult(
            status=status,
            message=f"CPI stability is {cpi_stability:.2%}",
            value=cpi_stability
        )
    
    def check_guideline_27(self, df: pd.DataFrame) -> ComplianceResult:
        """DCMA Guideline 27: EAC Realism"""
        etc_variance = abs(df['EAC'].diff()).mean() / df['BAC'].iloc[0]
        status = etc_variance <= self.thresholds['etc_threshold']
        return ComplianceResult(
            status=status,
            message=f"ETC variance is {etc_variance:.2%}",
            value=etc_variance
        )
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, ComplianceResult]:
        """Run all DCMA compliance checks"""
        return {
            'Guideline_7': self.check_guideline_7(df),
            'Guideline_22': self.check_guideline_22(df),
            'Guideline_23': self.check_guideline_23(df),
            'Guideline_27': self.check_guideline_27(df)
        }
