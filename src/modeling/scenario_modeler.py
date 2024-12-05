import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from ..generators.dataset_generator import DatasetGenerator

class ScenarioModeler:
    """Generate and analyze what-if scenarios for EVMS data"""
    
    def __init__(self, base_dataset: pd.DataFrame):
        self.base_dataset = base_dataset.copy()
        self.scenarios = {}
        
    def model_schedule_slip(self, 
                           delay_months: int,
                           impact_start_month: int,
                           productivity_impact: float = 0.9) -> pd.DataFrame:
        """Model impact of schedule delays"""
        scenario_data = self.base_dataset.copy()
        
        # Adjust earned value for reduced productivity
        mask = scenario_data.index >= impact_start_month
        scenario_data.loc[mask, 'EV'] *= productivity_impact
        
        # Extend timeline and adjust metrics
        extension = pd.DataFrame(index=range(len(scenario_data), 
                                          len(scenario_data) + delay_months))
        extension['Date'] = pd.date_range(start=scenario_data['Date'].iloc[-1],
                                        periods=delay_months + 1,
                                        freq='M')[1:]
        
        # Extrapolate values for extended period
        for col in ['PV', 'EV', 'AC']:
            if col == 'PV':
                extension[col] = scenario_data[col].iloc[-1]
            else:
                extension[col] = scenario_data[col].iloc[-1] * \
                               (1 + np.random.normal(0, 0.05, len(extension)))
        
        scenario_data = pd.concat([scenario_data, extension])
        self._recalculate_metrics(scenario_data)
        
        self.scenarios['schedule_slip'] = scenario_data
        return scenario_data
    
    def model_cost_overrun(self,
                          overrun_percentage: float,
                          impact_start_month: int,
                          gradual: bool = True) -> pd.DataFrame:
        """Model impact of cost overruns"""
        scenario_data = self.base_dataset.copy()
        
        # Calculate cost increase
        mask = scenario_data.index >= impact_start_month
        if gradual:
            # Gradually increase costs
            months_affected = len(scenario_data) - impact_start_month
            increase_factor = np.linspace(1, 1 + overrun_percentage, months_affected)
            scenario_data.loc[mask, 'AC'] *= increase_factor
        else:
            # Immediate cost increase
            scenario_data.loc[mask, 'AC'] *= (1 + overrun_percentage)
        
        self._recalculate_metrics(scenario_data)
        
        self.scenarios['cost_overrun'] = scenario_data
        return scenario_data
    
    def model_scope_change(self,
                          scope_change_percentage: float,
                          impact_month: int) -> pd.DataFrame:
        """Model impact of scope changes"""
        scenario_data = self.base_dataset.copy()
        
        # Adjust BAC for scope change
        new_bac = scenario_data['BAC'].iloc[0] * (1 + scope_change_percentage)
        scenario_data.loc[scenario_data.index >= impact_month, 'BAC'] = new_bac
        
        # Adjust planned value curve
        mask = scenario_data.index >= impact_month
        scenario_data.loc[mask, 'PV'] *= (1 + scope_change_percentage)
        
        # Adjust earned value based on typical performance
        avg_performance = (scenario_data.loc[:impact_month, 'EV'] / 
                         scenario_data.loc[:impact_month, 'PV']).mean()
        scenario_data.loc[mask, 'EV'] = scenario_data.loc[mask, 'PV'] * avg_performance
        
        self._recalculate_metrics(scenario_data)
        
        self.scenarios['scope_change'] = scenario_data
        return scenario_data
    
    def _recalculate_metrics(self, df: pd.DataFrame) -> None:
        """Recalculate all dependent metrics"""
        df['CV'] = df['EV'] - df['AC']
        df['SV'] = df['EV'] - df['PV']
        df['CPI'] = df['EV'] / df['AC']
        df['SPI'] = df['EV'] / df['PV']
        
        # Recalculate EAC using CPI
        df['EAC'] = df['AC'] + (df['BAC'] - df['EV']) / df['CPI']
        df['TCPI'] = (df['BAC'] - df['EV']) / (df['BAC'] - df['AC'])
        df['VAC'] = df['BAC'] - df['EAC']
    
    def compare_scenarios(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare key metrics across all scenarios"""
        if metrics is None:
            metrics = ['CPI', 'SPI', 'EAC', 'VAC']
            
        comparison = pd.DataFrame()
        
        # Add base scenario
        for metric in metrics:
            comparison[f'Base_{metric}'] = self.base_dataset[metric]
        
        # Add other scenarios
        for scenario_name, scenario_data in self.scenarios.items():
            for metric in metrics:
                comparison[f'{scenario_name}_{metric}'] = scenario_data[metric]
        
        return comparison
