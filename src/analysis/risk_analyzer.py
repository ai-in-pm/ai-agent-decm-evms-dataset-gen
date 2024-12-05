import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm

@dataclass
class RiskMetrics:
    probability: float
    impact: float
    severity: float
    mitigation_cost: float

class RiskAnalyzer:
    """Advanced risk analysis for EVMS metrics"""
    
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.risk_categories = {
            'schedule': ['SPI', 'SV'],
            'cost': ['CPI', 'CV'],
            'performance': ['EAC', 'VAC']
        }
        
    def calculate_risk_exposure(self) -> Dict[str, RiskMetrics]:
        """Calculate risk exposure for each category"""
        risk_exposure = {}
        
        for category, metrics in self.risk_categories.items():
            # Calculate probability based on metric volatility
            volatility = np.mean([self.dataset[m].std() / abs(self.dataset[m].mean()) 
                                for m in metrics])
            probability = norm.cdf(volatility, loc=0, scale=0.1)
            
            # Calculate impact based on worst-case deviation
            impact = np.mean([abs(self.dataset[m].min() - self.dataset[m].mean()) / 
                            abs(self.dataset[m].mean()) for m in metrics])
            
            # Calculate severity (probability * impact)
            severity = probability * impact
            
            # Estimate mitigation cost based on severity
            mitigation_cost = severity * self.dataset['BAC'].iloc[0] * 0.1
            
            risk_exposure[category] = RiskMetrics(
                probability=probability,
                impact=impact,
                severity=severity,
                mitigation_cost=mitigation_cost
            )
            
        return risk_exposure
    
    def identify_risk_triggers(self) -> Dict[str, List[Dict]]:
        """Identify specific events that could trigger risks"""
        triggers = {}
        
        for category, metrics in self.risk_categories.items():
            category_triggers = []
            
            for metric in metrics:
                # Detect significant deviations
                mean_value = self.dataset[metric].mean()
                std_dev = self.dataset[metric].std()
                threshold = 2 * std_dev  # 2 sigma events
                
                deviations = self.dataset[self.dataset[metric].abs() > threshold]
                
                if not deviations.empty:
                    for _, row in deviations.iterrows():
                        trigger = {
                            'metric': metric,
                            'date': row['Date'],
                            'value': row[metric],
                            'threshold': threshold,
                            'severity': abs(row[metric] - mean_value) / std_dev
                        }
                        category_triggers.append(trigger)
            
            triggers[category] = category_triggers
            
        return triggers
    
    def generate_risk_mitigation_strategies(self, 
                                          risk_exposure: Dict[str, RiskMetrics]
                                          ) -> Dict[str, List[Dict]]:
        """Generate risk mitigation strategies based on exposure"""
        strategies = {}
        
        for category, metrics in risk_exposure.items():
            category_strategies = []
            
            if metrics.severity > 0.7:  # High severity
                category_strategies.extend([
                    {
                        'action': 'Immediate escalation to senior management',
                        'cost': metrics.mitigation_cost * 0.3,
                        'priority': 'High'
                    },
                    {
                        'action': 'Develop detailed contingency plan',
                        'cost': metrics.mitigation_cost * 0.2,
                        'priority': 'High'
                    }
                ])
            elif metrics.severity > 0.4:  # Medium severity
                category_strategies.extend([
                    {
                        'action': 'Increase monitoring frequency',
                        'cost': metrics.mitigation_cost * 0.1,
                        'priority': 'Medium'
                    },
                    {
                        'action': 'Review and update risk registers',
                        'cost': metrics.mitigation_cost * 0.05,
                        'priority': 'Medium'
                    }
                ])
            else:  # Low severity
                category_strategies.extend([
                    {
                        'action': 'Regular monitoring',
                        'cost': metrics.mitigation_cost * 0.02,
                        'priority': 'Low'
                    }
                ])
                
            strategies[category] = category_strategies
            
        return strategies
    
    def calculate_confidence_intervals(self, 
                                    confidence_level: float = 0.95
                                    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for key metrics"""
        intervals = {}
        
        for category, metrics in self.risk_categories.items():
            category_intervals = {}
            
            for metric in metrics:
                mean = self.dataset[metric].mean()
                std_err = self.dataset[metric].std() / np.sqrt(len(self.dataset))
                z_score = norm.ppf((1 + confidence_level) / 2)
                
                margin = z_score * std_err
                category_intervals[metric] = {
                    'mean': mean,
                    'lower_bound': mean - margin,
                    'upper_bound': mean + margin,
                    'confidence_level': confidence_level
                }
                
            intervals[category] = category_intervals
            
        return intervals
