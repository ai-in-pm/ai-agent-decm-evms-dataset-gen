import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
from datetime import datetime
import os
from ..analysis.risk_analyzer import RiskAnalyzer
from ..simulation.monte_carlo import SimulationResults

class ReportGenerator:
    """Generate comprehensive EVMS reports with visualizations"""
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_performance_report(self,
                                  dataset: pd.DataFrame,
                                  risk_analysis: Dict[str, any],
                                  simulation_results: SimulationResults,
                                  report_date: Optional[datetime] = None) -> str:
        """Generate a comprehensive performance report"""
        report_date = report_date or datetime.now()
        report_filename = f"evms_report_{report_date.strftime('%Y%m%d')}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Create report figures
        figs = []
        
        # 1. Performance Metrics Overview
        fig_metrics = self._create_metrics_overview(dataset)
        figs.append(('Performance Metrics Overview', fig_metrics))
        
        # 2. Risk Analysis
        fig_risk = self._create_risk_analysis_plot(risk_analysis)
        figs.append(('Risk Analysis', fig_risk))
        
        # 3. Monte Carlo Simulation Results
        fig_simulation = self._create_simulation_plot(simulation_results)
        figs.append(('Monte Carlo Simulation', fig_simulation))
        
        # 4. Compliance Status
        fig_compliance = self._create_compliance_plot(dataset)
        figs.append(('Compliance Status', fig_compliance))
        
        # Generate HTML report
        self._generate_html_report(report_path, figs, dataset, 
                                 risk_analysis, simulation_results)
        
        return report_path
    
    def _create_metrics_overview(self, dataset: pd.DataFrame) -> go.Figure:
        """Create overview plot of key performance metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Earned Value Metrics', 'Performance Indices',
                          'Variances', 'EAC Trend')
        )
        
        # Earned Value Metrics
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['PV'],
                      name='Planned Value', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['EV'],
                      name='Earned Value', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['AC'],
                      name='Actual Cost', line=dict(color='red')),
            row=1, col=1
        )
        
        # Performance Indices
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['CPI'],
                      name='CPI', line=dict(color='purple')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['SPI'],
                      name='SPI', line=dict(color='orange')),
            row=1, col=2
        )
        
        # Variances
        fig.add_trace(
            go.Bar(x=dataset['Date'], y=dataset['CV'],
                  name='Cost Variance', marker_color='blue'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=dataset['Date'], y=dataset['SV'],
                  name='Schedule Variance', marker_color='green'),
            row=2, col=1
        )
        
        # EAC Trend
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['EAC'],
                      name='EAC', line=dict(color='red')),
            row=2, col=2
        )
        fig.add_hline(y=dataset['BAC'].iloc[0], line_dash='dash',
                     annotation_text='BAC', row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True)
        return fig
    
    def _create_risk_analysis_plot(self, risk_analysis: Dict) -> go.Figure:
        """Create risk analysis visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Risk Exposure', 'Risk Triggers')
        )
        
        # Risk Exposure
        categories = list(risk_analysis['exposure'].keys())
        probabilities = [m.probability for m in risk_analysis['exposure'].values()]
        impacts = [m.impact for m in risk_analysis['exposure'].values()]
        
        fig.add_trace(
            go.Scatter(
                x=probabilities,
                y=impacts,
                mode='markers+text',
                marker=dict(size=15),
                text=categories,
                name='Risk Categories'
            ),
            row=1, col=1
        )
        
        # Risk Triggers
        triggers = risk_analysis['triggers']
        for category in triggers:
            if triggers[category]:
                dates = [t['date'] for t in triggers[category]]
                severities = [t['severity'] for t in triggers[category]]
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=severities,
                        mode='markers',
                        name=f'{category} Triggers'
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(height=400, showlegend=True)
        return fig
    
    def _create_simulation_plot(self, simulation_results: SimulationResults) -> go.Figure:
        """Create Monte Carlo simulation results visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('EAC Distribution', 'CPI Distribution',
                          'Probability Analysis', 'Sensitivity Analysis')
        )
        
        # EAC Distribution
        eac_percentiles = list(simulation_results.percentiles['EAC'].values())
        fig.add_trace(
            go.Box(y=eac_percentiles, name='EAC'),
            row=1, col=1
        )
        
        # CPI Distribution
        cpi_percentiles = list(simulation_results.percentiles['CPI'].values())
        fig.add_trace(
            go.Box(y=cpi_percentiles, name='CPI'),
            row=1, col=2
        )
        
        # Probability Analysis
        probs = simulation_results.probabilities
        prob_categories = []
        prob_values = []
        for metric, events in probs.items():
            for event, prob in events.items():
                prob_categories.append(f"{metric}-{event}")
                prob_values.append(prob)
                
        fig.add_trace(
            go.Bar(x=prob_categories, y=prob_values),
            row=2, col=1
        )
        
        # Sensitivity Analysis
        sensitivities = simulation_results.sensitivity
        fig.add_trace(
            go.Bar(
                x=list(sensitivities.keys()),
                y=list(sensitivities.values()),
                name='Sensitivity'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig
    
    def _create_compliance_plot(self, dataset: pd.DataFrame) -> go.Figure:
        """Create compliance status visualization"""
        fig = go.Figure()
        
        # Add CPI compliance bounds
        fig.add_hline(y=0.95, line_dash='dash', line_color='red',
                     annotation_text='Lower Bound')
        fig.add_hline(y=1.05, line_dash='dash', line_color='red',
                     annotation_text='Upper Bound')
        
        # Plot CPI trend
        fig.add_trace(
            go.Scatter(x=dataset['Date'], y=dataset['CPI'],
                      name='CPI', line=dict(color='blue'))
        )
        
        fig.update_layout(
            title='CPI Compliance Tracking',
            yaxis_title='CPI Value',
            showlegend=True
        )
        
        return fig
    
    def _generate_html_report(self,
                            report_path: str,
                            figures: List[tuple],
                            dataset: pd.DataFrame,
                            risk_analysis: Dict,
                            simulation_results: SimulationResults) -> None:
        """Generate HTML report with all visualizations and analysis"""
        with open(report_path, 'w', encoding='utf-8') as f:
            # Write HTML header
            f.write('''
            <html>
            <head>
                <title>EVMS Performance Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin-bottom: 30px; }
                    .metric-card {
                        border: 1px solid #ddd;
                        padding: 15px;
                        margin: 10px;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
            ''')
            
            # Write report header
            f.write(f'''
            <h1>EVMS Performance Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            ''')
            
            # Add key metrics summary
            f.write('''
            <div class="section">
                <h2>Key Metrics Summary</h2>
                <div style="display: flex; flex-wrap: wrap;">
            ''')
            
            latest = dataset.iloc[-1]
            metrics = {
                'CPI': latest['CPI'],
                'SPI': latest['SPI'],
                'EAC': latest['EAC'],
                'VAC': latest['VAC']
            }
            
            for name, value in metrics.items():
                f.write(f'''
                <div class="metric-card">
                    <h3>{name}</h3>
                    <p>{value:.2f}</p>
                </div>
                ''')
            
            f.write('</div></div>')
            
            # Add figures
            for title, fig in figures:
                f.write(f'<div class="section"><h2>{title}</h2>')
                f.write(fig.to_html(full_html=False))
                f.write('</div>')
            
            # Add risk analysis summary
            f.write('''
            <div class="section">
                <h2>Risk Analysis Summary</h2>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr><th>Category</th><th>Probability</th><th>Impact</th><th>Severity</th></tr>
            ''')
            
            for category, metrics in risk_analysis['exposure'].items():
                f.write(f'''
                <tr>
                    <td>{category}</td>
                    <td>{metrics.probability:.2f}</td>
                    <td>{metrics.impact:.2f}</td>
                    <td>{metrics.severity:.2f}</td>
                </tr>
                ''')
            
            f.write('</table></div>')
            
            # Add simulation results summary
            f.write('''
            <div class="section">
                <h2>Simulation Results Summary</h2>
                <h3>Probability Analysis</h3>
                <ul>
            ''')
            
            for metric, events in simulation_results.probabilities.items():
                f.write(f'<li>{metric}:<ul>')
                for event, prob in events.items():
                    f.write(f'<li>{event}: {prob:.1%}</li>')
                f.write('</ul></li>')
            
            f.write('</ul></div>')
            
            # Close HTML
            f.write('</body></html>')
