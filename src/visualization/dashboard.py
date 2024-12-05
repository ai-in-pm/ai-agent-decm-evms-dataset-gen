import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

class EVMSDashboard:
    """Interactive dashboard for EVMS data visualization"""
    
    def __init__(self):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self._initialize_data()
        self._setup_layout()
        self._setup_callbacks()
        
    def _initialize_data(self):
        """Initialize sample EVMS data"""
        try:
            self.df = pd.read_csv('base_evms_data.csv')
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Create sample data if file loading fails
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
            n_points = len(dates)
            
            # Sample metrics
            pv = np.linspace(0, 1000000, n_points)  # Planned Value
            ev = pv * np.random.uniform(0.8, 1.2, n_points)  # Earned Value
            ac = ev * np.random.uniform(0.9, 1.1, n_points)  # Actual Cost
            
            # Create dataframe
            self.df = pd.DataFrame({
                'Date': dates,
                'PV': pv,
                'EV': ev,
                'AC': ac,
                'BAC': [1000000] * n_points,
                'CPI': ev / ac,
                'SPI': ev / pv,
                'CV': ev - ac,
                'SV': ev - pv
            })

    def _create_chart_card(self, title, chart_id, width=12):
        """Create a responsive card containing a chart"""
        return dbc.Col([
            dbc.Card([
                dbc.CardHeader(title, className="h5"),
                dbc.CardBody([
                    dcc.Graph(
                        id=chart_id,
                        config={'responsive': True},
                        style={'height': '100%'}
                    )
                ], style={'minHeight': '400px'})
            ], className="h-100 shadow-sm")
        ], width=width)

    def _setup_layout(self):
        """Create the responsive dashboard layout"""
        self.app.layout = dbc.Container([
            # Navigation bar
            dbc.Navbar(
                dbc.Container(
                    html.A(
                        dbc.Row([
                            dbc.Col(dbc.NavbarBrand("DCMA EVMS Dashboard", className="ms-2")),
                        ],
                        align="center",
                        className="g-0",
                        ),
                        style={"textDecoration": "none"},
                    )
                ),
                color="primary",
                dark=True,
                className="mb-4",
            ),

            # Store for data
            dcc.Store(id='stored-data'),
            
            # Main content
            dbc.Row([
                # Performance Metrics
                self._create_chart_card(
                    "Performance Metrics",
                    "performance-chart",
                    width=12
                ),
            ], className="mb-4"),
            
            dbc.Row([
                # Variance Analysis
                self._create_chart_card(
                    "Variance Analysis",
                    "variance-chart",
                    width=6
                ),
                
                # Performance Indices
                self._create_chart_card(
                    "Performance Indices",
                    "indices-chart",
                    width=6
                ),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("DCMA Compliance Status", className="h5"),
                        dbc.CardBody(
                            id='compliance-status',
                            className="d-flex flex-wrap justify-content-around"
                        )
                    ], className="shadow-sm")
                ], width=12, className="mb-4")
            ]),
            
            # Footer
            dbc.Row([
                dbc.Col(
                    html.Footer(
                        "EVMS Dashboard 2024",
                        className="text-center text-muted py-3"
                    ),
                    width=12
                )
            ])
        ], fluid=True, className="px-4")
        
    def _setup_callbacks(self):
        """Setup interactive callbacks"""
        @self.app.callback(
            Output('stored-data', 'data'),
            Input('stored-data', 'id')
        )
        def initialize_data(_):
            return self.df.to_dict('records')
            
        @self.app.callback(
            Output('performance-chart', 'figure'),
            Input('stored-data', 'data')
        )
        def update_performance_chart(data):
            if not data:
                return go.Figure()
                
            df = pd.DataFrame(data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['PV'], name='Planned Value'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EV'], name='Earned Value'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['AC'], name='Actual Cost'))
            
            fig.update_layout(
                title='EVMS Performance Metrics',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                hovermode='x unified',
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        @self.app.callback(
            Output('variance-chart', 'figure'),
            Input('stored-data', 'data')
        )
        def update_variance_chart(data):
            if not data:
                return go.Figure()
                
            df = pd.DataFrame(data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['Date'], y=df['CV'], name='Cost Variance'))
            fig.add_trace(go.Bar(x=df['Date'], y=df['SV'], name='Schedule Variance'))
            
            fig.update_layout(
                title='Cost and Schedule Variances',
                xaxis_title='Date',
                yaxis_title='Variance ($)',
                barmode='group',
                hovermode='x unified',
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        @self.app.callback(
            Output('indices-chart', 'figure'),
            Input('stored-data', 'data')
        )
        def update_indices_chart(data):
            if not data:
                return go.Figure()
                
            df = pd.DataFrame(data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['CPI'], name='Cost Performance Index (CPI)'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SPI'], name='Schedule Performance Index (SPI)'))
            
            fig.update_layout(
                title='Performance Indices',
                xaxis_title='Date',
                yaxis_title='Index',
                hovermode='x unified',
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        @self.app.callback(
            Output('compliance-status', 'children'),
            Input('stored-data', 'data')
        )
        def update_compliance_status(data):
            if not data:
                return html.Div()
                
            df = pd.DataFrame(data)
            
            # Check DCMA compliance thresholds
            cpi_compliant = (df['CPI'] >= 0.95).all() and (df['CPI'] <= 1.05).all()
            spi_compliant = (df['SPI'] >= 0.95).all() and (df['SPI'] <= 1.05).all()
            
            return [
                dbc.Card([
                    dbc.CardBody([
                        html.H4("CPI Status", className="text-center mb-2"),
                        html.Div(
                            "Within Thresholds" if cpi_compliant else "Outside Thresholds",
                            className=f"text-center {'text-success' if cpi_compliant else 'text-danger'}"
                        ),
                        html.P(f"Current: {df['CPI'].iloc[-1]:.2f}", className="text-center mt-2 mb-0")
                    ])
                ], className="m-2 flex-grow-1", style={"minWidth": "200px"}),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H4("SPI Status", className="text-center mb-2"),
                        html.Div(
                            "Within Thresholds" if spi_compliant else "Outside Thresholds",
                            className=f"text-center {'text-success' if spi_compliant else 'text-danger'}"
                        ),
                        html.P(f"Current: {df['SPI'].iloc[-1]:.2f}", className="text-center mt-2 mb-0")
                    ])
                ], className="m-2 flex-grow-1", style={"minWidth": "200px"})
            ]
    
    def run_server(self, debug: bool = True, port: int = 8501):
        """Run the dashboard server"""
        try:
            print(f"Starting dashboard server at http://127.0.0.1:{port}")
            print("Press Ctrl+C to stop the server")
            self.app.run_server(
                debug=debug,
                port=port,
                host='127.0.0.1',  # Use explicit localhost IP
                use_reloader=False,  # Disable reloader to prevent duplicate processes
                dev_tools_hot_reload=False  # Disable hot reloading
            )
        except Exception as e:
            print(f"Error starting server: {str(e)}")
            raise
