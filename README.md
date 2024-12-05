# DCMA EVMS Dataset Generator

An AI-powered tool for generating realistic Earned Value Management System (EVMS) datasets compliant with Defense Contract Management Agency (DCMA) guidelines.

This repository was inspired by the Defense Contract Management Agency (DCMA) Earned Value Management System, which is open source and can be found by visiting this website: https://www.dcma.mil/HQ/EVMS/ 

## Features

- Generate realistic EVMS metrics following DCMA's 32 guidelines
- Customizable project parameters and risk scenarios
- Time-phased data with multiple reporting periods
- Advanced forecasting and variance analysis
- Interactive dashboard for visualization
- Export capabilities to various formats (CSV, Excel, JSON)

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code for the dataset generator
  - `core/`: Core EVMS calculation modules
  - `generators/`: Dataset generation logic
  - `validation/`: DCMA compliance validation
  - `visualization/`: Plotting and dashboard components
- `config/`: Configuration files for different project scenarios
- `examples/`: Example datasets and usage scenarios
- `tests/`: Unit tests and validation scripts

## Usage

```python
from dcma_evms_gen import DatasetGenerator

# Initialize generator with project parameters
generator = DatasetGenerator(
    contract_value=1000000,
    duration_months=24,
    risk_level="medium"
)

# Generate dataset
dataset = generator.generate()

# Export to desired format
dataset.to_csv("evms_data.csv")
```

## Disclaimer

This tool generates simulated datasets for analysis and training purposes only. Generated data must be validated before use in official government reporting.
