import os
import sys
from typing import cast
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from dotenv import load_dotenv  # type: ignore
from crewai import Crew, Agent, Task  # type: ignore
from langchain.tools import Tool  # type: ignore

def eprint(*args, **kwargs):
    kwargs.pop('exc_info', None)  # Remove exc_info if present
    print(*args, file=sys.stderr, **kwargs)

try:
    eprint("Starting script...")

    # Load environment variables
    load_dotenv()
    eprint("Loaded environment variables")

    # Configure API keys
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = cast(str, api_key)
        eprint("Configured OpenAI API key")
    else:
        eprint("Warning: OPENAI_API_KEY not found in environment variables")
        raise ValueError("OPENAI_API_KEY is required but was not found")

    # Paths
    BASE_DIR = r'c:/cc-working-dir/AI Agent DECM Dataset Gen'
    METRICS_FILE = os.path.join(
        BASE_DIR,
        'DCMA EVMS Compliance Metrics v6.0 20221205.xlsx'
    )
    OUTPUT_DIR = os.path.join(
        BASE_DIR,
        'dataset_generation_crew',
        'generated_datasets'
    )

    eprint(f"METRICS_FILE path: {METRICS_FILE}")
    eprint(f"OUTPUT_DIR path: {OUTPUT_DIR}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eprint("Created output directory")

    def analyze_evms_metrics(query: str) -> str:
        eprint(f"Analyzing EVMS metrics with query: {query}")
        try:
            df = pd.read_excel(METRICS_FILE)
            msg = f"Analysis complete: Found {len(df)} metrics"
            eprint(msg)
            return msg
        except Exception as e:
            error_msg = f"Error analyzing metrics: {str(e)}"
            eprint(error_msg)
            return error_msg

    def generate_dataset(complexity: str) -> str:
        eprint(f"Generating {complexity} dataset...")
        try:
            df = generate_synthetic_evms_dataset(complexity)
            msg = (
                f"Successfully generated {complexity} dataset "
                f"with {len(df)} records"
            )
            eprint(msg)
            return msg
        except Exception as e:
            error_msg = f"Error generating dataset: {str(e)}"
            eprint(error_msg)
            return error_msg

    def validate_dataset(filename: str) -> str:
        eprint(f"Validating dataset: {filename}")
        try:
            filepath = os.path.join(OUTPUT_DIR, filename)
            df = pd.read_csv(filepath)
            msg = f"Validation complete: Dataset contains {len(df)} records"
            eprint(msg)
            return msg
        except Exception as e:
            error_msg = f"Error validating dataset: {str(e)}"
            eprint(error_msg)
            return error_msg

    # Create tools using langchain Tool
    analyze_tool = Tool(
        name="analyze_evms_metrics",
        func=analyze_evms_metrics,
        description="Analyze EVMS metrics based on the query"
    )

    generate_tool = Tool(
        name="generate_dataset",
        func=generate_dataset,
        description="Generate a synthetic EVMS dataset with specified complexity"
    )

    validate_tool = Tool(
        name="validate_dataset",
        func=validate_dataset,
        description="Validate a generated EVMS dataset"
    )

    eprint("Creating agents...")

    # Define Agents
    evms_research_agent = Agent(
        role='EVMS Research Specialist',
        goal='Analyze EVMS metrics and prepare for dataset generation',
        backstory='Expert in project management and EVMS principles',
        verbose=True,
        allow_delegation=True,
        tools=[analyze_tool]
    )

    evms_dataset_generator = Agent(
        role='EVMS Synthetic Dataset Creator',
        goal='Generate realistic synthetic EVMS datasets',
        backstory='Data engineer specializing in synthetic datasets',
        verbose=True,
        allow_delegation=True,
        tools=[generate_tool]
    )

    evms_dataset_validator = Agent(
        role='EVMS Dataset Quality Assurance Specialist',
        goal='Validate datasets for statistical integrity',
        backstory='Data scientist focused on EVMS standards',
        verbose=True,
        allow_delegation=True,
        tools=[validate_tool]
    )

    eprint("Created all agents")

    def load_existing_metrics() -> tuple[pd.DataFrame, str] | None:
        eprint("Loading existing metrics...")
        try:
            df = pd.read_excel(METRICS_FILE)
            query = 'Analyze key performance indicators from the EVMS data.'
            analysis_result = analyze_evms_metrics(query)
            eprint("Successfully loaded existing metrics")
            return df, analysis_result
        except Exception as e:
            eprint(f"Error loading metrics file: {e}")
            return None

    # Tasks
    eprint("Creating tasks...")

    research_task = Task(
        description="""
        1. Load and analyze the DCMA EVMS Compliance Metrics file
        2. Extract key performance indicators and metrics
        3. Document the relationships between different metrics
        4. Identify critical metrics for synthetic data generation
        """,
        agent=evms_research_agent,
        expected_output="Analysis report of DCMA EVMS Compliance Metrics"
    )

    generation_task = Task(
        description="""
        Generate two synthetic EVMS datasets:
        1. Simple dataset (10 projects, 12 periods)
        2. Complex dataset (50 projects, 24 periods)
        
        Include metrics: PV, EV, AC, SV, CV, SPI, CPI
        Ensure realistic values and relationships.
        """,
        agent=evms_dataset_generator,
        expected_output="Two synthetic EVMS datasets: simple and complex"
    )

    validation_task = Task(
        description="""
        For each generated dataset:
        1. Verify data completeness and structure
        2. Check metric calculations and relationships
        3. Validate statistical distributions
        4. Ensure DCMA compliance
        5. Generate validation report
        """,
        agent=evms_dataset_validator,
        expected_output="Validation report for generated datasets"
    )

    eprint("Created all tasks")

    def generate_synthetic_evms_dataset(
        complexity: str = 'simple'
    ) -> pd.DataFrame:
        eprint(f"Generating synthetic EVMS dataset with {complexity} complexity...")
        np.random.seed(42)
        
        num_projects = 10 if complexity == 'simple' else 50
        time_periods = 12 if complexity == 'simple' else 24
        
        data = []
        
        for project_id in range(num_projects):
            budget = np.random.uniform(100000, 10000000)
            planned_duration = np.random.randint(6, 36)
            
            for period in range(time_periods):
                planned_value = budget * (period / time_periods)
                earned_value = planned_value * np.random.uniform(0.8, 1.2)
                actual_cost = earned_value * np.random.uniform(0.9, 1.1)
                
                schedule_variance = earned_value - planned_value
                cost_variance = earned_value - actual_cost
                spi = earned_value / planned_value if planned_value != 0 else 1
                cpi = earned_value / actual_cost if actual_cost != 0 else 1
                
                row = {
                    'Project_ID': project_id,
                    'Period': period,
                    'Budget': budget,
                    'Planned_Duration': planned_duration,
                    'Planned_Value': planned_value,
                    'Earned_Value': earned_value,
                    'Actual_Cost': actual_cost,
                    'Schedule_Variance': schedule_variance,
                    'Cost_Variance': cost_variance,
                    'Schedule_Performance_Index': spi,
                    'Cost_Performance_Index': cpi
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        filename = f'evms_dataset_{complexity}_complexity.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        eprint(f"Generated {filename}")
        return df

    eprint("Creating crew...")

    # Create Crew
    evms_dataset_crew = Crew(
        agents=[
            evms_research_agent,
            evms_dataset_generator,
            evms_dataset_validator
        ],
        tasks=[research_task, generation_task, validation_task],
        verbose=True
    )

    eprint("Created crew")

    def main() -> None:
        eprint("Starting main execution...")
        try:
            eprint("Starting crew kickoff...")
            evms_dataset_crew.kickoff()
            eprint("Crew Research and Validation Complete.")
            
            eprint("Generating Synthetic EVMS Datasets...")
            generate_synthetic_evms_dataset('simple')
            generate_synthetic_evms_dataset('complex')
            
            eprint("Dataset Generation Complete!")
        except Exception as e:
            eprint(f"Error during execution: {str(e)}")
            raise

    if __name__ == "__main__":
        main()

except Exception as e:
    eprint(f"Critical error: {str(e)}")
    raise
