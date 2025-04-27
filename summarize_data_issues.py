import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the folder containing raw datasets
downloads_dir = Path.home() / "Downloads" / "building data"
output_report = "data_issues_summary.md"

# List of energy datasets with meter readings
energy_datasets = [
    "chilledwater.csv",
    "electricity.csv",
    "gas.csv",
    "hotwater.csv",
    "solar.csv",
    "irrigation.csv",
    "steam.csv",
    "water.csv"
]

# List of non-energy datasets
non_energy_datasets = ["metadata.csv", "weather.csv"]

# Initialize lists to store results
results = []

# Function to analyze a single energy dataset for missing and negative values
def analyze_energy_dataset(file_path, dataset_name, timeout=300):
    print(f"Processing {dataset_name}...")
    
    # Use Dask for large files, pandas for smaller ones
    try:
        # Read with Dask, small blocksize for memory efficiency
        print(f"{dataset_name}: Loading dataset...")
        df = dd.read_csv(file_path, blocksize="4MB", assume_missing=True)
        is_large_file = True
    except Exception as e:
        print(f"Dask failed for {dataset_name}: {e}. Trying pandas...")
        try:
            df = pd.read_csv(file_path)
            is_large_file = False
        except Exception as e:
            print(f"Pandas failed for {dataset_name}: {e}. Skipping...")
            return None
    
    # Assume first column is 'timestamp', others are building IDs
    if 'timestamp' not in df.columns:
        print(f"Warning: 'timestamp' column not found in {dataset_name}. Skipping...")
        return None
    
    # Get building columns (all except 'timestamp')
    building_cols = [col for col in df.columns if col != 'timestamp']
    if not building_cols:
        print(f"Warning: No building columns found in {dataset_name}. Skipping...")
        return None
    
    # Compute metrics
    print(f"{dataset_name}: Computing total readings and missing values...")
    start_time = time.time()
    try:
        total_readings = df[building_cols].size.compute() if is_large_file else df[building_cols].size
        missing_count = df[building_cols].isna().sum().sum().compute() if is_large_file else df[building_cols].isna().sum().sum()
        negative_count = (df[building_cols] < 0).sum().sum().compute() if is_large_file else (df[building_cols] < 0).sum().sum()
        
        if time.time() - start_time > timeout:
            print(f"Warning: Timeout while computing metrics for {dataset_name}. Skipping...")
            return None
    except Exception as e:
        print(f"Error computing metrics for {dataset_name}: {e}. Skipping...")
        return None
    
    # Calculate percentages
    missing_pct = (missing_count / total_readings * 100) if total_readings > 0 else 0
    negative_pct = (negative_count / total_readings * 100) if total_readings > 0 else 0
    
    # Clear Dask computation graph
    if is_large_file:
        df = None
        import gc
        gc.collect()
    
    return {
        "Dataset": dataset_name,
        "Total Readings": total_readings,
        "Missing Values (%)": round(missing_pct, 2),
        "Negative Values (%)": round(negative_pct, 2)
    }

# Function to check non-energy datasets for missing values
def analyze_non_energy_dataset(file_path, dataset_name):
    print(f"Processing {dataset_name}...")
    
    # Use pandas for metadata and weather (typically smaller)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {dataset_name}: {e}. Skipping...")
        return None
    
    total_cells = df.size
    missing_count = df.isna().sum().sum()
    missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
    
    return {
        "Dataset": dataset_name,
        "Total Readings": total_cells,
        "Missing Values (%)": round(missing_pct, 2),
        "Negative Values (%)": "N/A"
    }

# Process energy datasets
for dataset in energy_datasets:
    file_path = downloads_dir / dataset
    if file_path.exists():
        result = analyze_energy_dataset(file_path, dataset)
        if result:
            results.append(result)
    else:
        print(f"Warning: {dataset} not found in {downloads_dir}")

# Process non-energy datasets
for dataset in non_energy_datasets:
    file_path = downloads_dir / dataset
    if file_path.exists():
        result = analyze_non_energy_dataset(file_path, dataset)
        if result:
            results.append(result)
    else:
        print(f"Warning: {dataset} not found in {downloads_dir}")

# Create a summary report
with open(output_report, "w") as f:
    f.write("# Data Issues Summary Report\n\n")
    f.write("This report summarizes the percentage of missing and negative values for energy datasets (meter readings) and missing values for non-energy datasets (metadata and weather). For energy datasets, metrics are calculated across building columns (excluding 'timestamp').\n\n")
    
    # Write results as a table
    f.write("## Summary Table\n\n")
    f.write("| Dataset | Total Readings | Missing Values (%) | Negative Values (%) |\n")
    f.write("|---------|----------------|--------------------|---------------------|\n")
    for result in results:
        f.write(f"| {result['Dataset']} | {result['Total Readings']} | {result['Missing Values (%)']} | {result['Negative Values (%)']} |\n")
    
    f.write("\n## Notes\n")
    f.write("- **Energy Datasets**: Metrics are calculated across building columns (excluding 'timestamp').\n")
    f.write("- **Non-Energy Datasets**: Only missing values are reported, as negative values are not applicable.\n")
    f.write("- Total Readings: Number of meter readings (rows × building columns) for energy datasets, or total cells for non-energy datasets.\n")
    f.write("- Missing Values: Percentage of NaN values.\n")
    f.write("- Negative Values: Percentage of readings < 0.\n")
    f.write("- Datasets were processed using Dask for large files and pandas for smaller ones.\n")

# Convert results to DataFrame for display
results_df = pd.DataFrame(results)
print("\nSummary of Data Issues:")
print(results_df)

# Save results as CSV for further use
results_df.to_csv("data_issues_summary.csv", index=False)
print(f"\nReport saved as {output_report} and data_issues_summary.csv")
