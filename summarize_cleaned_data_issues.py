import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the folder containing cleaned datasets
downloads_dir = Path.home() / "Downloads" / "building data"
output_report = "cleaned_data_issues_summary.md"

# List of cleaned energy datasets (excluding electricity, handled separately)
energy_datasets = [
    "cleaned_chilledwater.csv",
    "cleaned_gas.csv",
    "cleaned_hotwater.csv",
    "cleaned_solar.csv",
    "cleaned_irrigation.csv",
    "cleaned_steam.csv",
    "cleaned_water.csv"
]

# Electricity is split into two files
electricity_files = ["cleaned_electricity-0.csv", "cleaned_electricity-1.csv"]

# List of non-energy datasets
non_energy_datasets = ["cleaned_metadata.csv", "cleaned_weather.csv"]

# Initialize lists to store results
results = []

# Function to analyze a single energy dataset (melted format) for missing and negative values
def analyze_energy_dataset(file_path, dataset_name, timeout=300, is_electricity=False):
    print(f"Processing {dataset_name}...")
    
    # Use Dask for large files, pandas for smaller ones
    try:
        print(f"{dataset_name}: Loading dataset...")
        if is_electricity:
            # Combine electricity files
            df = dd.read_csv([downloads_dir / f for f in file_path], blocksize="4MB", assume_missing=True)
        else:
            df = dd.read_csv(file_path, blocksize="4MB", assume_missing=True)
        is_large_file = True
    except Exception as e:
        print(f"Dask failed for {dataset_name}: {e}. Trying pandas...")
        try:
            if is_electricity:
                df_list = [pd.read_csv(downloads_dir / f) for f in file_path]
                df = pd.concat(df_list, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
            is_large_file = False
        except Exception as e:
            print(f"Pandas failed for {dataset_name}: {e}. Skipping...")
            return None
    
    # Check for required columns
    required_cols = ['timestamp', 'building_id', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns {missing_cols} in {dataset_name}. Skipping...")
        return None
    
    # Convert 'value' column to numeric, coercing errors to NaN
    print(f"{dataset_name}: Converting 'value' column to numeric...")
    try:
        if is_large_file:
            df['value'] = dd.to_numeric(df['value'], errors='coerce')
        else:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
    except Exception as e:
        print(f"Error converting 'value' column to numeric for {dataset_name}: {e}. Skipping...")
        return None
    
    # Compute metrics
    print(f"{dataset_name}: Computing total readings and missing values...")
    start_time = time.time()
    try:
        total_readings = df['value'].size.compute() if is_large_file else df['value'].size
        missing_count = df['value'].isna().sum().compute() if is_large_file else df['value'].isna().sum()
        negative_count = (df['value'] < 0).sum().compute() if is_large_file else (df['value'] < 0).sum()
        
        if time.time() - start_time > timeout:
            print(f"Warning: Timeout while computing metrics for {dataset_name}. Skipping...")
            return None
    except Exception as e:
        print(f"Error computing metrics for {dataset_name}: {e}. Skipping...")
        return None
    
    # Calculate percentages
    missing_pct = (missing_count / total_readings * 100) if total_readings > 0 else 0
    negative_pct = (negative_count / total_readings * 100) if total_readings > 0 else 0
    
    # Log non-numeric values (for debugging)
    if missing_count > 0:
        print(f"{dataset_name}: {missing_count} non-numeric or missing values in 'value' column")
    
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

# Process electricity datasets (special case)
if all((downloads_dir / f).exists() for f in electricity_files):
    result = analyze_energy_dataset(electricity_files, "cleaned_electricity.csv", is_electricity=True)
    if result:
        results.append(result)
else:
    print(f"Warning: One or both electricity files ({electricity_files}) not found in {downloads_dir}")

# Process other energy datasets
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
    f.write("# Cleaned Data Issues Summary Report\n\n")
    f.write("This report summarizes the percentage of missing and negative values for cleaned energy datasets (melted format: timestamp, building_id, value) and missing values for cleaned non-energy datasets (metadata and weather).\n\n")
    
    # Write results as a table
    f.write("## Summary Table\n\n")
    f.write("| Dataset | Total Readings | Missing Values (%) | Negative Values (%) |\n")
    f.write("|---------|----------------|--------------------|---------------------|\n")
    for result in results:
        f.write(f"| {result['Dataset']} | {result['Total Readings']} | {result['Missing Values (%)']} | {result['Negative Values (%)']} |\n")
    
    f.write("\n## Notes\n")
    f.write("- **Energy Datasets**: Metrics are calculated for the 'value' column in melted format (timestamp, building_id, value). Electricity data combines cleaned_electricity-0.csv and cleaned_electricity-1.csv.\n")
    f.write("- **Non-Energy Datasets**: Only missing values are reported, as negative values are not applicable.\n")
    f.write("- Total Readings: Number of rows in the 'value' column for energy datasets, or total cells for non-energy datasets.\n")
    f.write("- Missing Values: Percentage of NaN values in the 'value' column.\n")
    f.write("- Negative Values: Percentage of 'value' < 0.\n")
    f.write("- Datasets were processed using Dask for large files and pandas for smaller ones.\n")

# Convert results to DataFrame for display
results_df = pd.DataFrame(results)
print("\nSummary of Cleaned Data Issues:")
print(results_df)

# Save results as CSV for further use
results_df.to_csv("cleaned_data_issues_summary.csv", index=False)
print(f"\nReport saved as {output_report} and cleaned_data_issues_summary.csv")
