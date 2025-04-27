import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

# Define the path to the "building data" folder
downloads_path = 'C:/Users/jaden/Downloads'
building_data_path = os.path.join(downloads_path, 'building data')

# Verify that the building data folder exists
if not os.path.exists(building_data_path):
    raise FileNotFoundError(f"The folder '{building_data_path}' does not exist. Please check the path and folder name.")

# Define all utility datasets to analyze
utility_files = [
    ('cleaned_electricity-*.csv', 'electricity'),
    ('cleaned_gas.csv', 'gas'),
    ('cleaned_hotwater.csv', 'hotwater'),
    ('cleaned_irrigation.csv', 'irrigation'),
    ('cleaned_solar.csv', 'solar'),
    ('cleaned_steam.csv', 'steam'),
    ('cleaned_water.csv', 'water')
]

# Initialize the report content
report_lines = ["# Energy Usage Analysis Report\n"]
report_lines.append(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report_lines.append("This report analyzes energy usage trends and detects anomalies using the individual cleaned datasets.\n")

# Load metadata to map building_id to primaryspaceusage
metadata_path = os.path.join(building_data_path, 'cleaned_metadata.csv')
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"The file '{metadata_path}' does not exist.")
print("Loading metadata...")
metadata = pd.read_csv(metadata_path)[['building_id', 'primaryspaceusage', 'site_id']]
print("Metadata loaded.")

# Load weather data to correlate with usage
weather_path = os.path.join(building_data_path, 'cleaned_weather.csv')
if not os.path.exists(weather_path):
    raise FileNotFoundError(f"The file '{weather_path}' does not exist.")
print("Loading weather data...")
weather = pd.read_csv(weather_path, parse_dates=['timestamp'])
# Ensure weather timestamp is timezone-aware (UTC)
weather['timestamp'] = pd.to_datetime(weather['timestamp'], utc=True)
print("Weather data loaded.")

# Step 1: Aggregate energy usage per building across all utility types
print("Step 1: Aggregating energy usage per building...")
total_usage_per_building = pd.DataFrame()
for file_pattern, utility_type in utility_files:
    file_path = os.path.join(building_data_path, file_pattern)
    # Verify that the file(s) exist
    if '*' in file_pattern:
        electricity_files = glob.glob(file_path)
        if not electricity_files:
            report_lines.append(f"\n## Error\n")
            report_lines.append(f"- No electricity files found matching pattern: '{file_path}'.\n")
            continue
    else:
        if not os.path.exists(file_path):
            report_lines.append(f"\n## Error\n")
            report_lines.append(f"- The file '{file_path}' does not exist.\n")
            continue
    
    print(f"Processing {utility_type} dataset...")
    df = dd.read_csv(
        file_path,
        blocksize='128MB',
        parse_dates=['timestamp'],
        date_format='%Y-%m-%d %H:%M:%S%z'
    )

    # Aggregate usage per building
    print(f"Aggregating usage for {utility_type}...")
    usage_per_building = df.groupby('building_id')['value'].sum().compute().reset_index()
    usage_per_building = usage_per_building.rename(columns={'value': f'{utility_type}_usage'})
    print(f"Finished aggregating usage for {utility_type}.")
    
    # Merge with total usage
    if total_usage_per_building.empty:
        total_usage_per_building = usage_per_building
    else:
        total_usage_per_building = total_usage_per_building.merge(usage_per_building, on='building_id', how='outer')

# Fill NaN with 0 (if a building doesn't use a particular utility)
total_usage_per_building = total_usage_per_building.fillna(0)

# Calculate total usage across all utility types
utility_columns = [f'{utility_type}_usage' for _, utility_type in utility_files]
total_usage_per_building['total_usage'] = total_usage_per_building[utility_columns].sum(axis=1)

# Merge with metadata to get primaryspaceusage and site_id
total_usage_per_building = total_usage_per_building.merge(metadata, on='building_id', how='left')
print("Step 1 completed: Total usage aggregated.")

# Energy Usage Trends: Which types of buildings consume the most energy?
report_lines.append("\n## Energy Usage Trends\n")
report_lines.append("\n### Building Types with Highest Energy Consumption\n")
usage_by_building_type = total_usage_per_building.groupby('primaryspaceusage')['total_usage'].sum().sort_values(ascending=False)
report_lines.append("Total energy usage by building type (sum of all utility usage):\n")
for building_type, usage in usage_by_building_type.items():
    report_lines.append(f"- **{building_type}**: {usage:,.2f} units\n")

# Step 2: Analyze external factors (seasonality, time of day, temperature) across all utilities
report_lines.append("\n### Influence of External Factors\n")
for file_pattern, utility_type in utility_files:
    file_path = os.path.join(building_data_path, file_pattern)
    if '*' in file_pattern:
        electricity_files = glob.glob(file_path)
        if not electricity_files:
            continue
    else:
        if not os.path.exists(file_path):
            continue
    
    print(f"Analyzing external factors for {utility_type} data...")
    df = dd.read_csv(
        file_path,
        blocksize='128MB',
        parse_dates=['timestamp'],
        date_format='%Y-%m-%d %H:%M:%S%z'
    )
    print(f"Loaded {utility_type} dataset.")

    # Extract month and hour from timestamp
    print(f"Extracting month and hour for {utility_type}...")
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour

    # Aggregate usage by month and hour
    print(f"Aggregating usage by month for {utility_type}...")
    usage_by_month = df.groupby('month')['value'].mean().compute().sort_index()
    print(f"Aggregating usage by hour for {utility_type}...")
    usage_by_hour = df.groupby('hour')['value'].mean().compute().sort_index()

    # Merge with weather data to correlate with temperature
    # Round timestamps to the nearest hour for merging
    print(f"Rounding timestamps for {utility_type}...")
    df['timestamp_hour'] = df['timestamp'].dt.round('h')
    weather['timestamp_hour'] = weather['timestamp'].dt.round('h')
    print(f"Merging {utility_type} with metadata for site_id...")
    df = df.merge(
        total_usage_per_building[['building_id', 'site_id']],
        on='building_id',
        how='left'
    )
    print(f"Merging {utility_type} with weather data...")
    df_with_weather = df.merge(
        weather[['timestamp_hour', 'site_id', 'airTemperature']],
        left_on=['timestamp_hour', 'site_id'],
        right_on=['timestamp_hour', 'site_id'],
        how='left'
    )

    # Group by temperature bins and calculate average usage
    print(f"Binning temperatures for {utility_type}...")
    df_with_weather['temp_bin'] = df_with_weather['airTemperature'].map_partitions(
        lambda s: pd.cut(s, bins=range(-20, 41, 5), labels=range(-20, 36, 5)),
        meta=('airTemperature', 'category')
    )
    print(f"Aggregating usage by temperature for {utility_type}...")
    usage_by_temp = df_with_weather.groupby('temp_bin', observed=True)['value'].mean().compute()

    # Report results
    report_lines.append(f"\n#### {utility_type.capitalize()} Usage - Monthly Average\n")
    for month, usage in usage_by_month.items():
        report_lines.append(f"- **Month {month}**: {usage:.2f} units (average per reading)\n")

    report_lines.append(f"\n#### {utility_type.capitalize()} Usage - Hourly Average\n")
    for hour, usage in usage_by_hour.items():
        report_lines.append(f"- **Hour {hour}**: {usage:.2f} units (average per reading)\n")

    report_lines.append(f"\n#### {utility_type.capitalize()} Usage - By Temperature (°C)\n")
    for temp, usage in usage_by_temp.items():
        report_lines.append(f"- **{temp}°C to {temp+5}°C**: {usage:.2f} units (average per reading)\n")
    print(f"Finished analyzing external factors for {utility_type}.")

# Step 3: Anomaly Detection - Identify buildings with abnormal energy usage (Global)
print("Step 3: Detecting abnormal energy usage (global)...")
report_lines.append("\n## Anomaly Detection\n")
report_lines.append("\n### Buildings with Abnormal Energy Usage (Global)\n")
# Calculate z-scores for total usage per building
mean_usage = total_usage_per_building['total_usage'].mean()
std_usage = total_usage_per_building['total_usage'].std()
total_usage_per_building['z_score'] = (total_usage_per_building['total_usage'] - mean_usage) / std_usage

# Identify outliers (z-score > 2.5 or < -2.5)
anomalous_buildings = total_usage_per_building[total_usage_per_building['z_score'].abs() > 2.5]
anomalous_buildings = anomalous_buildings.sort_values('z_score', ascending=False)
if not anomalous_buildings.empty:
    report_lines.append("Buildings with abnormal energy usage (z-score > 2.5 or < -2.5):\n")
    for _, row in anomalous_buildings.iterrows():
        report_lines.append(f"- **Building ID {row['building_id']} ({row['primaryspaceusage']})**: Total Usage = {row['total_usage']:,.2f}, Z-Score = {row['z_score']:.2f}\n")
else:
    report_lines.append("No buildings with abnormal energy usage detected (z-score > 2.5 or < -2.5).\n")
print("Step 3 completed.")

# Step 4: Anomaly Detection - Identify buildings with abnormal energy usage (by Building Type)
print("Step 4: Detecting abnormal energy usage (by building type)...")
report_lines.append("\n### Buildings with Abnormal Energy Usage (by Building Type)\n")
for building_type in total_usage_per_building['primaryspaceusage'].unique():
    type_subset = total_usage_per_building[total_usage_per_building['primaryspaceusage'] == building_type]
    mean_usage = type_subset['total_usage'].mean()
    std_usage = type_subset['total_usage'].std()
    if std_usage > 0:  # Avoid division by zero
        type_subset = type_subset.copy()
        type_subset['z_score_type'] = (type_subset['total_usage'] - mean_usage) / std_usage
        anomalous_type = type_subset[type_subset['z_score_type'].abs() > 2.5]
        if not anomalous_type.empty:
            report_lines.append(f"\n**{building_type} Buildings**\n")
            for _, row in anomalous_type.sort_values('z_score_type', ascending=False).iterrows():
                report_lines.append(f"- **Building ID {row['building_id']}**: Total Usage = {row['total_usage']:,.2f}, Z-Score = {row['z_score_type']:.2f}\n")
print("Step 4 completed.")

# Step 5: Identify time periods with potential energy waste (across all utilities)
print("Step 5: Identifying time periods with potential energy waste...")
report_lines.append("\n### Time Periods with Potential Energy Waste\n")
for file_pattern, utility_type in utility_files:
    file_path = os.path.join(building_data_path, file_pattern)
    if '*' in file_pattern:
        electricity_files = glob.glob(file_path)
        if not electricity_files:
            continue
    else:
        if not os.path.exists(file_path):
            continue
    
    print(f"Analyzing time periods for potential waste in {utility_type} data...")
    df = dd.read_csv(
        file_path,
        blocksize='128MB',
        parse_dates=['timestamp'],
        date_format='%Y-%m-%d %H:%M:%S%z'
    )

    # Extract month and hour
    print(f"Extracting month and hour for {utility_type} (waste analysis)...")
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour

    # Group by month and hour
    print(f"Aggregating usage by month and hour for {utility_type} (waste analysis)...")
    usage_by_month_hour = df.groupby(['month', 'hour'])['value'].mean().compute().reset_index()
    # Calculate z-scores
    mean_usage_mh = usage_by_month_hour['value'].mean()
    std_usage_mh = usage_by_month_hour['value'].std()
    usage_by_month_hour['z_score'] = (usage_by_month_hour['value'] - mean_usage_mh) / std_usage_mh

    # Identify periods with high usage (z-score > 2.5)
    high_usage_periods = usage_by_month_hour[usage_by_month_hour['z_score'] > 2.5].sort_values('z_score', ascending=False)
    if not high_usage_periods.empty:
        report_lines.append(f"\n**{utility_type.capitalize()} Usage**\n")
        for _, row in high_usage_periods.iterrows():
            report_lines.append(f"- **Month {int(row['month'])}, Hour {int(row['hour'])}**: Average Usage = {row['value']:.2f}, Z-Score = {row['z_score']:.2f}\n")
    print(f"Finished analyzing time periods for {utility_type}.")

# Save the report as a Markdown file
print("Saving report...")
report_path = os.path.join(os.getcwd(), 'energy_analysis_report.md')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"Energy analysis report generated and saved as '{report_path}'.")
