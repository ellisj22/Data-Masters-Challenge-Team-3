import os
import dask.dataframe as dd
import pandas as pd
import glob
import shutil

# Step 1: Set the file path to the "building data" folder in Downloads for Windows
username = os.environ.get('USERNAME')
base_path = os.path.join('C:\\', 'Users', username, 'Downloads', 'building data')

# Verify that the folder exists and is writable
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The directory {base_path} does not exist. Please check the path and folder name.")
try:
    # Test write permissions by creating a temporary file
    test_file = os.path.join(base_path, 'test_write_permissions.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
except PermissionError:
    raise PermissionError(f"Cannot write to directory {base_path}. Please check permissions or close any programs using files in this directory.")
except Exception as e:
    raise Exception(f"Error accessing directory {base_path}: {e}")

# Step 2: Define a function to clean utility datasets (electricity, gas, hotwater, irrigation, solar, steam, water)
def clean_utility_dataset(file_name):
    try:
        # Load the dataset with Dask
        df = dd.read_csv(os.path.join(base_path, file_name), blocksize='64MB')
        
        # Standardize timestamp
        df['timestamp'] = dd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
        
        # Melt the DataFrame to long format
        df_long = df.melt(id_vars=['timestamp'], var_name='building_id', value_name='value')
        
        # Sort by building_id and timestamp for proper interpolation
        df_long = df_long.sort_values(['building_id', 'timestamp'])
        
        # Handle anomalies: Replace negative values with 0
        df_long['value'] = df_long['value'].clip(lower=0)
        
        # Handle outliers: Cap values at 5 standard deviations from the mean within each building_id
        stats = df_long.groupby('building_id')['value'].agg(['mean', 'std']).compute()
        stats = stats.reset_index()
        
        df_long = df_long.merge(stats, on='building_id', how='left')
        df_long['upper_threshold'] = df_long['mean'] + 5 * df_long['std']
        df_long['value'] = df_long[['value', 'upper_threshold']].min(axis=1)
        df_long = df_long.drop(columns=['mean', 'std', 'upper_threshold'])
        
        # Define a function to interpolate within each group
        def interpolate_group(group):
            group = group.sort_values('timestamp')
            group['value'] = group['value'].interpolate(method='linear', limit_direction='both')
            group['value'] = group['value'].ffill().bfill()
            return group
        
        # Apply interpolation to each building_id group
        meta = {'timestamp': 'datetime64[ns, UTC]', 'building_id': 'object', 'value': 'float64'}
        df_long = df_long.groupby('building_id').apply(interpolate_group, meta=meta)
        
        # Reset the index to remove any multi-index or duplicates
        df_long = df_long.reset_index(drop=True)
        
        # Convert building_id to string
        df_long['building_id'] = df_long['building_id'].astype('string')
        
        # Prepare output path
        output_file = os.path.join(base_path, f'cleaned_{file_name.split(".")[0]}')
        
        # Check for existing files and delete them to avoid conflicts
        existing_files = glob.glob(output_file + '-*.csv')
        if existing_files:
            print(f"Found existing files for {output_file}-*.csv. Deleting them to avoid conflicts.")
            for f in existing_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not delete {f}: {e}")
        
        # Save to multiple CSV files with error handling
        try:
            df_long.to_csv(output_file + '-*.csv', index=False)
            print(f"Cleaned {file_name} and saved to {output_file}-*.csv")
        except PermissionError as e:
            raise PermissionError(f"Failed to write to {output_file}-*.csv: {e}. Ensure no files are open in other programs and you have write permissions.")
        except Exception as e:
            raise Exception(f"Error writing to {output_file}-*.csv: {e}")
        
        return df_long
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        raise

# Step 3: Clean each utility dataset
utility_files = ['electricity.csv', 'gas.csv', 'hotwater.csv', 'irrigation.csv', 'solar.csv', 'steam.csv', 'water.csv']
for file in utility_files:
    clean_utility_dataset(file)

# Step 4: Clean weather dataset
weather = pd.read_csv(os.path.join(base_path, 'weather.csv'))

# Standardize timestamp
weather['timestamp'] = pd.to_datetime(weather['timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)

# Impute missing values in numeric columns with mean
numeric_weather_cols = [
    'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR',
    'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed'
]
for col in numeric_weather_cols:
    weather[col] = weather[col].fillna(weather[col].mean())

# Convert site_id to string
weather['site_id'] = weather['site_id'].astype('string')

# Save the cleaned weather dataset with error handling
try:
    weather.to_csv(os.path.join(base_path, 'cleaned_weather.csv'), index=False)
    print("Cleaned weather.csv and saved to cleaned_weather.csv")
except PermissionError as e:
    raise PermissionError(f"Failed to write cleaned_weather.csv: {e}. Ensure the file is not open in another program and you have write permissions.")
except Exception as e:
    raise Exception(f"Error writing cleaned_weather.csv: {e}")

# Step 5: Clean metadata dataset
metadata = pd.read_csv(os.path.join(base_path, 'metadata.csv'))

# Convert numeric columns to numeric, coercing non-numeric values to NaN
numeric_metadata_cols = ['sqm', 'sqft', 'lat', 'lng', 'yearbuilt', 'eui']
for col in numeric_metadata_cols:
    metadata[col] = pd.to_numeric(metadata[col], errors='coerce')

# Impute missing values
metadata_clean = metadata.copy()

# Categorical columns: fill with "Unknown"
categorical_metadata_cols = [
    'primaryspaceusage', 'sub_primaryspaceusage', 'timezone', 'electricity',
    'gas', 'water', 'irrigation', 'leed_level'
]
for col in categorical_metadata_cols:
    metadata_clean[col] = metadata_clean[col].replace('', 'Unknown').fillna('Unknown')

# Numeric columns: fill with median
for col in numeric_metadata_cols:
    metadata_clean[col] = metadata_clean[col].fillna(metadata_clean[col].median())

# Drop unnecessary columns
columns_to_drop = [
    'hotwater', 'chilledwater', 'steam', 'solar', 'industry', 'subindustry',
    'heatingtype', 'date_opened', 'numberoffloors', 'occupants',
    'energystarscore', 'site_eui', 'source_eui', 'rating'
]
metadata_clean = metadata_clean.drop(columns=columns_to_drop)

# Convert building_id and site_id to string dtype
metadata_clean['building_id'] = metadata_clean['building_id'].astype('string')
metadata_clean['site_id'] = metadata_clean['site_id'].astype('string')

# Save the cleaned metadata dataset with error handling
try:
    metadata_clean.to_csv(os.path.join(base_path, 'cleaned_metadata.csv'), index=False)
    print("Cleaned metadata.csv and saved to cleaned_metadata.csv")
except PermissionError as e:
    raise PermissionError(f"Failed to write cleaned_metadata.csv: {e}. Ensure the file is not open in another program and you have write permissions.")
except Exception as e:
    raise Exception(f"Error writing cleaned_metadata.csv: {e}")

# Step 6: Summary of cleaned datasets
print("\nCleaning completed for all datasets.")
print("Cleaned files are saved in the 'building data' folder with 'cleaned_' prefix.")
