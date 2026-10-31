import pandas as pd

# Load datasets
modeling_path = 'modeling/aligned_modeling_data.csv'
meteostat_path = 'meteorology/meteostat_data_2023_2025.csv'

# Read modeling data
modeling = pd.read_csv(modeling_path, dtype={'site_number': str})
# Remove leading zeros for matching
modeling['site_number'] = modeling['site_number'].str.lstrip('0')

# Read meteostat data
meteostat = pd.read_csv(meteostat_path, dtype={'site_number': str})
meteostat['site_number'] = meteostat['site_number'].str.lstrip('0')
meteostat = meteostat.rename(columns={'datetime': 'date'})

# Ensure date columns are the same format
def fix_date(df, col):
    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
fix_date(modeling, 'date')
fix_date(meteostat, 'date')

# Merge, prioritizing modeling data, filling missing met columns from meteostat
to_fill = ['prcp', 'tmax', 'tmin', 'srad', 'vp', 'swe']
merged = pd.merge(modeling, meteostat[['site_number', 'date'] + to_fill],
                  on=['site_number', 'date'], how='left', suffixes=('', '_meteo'))
for col in to_fill:
    merged[col] = merged[col].combine_first(merged[f'{col}_meteo'])
    merged.drop(columns=[f'{col}_meteo'], inplace=True)

# Save back to CSV
merged.to_csv(modeling_path, index=False)
print('Merge complete. File updated.')
