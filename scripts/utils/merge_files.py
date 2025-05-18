import pandas as pd
import os

# Source files directory
dir = 'data/raw/spot_prices'  

# Create a list to store the dataframes
dfs = []

# Iter by each file in the direct
for file in os.listdir(dir):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        # Read Excel file and add it to the list (dfs)
        df = pd.read_excel(os.path.join(dir, file))
        dfs.append(df)

# Concatenate all dataframes into one
df_merged = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe to a new Excel file 
df_merged.to_excel('../data/merged/merged_files.xlsx', index=False)