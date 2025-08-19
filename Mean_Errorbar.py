import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to CSV folders
datapath1 = Path('/home/ProjB2125/Desktop/Project/LSTM100/csvs')
#datapath2 = Path('/home/ProjB2125/Desktop/Project/LSTM20/csvs')
#datapath3 = Path('/home/ProjB2125/Desktop/Project/LSTM6/csvs')


# Collect all CSV files
csv_files = list(datapath1.glob('*.csv'))
#csv_files = list(datapath1.glob('*.csv')) + list(datapath2.glob('*.csv')) + list(datapath3.glob('*.csv'))

dfs = []

# Read and rename forecast column uniquely
for i, file in enumerate(csv_files, start=1):
    df = pd.read_csv(file)
    df['Forecasted Sunspot Number'] = pd.to_numeric(df['Forecasted Sunspot Number'], errors='coerce')
    df = df.rename(columns={'Forecasted Sunspot Number': f'sn{i}'})
    dfs.append(df[['Date', f'sn{i}']])  # only keep Date and renamed forecast

# Merge all dataframes on 'Date'
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='Date', how='outer')

# Prepare the final DataFrame
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%Y-%m')
merged_df = merged_df.sort_values(by='Date')
sn_columns = [col for col in merged_df.columns if col.startswith('sn')]

# Compute mean and std
merged_df['mean'] = merged_df[sn_columns].mean(axis=1)
merged_df['std'] = merged_df[sn_columns].std(axis=1)

# Plotting only mean ± std
fig, ax = plt.subplots(figsize=(15, 8))
ax.errorbar(
    merged_df['Date'], merged_df['mean'], yerr=merged_df['std'],
    fmt='-o', ecolor='gray', capsize=3, label='Mean ± 1σ', color='black'
)

# Labels and layout
ax.set_title('Forecasted Sunspot Numbers (Mean ± Std Dev)')
ax.set_xlabel('Date')
ax.set_ylabel('Forecasted Sunspot Number')
ax.legend()
plt.tight_layout()
plt.show()

