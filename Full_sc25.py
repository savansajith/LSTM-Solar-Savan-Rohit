import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv('/home/ProjB2125/Desktop/Project/PLOTSandCSV_LSTM/SC25_half_pred.csv')
df2 = pd.read_csv('/home/ProjB2125/Desktop/Project/PLOTSandCSV_LSTM/forecast_csv_run_final.csv')

# Parse date columns
df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m')

# Choose a common color
common_color = '#1f77b4'

# Plotting
plt.figure(figsize=(12, 6))

# Plot as dots (scatter-like), using the same color
plt.plot(df1['Date'], df1['mean'], '-o', markersize=4, color=common_color)
plt.plot(df2['Date'], df2['Forecasted Sunspot Number'], '-o', markersize=4, color=common_color)

# Combine into a single series for line plot
combined_dates = pd.concat([df1['Date'], df2['Date']])
combined_values = pd.concat([df1['mean'], df2['Forecasted Sunspot Number']])

# Sort by date to ensure proper line connection
combined = pd.DataFrame({'Date': combined_dates, 'Value': combined_values})
combined = combined.sort_values('Date')

# Plot the joined line with the same color
plt.plot(combined['Date'], combined['Value'], linewidth=1, color=common_color)

# Labels and formatting
plt.xlabel('Date')
plt.ylabel('Sunspot Number (SSN)')
plt.grid(True)
plt.tight_layout()
plt.show()

