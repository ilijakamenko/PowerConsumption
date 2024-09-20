#---------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------------------


num_epochs = 5
# Load data from CSV
df = pd.read_csv('transformer_power_measurements_cuda.csv')

# Calculate cumulative elapsed time
df['Cumulative Elapsed Time (s)'] = 0
cumulative_time = 0

for epoch in range(num_epochs):
    df_epoch = df[df['Epoch'] == epoch]
    if not df_epoch.empty:
        df.loc[df['Epoch'] == epoch, 'Cumulative Elapsed Time (s)'] = df_epoch['Elapsed Time (s)'] + cumulative_time
        cumulative_time += df_epoch['Elapsed Time (s)'].iloc[-1]

plt.figure(figsize=(12, 8))

# Plot the entire series of power consumption over time with cumulative time
plt.plot(df['Cumulative Elapsed Time (s)'], df['Power Consumption (W)'], label='Power Consumption')

# Add vertical lines and labels at the end of each epoch
cumulative_time = 0
for epoch in range(num_epochs):
    df_epoch = df[df['Epoch'] == epoch]
    if not df_epoch.empty:
        cumulative_time += df_epoch['Elapsed Time (s)'].iloc[-1]
        plt.axvline(x=cumulative_time, color='r', linestyle='--', label=f'End of Epochs {epoch+1}' if epoch == 0 else "")
        plt.text(cumulative_time, df['Power Consumption (W)'].min(), f'Epoch {epoch+1}', rotation=90, verticalalignment='bottom', horizontalalignment='right', color='r')

plt.xlabel('Elapsed Time (s)')
plt.ylabel('Power Consumption (W)')
plt.title('GPU Power Consumption Over Time')

plt.grid(True)
plt.savefig('transformer_power_measurements_cuda.png')  # Save plot as PNG
plt.show()
