import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change the working directory
os.chdir('D:/THESIS')

# Load the data from the CSV file
data = pd.read_csv('PUBLICATION_OUTPUT/strata_bias_L_band_all_models.csv')

# Convert the RESIDUALS column to numeric, handling any non-numeric values
data['RESIDUALS'] = pd.to_numeric(data['RESIDUALS'], errors='coerce')

# Drop rows with NaN in the RESIDUALS column, if any conversion issues were encountered
data = data.dropna(subset=['RESIDUALS'])

# Create height strata based on 10-meter intervals in the TRUE_CHM column
data['TRUE_CHM_STRATA'] = pd.cut(data['TRUE_CHM'], bins=range(0, int(data['TRUE_CHM'].max()) + 10, 10), right=False)

# Calculate the number of samples in each strata
strata_counts = data.groupby('TRUE_CHM_STRATA').size()

# Set up the figure
plt.figure(figsize=(12, 8))

# Add a thick horizontal line at 0 residuals
plt.axhline(0, color='black', linewidth=2.5)

# Define custom grayscale color palette
custom_palette = ['#2E86C1', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#16A085']
# Create the boxplot using TRUE_CHM_STRATA as x-axis and RESIDUALS as y-axis, with MODEL as the hue
# Replace your existing boxplot code with:
sns.boxplot(
    x='TRUE_CHM_STRATA',
    y='RESIDUALS',
    hue='MODEL',
    data=data,
    palette=custom_palette,
    showfliers=True,  # Keep outliers visible but smaller
    whis=[0, 100],    # Whiskers span min to max (0th to 100th percentile)
    flierprops={
        'marker': 'o',      # Smaller outlier markers
        'markerfacecolor': '0.3',
        'markersize': 4,     # Reduce size (default is 6)
        'alpha': 0.6         # Make them semi-transparent
    },
    linewidth=1.5       # Thicker box/whisker lines for clarity
)
# Customize the plot labels
plt.xlabel('Forest Heights [m]', fontsize=18)
plt.ylabel('Residuals [m]\n Overestimation        Underestimation ', fontsize=18)

# Set legend position
plt.legend(loc='upper left', fontsize=14)

# Set y-axis limits
plt.ylim(-50, 50)

# Set custom x-axis labels
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']  # Custom labels
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=0, fontsize=16)
plt.yticks(fontsize=16)

# Annotate the plot with the number of samples in each strata
for i, strata in enumerate(strata_counts.index):
    count = int(strata_counts[strata] / 4)  # Adjusting count to reflect each model
    y_max = data[data['TRUE_CHM_STRATA'] == strata]['RESIDUALS'].max()
    margin = 1.0  # Adjust this value for spacing
    text_position = y_max * margin if strata.left > 20 else y_max + margin

    # Annotate with adjusted position
    plt.text(i, text_position, f'n={count}', ha='center', va='bottom', fontsize=14, color='black')

# Add additional labels for Underestimation and Overestimation
# plt.text(-0.4, 35, 'Underestimation', fontsize=14, color='black', ha='right')
# plt.text(-0.4, -35, 'Overestimation', fontsize=14, color='black', ha='right')

# Save the plot with high resolution
plt.savefig('L_band_residual_plot_all_models.png', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()
