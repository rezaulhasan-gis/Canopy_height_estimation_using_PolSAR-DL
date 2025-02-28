import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

# Change the working directory
os.chdir('D:/THESIS')

# Load the data from the CSV file
data = pd.read_csv('PUBLICATION_OUTPUT/strata_bias_L_band_all_models.csv')

# Convert the RESIDUALS column to numeric, handling any non-numeric values
data['RESIDUALS'] = pd.to_numeric(data['RESIDUALS'], errors='coerce')

# Drop rows with NaN in the RESIDUALS column
data = data.dropna(subset=['RESIDUALS'])

# Create height strata based on 10-meter intervals in the TRUE_CHM column
data['TRUE_CHM_STRATA'] = pd.cut(data['TRUE_CHM'], 
                                 bins=range(0, int(data['TRUE_CHM'].max()) + 10, 10), 
                                 right=False)

# Statistical summary for each stratum
summary_stats = data.groupby('TRUE_CHM_STRATA', observed=False)['RESIDUALS'].describe()
print("Summary statistics for residuals by strata:")
print(summary_stats)

# Statistical test: Kruskal-Wallis Test across all strata
kruskal_results = kruskal(*[data.loc[data['TRUE_CHM_STRATA'] == strata, 'RESIDUALS'] 
                            for strata in data['TRUE_CHM_STRATA'].unique()])
print(f"\nKruskal-Wallis test results: H-statistic = {kruskal_results.statistic}, p-value = {kruskal_results.pvalue}")


# Debug TRUE_CHM_STRATA unique values
print("Unique values in TRUE_CHM_STRATA:")
print(data['TRUE_CHM_STRATA'].unique())

# Filter data for extreme strata (0-10 and 50-60)
extreme_strata = data[data['TRUE_CHM_STRATA'].isin(['[0, 10)', '[50, 60)'])]

# Ensure TRUE_CHM_STRATA is categorical or string for hue mapping
extreme_strata['TRUE_CHM_STRATA'] = extreme_strata['TRUE_CHM_STRATA'].astype(str)

# Verify filtered data
print("\nExtreme strata captured:")
print(extreme_strata['TRUE_CHM_STRATA'].unique())

# KDE plot for extreme strata
plt.figure(figsize=(10, 6))

# Check if the hue column has valid values
if extreme_strata['TRUE_CHM_STRATA'].notnull().all() and len(extreme_strata['TRUE_CHM_STRATA'].unique()) > 1:
    sns.kdeplot(data=extreme_strata, x='RESIDUALS', hue='TRUE_CHM_STRATA', fill=True)
    plt.title('Density Plot of Residuals for Extreme Strata', fontsize=16)
    plt.xlabel('Residuals [m]', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Strata', fontsize=12)
    plt.show()
else:
    print("The TRUE_CHM_STRATA column contains null or insufficient values. Cannot create KDE plot.")

# Boxplot for all strata with annotations
plt.figure(figsize=(12, 8))
sns.boxplot(x='TRUE_CHM_STRATA', y='RESIDUALS', hue='MODEL', data=data, showfliers=False)
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.xlabel('Forest Heights [m]', fontsize=14)
plt.ylabel('Residuals [m]', fontsize=14)
plt.title('Bias Analysis by Canopy Height Strata', fontsize=16)
plt.legend(loc='upper left', fontsize=12)

# Annotate boxplot with the number of samples
strata_counts = data.groupby('TRUE_CHM_STRATA').size()
for i, strata in enumerate(strata_counts.index):
    count = int(strata_counts[strata] / 4)  # Adjust as needed
    y_max = data[data['TRUE_CHM_STRATA'] == strata]['RESIDUALS'].max()
    margin = 0.6
    text_position = y_max + margin if strata.left > 20 else y_max * margin
    plt.text(i, text_position, f'n={count}', ha='center', va='bottom', fontsize=12, color='black')

plt.show()

# Scatter plot of Residuals vs. True Canopy Height
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='TRUE_CHM', y='RESIDUALS', hue='MODEL', alpha=0.6)
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.title('Residuals vs. True Canopy Height', fontsize=16)
plt.xlabel('True Canopy Height [m]', fontsize=14)
plt.ylabel('Residuals [m]', fontsize=14)
plt.legend(title='Model', fontsize=12)
plt.show()





##Another


# Group data by MODEL and TRUE_CHM_STRATA, then calculate summary statistics for RESIDUALS
model_strata_stats = data.groupby(['MODEL', 'TRUE_CHM_STRATA'])['RESIDUALS'].describe()

# Display the statistics
print("Summary statistics for residuals by model and strata:")
print(model_strata_stats)

# Save the summary statistics to a CSV for detailed review (optional)
output_path = 'D:/THESIS/PUBLICATION_OUTPUT/model_strata_stats.csv'
model_strata_stats.to_csv(output_path)
print(f"\nDetailed statistics saved to: {output_path}")

# Visualize residual distributions for all models across strata
plt.figure(figsize=(14, 8))
sns.boxplot(x='TRUE_CHM_STRATA', y='RESIDUALS', hue='MODEL', data=data, showfliers=False)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Forest Heights [m]', fontsize=14)
plt.ylabel('Residuals [m]', fontsize=14)
plt.title('Residuals by Model and Canopy Height Strata', fontsize=16)
plt.legend(title='Model', fontsize=12, loc='upper left')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Scatter plot of Residuals vs. True Canopy Height, separated by model
plt.figure(figsize=(14, 8))
sns.scatterplot(data=data, x='TRUE_CHM', y='RESIDUALS', hue='MODEL', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('True Canopy Height [m]', fontsize=14)
plt.ylabel('Residuals [m]', fontsize=14)
plt.title('Residuals vs. True Canopy Height by Model', fontsize=16)
plt.legend(title='Model', fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()

# Define the output path for the Excel file
output_excel_path = 'D:/THESIS/PUBLICATION_OUTPUT/model_strata_stats_L.xlsx'

# Save the summary statistics to an Excel file
model_strata_stats.to_excel(output_excel_path)
print(f"\nDetailed statistics saved to: {output_excel_path}")