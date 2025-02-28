
import os
os.chdir(r"D:\THESIS\Thesis_data")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from patchify import patchify
from sklearn.preprocessing import StandardScaler

from matplotlib import gridspec
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# File paths
large_image_path = 'P_stacked_glcm_with_rvi.tif'
large_CHM_path = 'Lope/D/Lope_CHM_rotated_clipped.tif'
dem_path = 'Lope/D/DEM_rotated_clipped.tif'

def read_and_pad_image(image_path, chm_path, dem_path, patch_size):
    image = tiff.imread(image_path)
    chm = tiff.imread(chm_path)
    dem = tiff.imread(dem_path)

    pad_h = (patch_size - image.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - image.shape[1] % patch_size) % patch_size

    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=np.nan)
    chm = np.pad(chm, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
    dem = np.pad(dem, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
    return image, chm, dem

def standardize_image(image):
    scaler = StandardScaler()
    reshaped_image = image.reshape(-1, image.shape[2])
    standardized_image = scaler.fit_transform(reshaped_image)
    return standardized_image.reshape(image.shape)

def create_patches(image, chm, dem, patch_size, step_size):
    image_patches = patchify(image, (patch_size, patch_size, image.shape[2]), step=step_size)
    chm_patches = patchify(chm, (patch_size, patch_size), step=step_size)
    dem_patches = patchify(dem, (patch_size, patch_size), step=step_size)
    return image_patches, chm_patches, dem_patches

def filter_valid_patches(image_patches, chm_patches, dem_patches):
    valid_image_patches, valid_chm_patches, valid_dem_patches = [], [], []
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            img_patch = image_patches[i, j, 0, :, :, :]
            chm_patch = chm_patches[i, j, :, :]
            dem_patch = dem_patches[i, j, :, :]
            # Ensure none of the patches contain NaN values
            if not np.isnan(img_patch).any() and not np.isnan(chm_patch).any() and not np.isnan(dem_patch).any():
                valid_image_patches.append(img_patch)
                valid_chm_patches.append(chm_patch)
                valid_dem_patches.append(dem_patch)
    return np.array(valid_image_patches), np.array(valid_chm_patches), np.array(valid_dem_patches)

def create_val_indices(total_patches, group_size=5, select_size=1):
    indices = []
    for i in range(0, total_patches, group_size):
        indices.extend(range(i, min(i + select_size, total_patches)))
    return np.array(indices)

def calculate_strata_bias(y_test, y_pred, strata_ranges):
    strata_bias = {}
    for strata in strata_ranges:
        mask = (y_test >= strata[0]) & (y_test < strata[1])
        if np.any(mask):
            bias_values = y_test[mask] - y_pred[mask]
            strata_bias[str(strata)] = {'bias': bias_values, 'pixels': np.sum(mask)}
    return strata_bias

def visualize_bias_whisker_plots(strata_bias):
    strata_labels = list(strata_bias.keys())
    bias_data = [strata_bias[strata]['bias'] for strata in strata_labels]
    pixel_counts = [strata_bias[strata]['pixels'] for strata in strata_labels]

    plt.figure(figsize=(10, 8))
    box = plt.boxplot(bias_data, labels=strata_labels, patch_artist=True, widths=0.3, showfliers=False)
    plt.xlabel('Height Strata [m]', fontsize=14)
    plt.ylabel('Bias [m]', fontsize=14)
    plt.grid(False)

    # Add a dashed line at 0 bias
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

    # Set box color to ash
    ash_color = '#878e95'
    for patch in box['boxes']:
        patch.set_facecolor(ash_color)

    # Add pixel counts as horizontal annotations beside the upper whisker
    for i, count in enumerate(pixel_counts, start=1):
        # Place the text beside the upper whisker for each stratum
        upper_whisker = box['whiskers'][2 * (i - 1) + 1].get_ydata()[1]
        plt.text(i, upper_whisker + 0.1, f'N={count}', horizontalalignment='center', verticalalignment='bottom', fontsize=12)
    plt.savefig('bias_whisker_plot_P_RF.png', dpi=600)
    plt.show()


def scatter_density_plot(x, y, ax, ax_histx, ax_histy):
    # No labels for marginal histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # The density scatter plot with hexbin
    hb = ax.hexbin(x, y, gridsize=50, cmap='viridis', bins='log', norm=LogNorm())

    # Add a color bar next to the right histogram
    cb = plt.colorbar(hb, ax=ax_histy, orientation='vertical', pad=0.1)
    cb.set_label('Number of Pixels \n(log scale)', fontsize=18)

    # Set the same limits for x in the main plot and the upper histogram

    lim = max(np.max(x), np.max(y))  # Adjust the range based on max values of x and y
    bins = np.logspace(np.log10(1), np.log10(lim), 50)
    
    # Match upper histogram scale to main plot x-axis
    ax_histx.hist(x, bins=bins, log=True, color='gray')
    ax_histx.set_xlim(ax.get_xlim())  # Ensure they share the same x-limits

    # Create the side histogram
    ax_histy.hist(y, bins=bins, log=True, orientation='horizontal', color='gray')
    
    # Enable grid on histograms
    ax_histx.grid(True, which='both', axis='x', color='whitesmoke', linestyle='-', linewidth=0.6)
    ax_histy.grid(True, which='both', axis='y', color='whitesmoke', linestyle='-', linewidth=0.6)

def visualize_results(y_test, y_pred, r2, rmse):
    
    # Flatten and filter out invalid values (if any)
    valid_indices = np.isfinite(y_test) & np.isfinite(y_pred)
    y_test_valid = y_test[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    
    
    # Define axis limits based on data range
    min_val = 0
    max_val = 58
    # Create figure with subplots
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    # Create the Axes
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)


    # Draw the scatter density plot and marginals
    scatter_density_plot(y_test_valid, y_pred_valid, ax_main, ax_histx, ax_histy)
    

    # Add 1:1 line (ideal fit line)
    ax_main.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2, label='Ideal Fit', alpha=0.8)

    # Add R-squared and RMSE to the plot
    ax_main.text(0.05, 0.95, f'R² = {r2:.2f}', ha='left', va='top', transform=ax_main.transAxes, fontsize=16, fontweight='bold')
    ax_main.text(0.05, 0.90, f'RMSE = {rmse:.2f} m', ha='left', va='top', transform=ax_main.transAxes, fontsize=16, fontweight='bold')

    # Set labels
    ax_main.set_xlabel('Reference Height [m]', fontsize=22)
    ax_main.set_ylabel('Estimated Height [m]', fontsize=22)

    # Set axis limits
    ax_main.set_xlim([0, 58])
    ax_main.set_ylim([0, 58])

    # Customize tick labels
    ax_main.tick_params(axis='both', which='major', labelsize=14, width=2, direction='in', length=6)
    # Enable grid on the main scatter plot
    ax_main.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.6)


    # Show the plot
    plt.tight_layout()
    plt.savefig('LGBM_P.png', dpi=600)
    plt.show()



# Main script
patch_size, step_size = 32, 32
large_image, large_chm, dem = read_and_pad_image(large_image_path, large_CHM_path, dem_path, patch_size)

nan_mask = np.isnan(large_image).any(axis=2)
large_chm[nan_mask] = np.nan
dem[nan_mask] = np.nan

image_patches, chm_patches, dem_patches = create_patches(large_image, large_chm, dem, patch_size, step_size)
valid_image_patches, valid_chm_patches, valid_dem_patches = filter_valid_patches(image_patches, chm_patches, dem_patches)

val_indices = create_val_indices(valid_image_patches.shape[0], group_size=7, select_size=2)
train_indices = np.setdiff1d(np.arange(valid_image_patches.shape[0]), val_indices)

X_train, X_test = valid_image_patches[train_indices], valid_image_patches[val_indices]
y_train, y_test = valid_chm_patches[train_indices], valid_chm_patches[val_indices]
X_train.shape


# Split train data into 70% for training and 20% for validation (hyperparameter tuning)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# Reshape to flatten (samples, height, width) into a single dimension, keeping channels intact
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  # Shape: (46 * 32 * 32, 13)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])     # Shape: (28 * 32 * 32, 13)
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])        # Shape: (21 * 32 * 32, 13)

# Print to verify shapes
print("X_train reshaped:", X_train_reshaped.shape)
print("X_test reshaped:", X_test_reshaped.shape)
print("X_val reshaped:", X_val_reshaped.shape)   


# reshape label dataset
y_train_reshaped = y_train.flatten()
y_test_reshaped  = y_test.flatten()
y_val_reshaped = y_val.flatten()

# Print to verify shapes
print("Y_train reshaped:", y_train_reshaped.shape)
print("Y_test reshaped:", y_test_reshaped.shape)
print("Y_val reshaped:", y_val_reshaped.shape)   



# Define column names for each channel
channel_names = [f'channel_{i+1}' for i in range(X_train_reshaped.shape[1])]

# Create DataFrames for each dataset
X_train_df = pd.DataFrame(X_train_reshaped, columns=channel_names)
X_val_df = pd.DataFrame(X_val_reshaped, columns=channel_names)
X_test_df = pd.DataFrame(X_test_reshaped, columns=channel_names)

# Create DataFrames for labels (as a single column named 'target')
y_train_df = pd.DataFrame(y_train_reshaped, columns=['target'])
y_val_df = pd.DataFrame(y_val_reshaped, columns=['target'])

y_test_df = pd.DataFrame(y_test_reshaped, columns=['target'])

# Display the first few rows of X_train and y_train to verify
print(X_train_df.head())
print(y_train_df.head())
y_val_df.shape



# Step 2: Define the LGBM model and parameter grid for hyperparameter tuning
lgbm = LGBMRegressor(random_state=42)

param_grid = {
    'n_estimators': [100],       # Number of boosting rounds
    'max_depth': [10],            # Maximum depth of trees
    'learning_rate': [0.05],   # Learning rate
    'num_leaves': [50],           # Number of leaves in one tree
    'min_child_samples': [100]    # Minimum number of data needed in a child
}
# Step 3: Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='r2')

# Step 4: Fit GridSearchCV on the validation data
grid_search.fit(X_val_df, y_val_df)

# Get the best model and best hyperparameters
best_lgbm_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")


# Step 5: Train the best model on the entire training set
best_lgbm_model.fit(X_train_df, y_train_df.values.ravel())  # Use .ravel() if y is a single column

# Step 6: Predict on the validation set using the best model
y_train_pred = best_lgbm_model.predict(X_train_df)

# Evaluate the performance on the validation set
train_rmse = mean_squared_error(y_train_df, y_train_pred, squared=False)
train_r2 = r2_score(y_train_df, y_train_pred)
print(f"Validation RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

# Step 7: Test the model on the test set
y_test_pred = best_lgbm_model.predict(X_test_df)

# Calculate standard metrics for the test set
test_rmse = mean_squared_error(y_test_df, y_test_pred, squared=False)
test_r2 = r2_score(y_test_df, y_test_pred)
test_mae = mean_absolute_error(y_test_df, y_test_pred)
test_bias = (y_test_df.values.ravel() - y_test_pred).mean() # Calculating bias as the mean difference
test_bias
# Print the results
print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}, Bias: {test_bias:.4f}")


# Convert y_test_df to a 1D array before passing to visualize_results
visualize_results(y_test_df.values.ravel(), y_test_pred, test_r2, test_rmse)

# Define strata ranges
strata_ranges = [(0, 20), (20, 30), (30, 40), (40,60)] 

# Calculate bias for each stratum

strata_bias = calculate_strata_bias(y_test.flatten(), y_test_pred.flatten(), strata_ranges)
print(f"Strata-wise Bias: \n{strata_bias}")


# for strata, bias_info in strata_bias.items():
#     bias = bias_info['bias']
#     pixels = bias_info['pixels']
#     print(f"Strata {strata}: Bias Mean = {np.mean(bias):.2f}, Bias Std = {np.std(bias):.2f}, Pixels = {pixels}")

