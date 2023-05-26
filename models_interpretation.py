import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.inspection import plot_partial_dependence
import pre_ml

# Create the interpretation_plots directory if it doesn't exist
os.makedirs("interpretation_plots", exist_ok=True)

# Load the saved models
model_filenames = [
    'models/XGBoost_best_model.pkl',
    'models/SVM_best_model.pkl',
    'models/Random Forest_best_model.pkl',
    'models/Logistic Regression_best_model.pkl',
    'models/K-Nearest Neighbors_best_model.pkl',
    'models/Decision Tree_best_model.pkl'
]

models = [joblib.load(filename) for filename in model_filenames]

# Load the feature names
feature_names = list(pre_ml.scaled_data.columns)  # Use the column names of the scaled_data DataFrame

# Create partial dependence plots for each model
for i, model in enumerate(models):
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot partial dependence for each feature
    plot_partial_dependence(model, pre_ml.scaled_data, features=list(range(len(feature_names))), feature_names=feature_names,
                            n_jobs=-1, grid_resolution=100)

    # Set the title and labels
    ax.set_title(f"Partial Dependence Plot - {model_filenames[i]}")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Partial Dependence")

    # Save the plot
    plot_filename = f"interpretation_plots/{model_filenames[i].split('/')[1]}_partial_dependence_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    print(f"Partial dependence plot for {model_filenames[i]} saved as {plot_filename}")
