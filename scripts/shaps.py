import shap
import pre_ml1
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline

# Load the model along with the pipeline
loaded_pipeline = joblib.load('ensemble/vc/Random Forest_best_model.pkl')
loaded_model = loaded_pipeline.steps[-1][1]  # Extract the model from the pipeline

# Assuming you have your input data in a variable named 'X_test' from pre_ml1 module
X_test_np = np.array(pre_ml1.X_test)

sample_size = 30
reference_sample = X_test_np[:sample_size]

if not os.path.exists("ensemble/shap_figures"):
    os.makedirs("ensemble/shap_figures")

# Initialize the SHAP explainer using TreeExplainer for Random Forest models
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X_test_np)

name_mapping = {
    'Sex': 'Sex',
    'Grade (0=Well/moderately differentiated, 1=poorly differentiated)': 'Grade',
    'Lymphovascular (0= Absent, 1= Present)': 'Lymphovascular',
    'Perineural  (0= Absent, 1= Present)': 'Perineural',
    'Immunosuppression (0= not immunosupressed, 1= immunosuppressed)': 'Immunosuppression',
    'Age Group (16-78, 78-87, 87-112)': 'Age Group',
    'Tumour Size Category (1=Small, 2=Medium, 3=Large)': 'Tumour Size',
    'Tumour Depth Category (1=Superficial, 2=Deep)': 'Tumour Depth',
    'Excision Margin Category (1=<1mm, 2=1-5mm, 3=>5mm)': 'Excision Margin'
}

shortened_feature_names = [name_mapping.get(name, name) for name in pre_ml1.X_test.columns]
print(shortened_feature_names)

# Generate the SHAP summary plot with the title
plt.figure()
shap.summary_plot(shap_values, X_test_np, feature_names=shortened_feature_names, plot_type="bar",
                  show=False)

# Set the title for the plot
plt.title("SHAP Summary Plot for\n Random Forest model", fontsize=20, loc='center', pad=0, fontweight='bold')

# Set the font size for axis titles
plt.xlabel("Feature Importance", fontsize=18)
plt.ylabel("Features", fontsize=18)
plt.tight_layout()
# Save the figureg
plt.savefig("ensemble/shap_figures/shap_rf_plot.png", dpi=300)
plt.show()
