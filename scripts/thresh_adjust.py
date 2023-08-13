import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pre_ml1
import libs
import shap


models_dir = 'modelswpipe/'

# Threshholds to try.
thresholds = np.linspace(0.1, 0.9, 9)

# Iterate over the files in the models directory
for filename in os.listdir(models_dir):
    if not filename.endswith('.pkl'):
        continue

    model_path = os.path.join(models_dir, filename)
    model = joblib.load(model_path)
    model_name = filename.split('_')[0]
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Iterate over the thresholds
    for threshold in thresholds:
        if hasattr(model, 'predict_proba'):
            # Make predictions using the current threshold
            y_pred_proba = model.predict_proba(pre_ml1.X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(pre_ml1.X_test)
            y_pred = (decision_scores >= threshold).astype(int)
        else:
            print(f"Model '{model_name}' does not support threshold analysis.")
            break

        # Evaluation metrics
        precision = libs.precision_score(pre_ml1.y_test, y_pred)
        recall = libs.recall_score(pre_ml1.y_test, y_pred)
        f1 = libs.f1_score(pre_ml1.y_test, y_pred)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Plot the precision, recall, and F1-score curves
    plt.figure()
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'{model_name} - Threshold Analysis')
    plt.legend()
    plt.show()

    # Explain model predictions with SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(pre_ml1.X_test)
    plt.figure()
    shap.summary_plot(shap_values, pre_ml1.X_test, show=False)
    plt.title(f'{model_name} - SHAP Summary Plot')
    plt.show()

