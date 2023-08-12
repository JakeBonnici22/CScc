import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import pre_ml1
import os
import textwrap

# Specify the directory containing the pkl models
directory = "ensemble/vc"

auc_scores = []
if not os.path.exists("ensemble/rocprfigures"):
    os.makedirs("ensemble/rocprfigures")


color1 = "#5975a4"
color2 = "#cc8963"
line_width = 2


for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        # Load the pickle file
        file_path = os.path.join(directory, filename)
        model = joblib.load(file_path)

        y_pred = model.predict_proba(pre_ml1.X_test)[:, 1]

        # Calculate precision and recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(pre_ml1.y_test, y_pred)
        plt.figure(figsize=(10, 6))
        # Plot precision and recall at different thresholds
        plt.plot(thresholds, precision[:-1], label="Precision", color=color1, linewidth=line_width)
        plt.plot(thresholds, recall[:-1], label="Recall", color=color2, linewidth=line_width)
        plt.xlabel("Threshold", fontsize=18)
        plt.ylabel("Score", fontsize=18)
        wrapped_title = textwrap.fill(filename.replace("_", " ").replace(".pkl", "").replace("best", ""), 80)
        plt.title("Precision and Recall at Different Thresholds for\n " + wrapped_title, loc='center', fontsize=20, pad=0,
                  fontweight='bold')
        print(filename)
        plt.legend(fontsize=16)
        fig_path = os.path.join("ensemble/rocprfigures", "pr" + filename.replace("_", "").replace(".pkl", "") + ".png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
        fpr, tpr, thresholds = roc_curve(pre_ml1.y_test, y_pred)
        # Calculate the AUC-ROC score
        auc_roc = roc_auc_score(pre_ml1.y_test, y_pred)
        auc_scores.append(auc_roc)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color=color1, linewidth=line_width)
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        wrapped_title = textwrap.fill(filename.replace("_", " ").replace(".pkl", "").replace("best", ""), 80)
        plt.title('Receiver Operating Characteristic (ROC) Curve for\n' + wrapped_title, loc='center', fontsize=20, pad=0,
                  fontweight='bold')
        fig_path = os.path.join("ensemble/rocprfigures", "roc" + filename.replace("_", "").replace(".pkl", "") + ".png")
        plt.savefig(fig_path, dpi=300)
        plt.close()


# Print the AUC-ROC scores for each model
for i, filename in enumerate(os.listdir(directory)):
    if filename.endswith(".pkl"):
        print("AUC-ROC Score for", filename, ":", auc_scores[i])


