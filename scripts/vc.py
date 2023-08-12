import os
from sklearn.ensemble import VotingClassifier
from joblib import load, dump
from sklearn.metrics import accuracy_score
import pre_ml1
import libs

# Specify the folder path where the pkl files are located
folder_path = "ensemble/vc"

# Get a list of file names in the folder
file_names = os.listdir(folder_path)

# Load the pkl files and store the classifiers in a dictionary
classifiers = {}
for file_name in file_names:
    if file_name.endswith(".pkl"):
        file_path = os.path.join(folder_path, file_name)
        classifier = load(file_path)
        classifiers[file_name[:-4]] = classifier

# Create the voting classifier
voting_classifier = VotingClassifier(estimators=list(classifiers.items()), voting='soft', verbose=3)

# Example usage of the voting classifier
# X_train and y_train are your training data
voting_classifier.fit(pre_ml1.X_train, pre_ml1.y_train)

# X_test is your test data
y_pred = voting_classifier.predict(pre_ml1.X_test)


accuracy = accuracy_score(pre_ml1.y_test, y_pred)
precision = libs.precision_score(pre_ml1.y_test, y_pred)
recall = libs.recall_score(pre_ml1.y_test, y_pred)
f1 = libs.f1_score(pre_ml1.y_test, y_pred)
auc_roc = libs.roc_auc_score(pre_ml1.y_test, y_pred)
auc_pr = libs.average_precision_score(pre_ml1.y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("AUC-ROC:", auc_roc)
print("AUC-PR:", auc_pr)

# Save the voting classifier
voting_classifier_file = os.path.join(folder_path, "voting_classifier.pkl")
dump(voting_classifier, voting_classifier_file)

# Save the metrics to a text file
metrics_file = os.path.join(folder_path, "metrics.txt")
with open(metrics_file, "w") as file:
    file.write("Accuracy: " + str(accuracy)
               + "\nPrecision: " + str(precision)
                + "\nRecall: " + str(recall)
                + "\nF1: " + str(f1)
                + "\nAUC-ROC: " + str(auc_roc)
                + "\nAUC-PR: " + str(auc_pr))
