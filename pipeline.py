import libs
import ml_algos
import pre_ml1

#Voting Classifier(To try)

results_df = libs.pd.DataFrame(columns=['Algorithm', 'Parameters', 'Best Score', 'Best Model', 'Accuracy', 'Precision', 'Recall',
                                   'F1-Score', 'AUC-ROC', 'AUC-PR', 'Classification Report'])

for algo in ml_algos.algorithms:

    pipeline = libs.Pipeline([
        ('preprocessor', pre_ml1.preprocessor),
        ('selector', libs.SelectKBest(score_func=libs.mutual_info_classif, k=5)),
        ('model', algo['model'])
    ])

    scores = libs.cross_val_score(pipeline, pre_ml1.X_train, pre_ml1.y_train, cv=5, scoring='roc_auc')
    mean_score = scores.mean()

    grid_search = libs.GridSearchCV(pipeline, param_grid=algo['param_grid'], cv=5, scoring='roc_auc', verbose=3)
    grid_search.fit(pre_ml1.X_train, pre_ml1.y_train)

    best_model = grid_search.best_estimator_
    model_name = algo['name']
    model_filename = f'modelswpipe/{model_name}_best_model.pkl'
    libs.joblib.dump(best_model, model_filename)

    y_pred = best_model.predict(pre_ml1.X_test)
    report = libs.classification_report(pre_ml1.y_test, y_pred)
    accuracy = libs.accuracy_score(pre_ml1.y_test, y_pred)
    precision = libs.precision_score(pre_ml1.y_test, y_pred)
    recall = libs.recall_score(pre_ml1.y_test, y_pred)
    f1 = libs.f1_score(pre_ml1.y_test, y_pred)
    auc_roc = libs.roc_auc_score(pre_ml1.y_test, y_pred)
    auc_pr = libs.average_precision_score(pre_ml1.y_test, y_pred)


    results_df = results_df.append({
        'Algorithm': model_name,
        'Parameters': grid_search.best_params_,
        'Best Score': grid_search.best_score_,
        'Best Model': grid_search.best_estimator_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Classification Report': report_filename,
    }, ignore_index=True)

    print(f"Predictions for {model_name}:")
    print(y_pred)
    print(f"True labels for {model_name}:")
    print(pre_ml1.y_test)
    print(f"Classification report for {model_name}:")
    print(report)

# Create a directory to store the classification report files
if not libs.os.path.exists('modelswpipe/reports'):
    libs.os.makedirs('modelswpipe/reports')


# Save the classification reports and the results dataframe
for index, row in results_df.iterrows():
    model_name = row['Algorithm']
    report_filename = f'modelswpipe/reports/{model_name}_classification_report.txt'

    # Get the actual classification report text
    report_text = open(row['Classification Report'], 'r').read()

    # Write the actual classification report text to the file
    with open(report_filename, 'w') as file:
        file.write(report_text)

    # Update the 'Classification Report' column in the results dataframe
    row['Classification Report'] = report_filename

results_df.to_csv('modelswpipe/model_results.csv', index=False)