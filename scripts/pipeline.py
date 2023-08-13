import libs
import ml_algos
import pre_ml1
import imblearn.pipeline as imbpipeline


results_df = libs.pd.DataFrame(columns=['Algorithm', 'Parameters', 'Best Score', 'Best Model', 'Accuracy', 'Precision', 'Recall',
                                   'F1-Score', 'AUC-ROC', 'AUC-PR', 'Classification Report'])

# Ensemble for each algorithm.
for algo in ml_algos.algorithms:

    # Pipeline for each algorithm.
    pipeline = imbpipeline.Pipeline([
        ('preprocessor', pre_ml1.preprocessor),
        ('sampler', pre_ml1.sampler),
        ('selector', algo['selector']),
        ('model', algo['model'])
    ])
    random_search = libs.RandomizedSearchCV(pipeline, param_distributions=algo['param_grid'],
                                            cv=libs.StratifiedKFold(n_splits=5),
                                            scoring='roc_auc', n_iter=10, random_state=42, verbose=3)

    # grid_search = libs.GridSearchCV(pipeline, param_grid=algo['param_grid'], cv=libs.StratifiedKFold(n_splits=5),
    #                                 scoring='roc_auc', verbose=3)
    random_search.fit(pre_ml1.X_train, pre_ml1.y_train)

    best_model = random_search.best_estimator_
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

    report_filename = f'{model_name}_classification_report.txt'
    with open(report_filename, 'w') as file:
        file.write(report)

    results_df = results_df.append({
        'Algorithm': model_name,
        'Parameters': random_search.best_params_,
        'Best Score': random_search.best_score_,
        'Best Model': random_search.best_estimator_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Classification Report': report_filename
    }, ignore_index=True)

    print(f"Predictions for {model_name}:")
    print(y_pred)
    print(f"True labels for {model_name}:")
    print(pre_ml1.y_test)
    print(f"Classification report for {model_name}_smote:")
    print(report)


if not libs.os.path.exists('modelswpipe/reports'):
    libs.os.makedirs('modelswpipe/reports')

for index, row in results_df.iterrows():
    model_name = row['Algorithm']
    report_filename = f'modelswpipe/reports/{model_name}_classification_report_smote.txt'

    report_text = open(row['Classification Report'], 'r').read()

    with open(report_filename, 'w') as file:
        file.write(report_text)

    results_df.at[index, 'Classification Report'] = report_filename

results_df.to_csv('modelswpipe/model_results.csv', index=False)

