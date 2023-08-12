import libs
import pre_ml1
from sklearn import tree



feature_names = ['TumourDiamater (mm)', 'ExcisionMargin (mm)', 'TumourDepth', 'TumourStatus_Primary',
                 'TumourStatus_Recurrence', 'AnatomicalLoc_cheek', 'AnatomicalLoc_chest', 'AnatomicalLoc_ear',
                 'AnatomicalLoc_forehead', 'AnatomicalLoc_jaw', 'AnatomicalLoc_lip', 'AnatomicalLoc_lower limb',
                 'AnatomicalLoc_neck', 'AnatomicalLoc_nose', 'AnatomicalLoc_scalp', 'AnatomicalLoc_temple',
                 'AnatomicalLoc_trunk', 'AnatomicalLoc_upper limb', 'HistDiag_squamous cell carcinoma'] #, 0, 1, 2, 6, 10, 11, 12
target_names = ['No recurrence', 'Recurrence']

# # Load the trained pipeline model from the .pkl file
# model_filename = 'modelswpipe/Decision Tree_best_model.pkl'
# pipeline = joblib.load(model_filename)
# # print(pipeline)


# selected_features_mask = pipeline.named_steps['selector'].get_support()
# preprocessor = pipeline.named_steps['preprocessor']
# feature_names = []
#
# for transformer_name, transformer, features in preprocessor.transformers_:
#     if transformer_name == 'scaler':
#         feature_names.extend(features)
#     elif transformer_name == 'encoder':
#         encoded_feature_names = []
#         for category, feature in zip(transformer.categories_, features):
#             encoded_feature_names.extend([f"{feature}_{value}" for value in category])
#         feature_names.extend(encoded_feature_names)
#     else:
#         if hasattr(transformer, 'get_feature_names_out'):
#             feature_names.extend(transformer.get_feature_names_out())
#         else:
#             feature_names.extend(features)
#
# selected_feature_names = [feature_names[i] for i, mask_value in enumerate(selected_features_mask) if mask_value]
#
# print(selected_feature_names)
# print(feature_names)
#
# # Access the decision tree model from the 'model' step of the pipeline
# decision_tree = pipeline.named_steps['model']
#
# fig = libs.plt.figure(figsize=(20,15))
# _ = tree.plot_tree(decision_tree,
#                    feature_names=feature_names,
#                    class_names=target_names,
#                    filled=True)
# libs.plt.show()



# model_filename = 'modelswpipe/K-Nearest Neighbors_best_model.pkl'
# pipeline = libs.joblib.load(model_filename)
# knn = pipeline.named_steps['model']
# X_test = pre_ml1.X_test
# y_test = pre_ml1.y_test
# libs.plot_confusion_matrix(knn, X_test, y_test)
# libs.plt.title('KNN Confusion Matrix')
# libs.plt.show()







model_filename = 'modelswpipe/XGBoost_best_model.pkl'
pipeline = libs.joblib.load(model_filename)
xg = pipeline.named_steps['model']


selected_features_mask = pipeline.named_steps['selector'].get_support()
preprocessor = pipeline.named_steps['preprocessor']
feature_names = []

for transformer_name, transformer, features in preprocessor.transformers_:
    if transformer_name == 'scaler':
        feature_names.extend(features)
    elif transformer_name == 'encoder':
        encoded_feature_names = []
        for category, feature in zip(transformer.categories_, features):
            encoded_feature_names.extend([f"{feature}_{value}" for value in category])
        feature_names.extend(encoded_feature_names)
    else:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.extend(features)

selected_feature_names = [feature_names[i] for i, mask_value in enumerate(selected_features_mask) if mask_value]
print('selected_feature_names: ')
print(selected_feature_names)
print('len(selected_feature_names): ')
print(len(selected_feature_names))
print(feature_names)



# Visualize the feature importance
libs.xgb.plot_importance(xg)
libs.plt.title('XGBoost Feature Importance')
libs.plt.show()

# Visualize individual trees
libs.xgb.plot_tree(xg, num_trees=0)
libs.plt.title('XGBoost Decision Tree')
libs.plt.show()



# Filter out the numeric values from the selected_feature_names list
filtered_selected_feature_names = [feature_name for feature_name in selected_feature_names if isinstance(feature_name,
                                                                                                         str)]

# Visualize the feature importance
libs.xgb.plot_importance(xg, importance_type='weight')
libs.plt.title('XGBoost Feature Importance')
libs.plt.xlabel('Importance Score')
libs.plt.ylabel('Feature Name')

# Get the tick positions and labels for the x-axis
ticks, _ = libs.plt.xticks()
ticks = [int(tick) for tick in ticks]  # Convert the ticks to a list of integers

# Update the tick labels with the selected feature names
selected_ticks = [filtered_selected_feature_names[i] for i in ticks if i < len(filtered_selected_feature_names)]

# Set the tick positions and labels
libs.plt.xticks(ticks[:len(selected_ticks)], selected_ticks[:len(ticks)], rotation=90)

libs.plt.show()