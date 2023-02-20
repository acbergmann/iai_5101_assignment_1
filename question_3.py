from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from seaborn import pairplot
import seaborn as sns
import numpy as np

# Read the data using csv
data = pd.read_csv('my_clean_data.csv')
print(data)
# pairplot(data, hue='No_show_1')
# plt.show()

# Splitting data into train and test
target_name = 'No_show_1'
# given predictions - training data
y = data[target_name]
# dropping the Outcome column and keeping all other columns as X
X = data.drop(target_name, axis=1)

# splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# ----------------------------------------------------------------------------------------------------------------------
# B. Model Development (20 marks):
# 1. Develop a Naïve Bayes classifier to predict the outcome of the test data using Python.
# The performance of the classifier should be evaluated by partitioning the dataset into a train dataset (70%)
# and test dataset (30%).
# Use the train dataset to build the Naïve Bayes and the test dataset to
# evaluate how well the model generalizes to future results.
print("Naive Bayes")

# Calling the Class
naive_bayes = GaussianNB()
# Fitting the data to the classifier
naive_bayes.fit(X_train, y_train)
# Predict on test data
y_predicted = naive_bayes.predict(X_test)

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predicted)))
print("Classification Report is:\n",classification_report(y_test,y_predicted))
print("\n F1:\n",f1_score(y_test,y_predicted))
print("\n Precision score is:\n",precision_score(y_test,y_predicted))
print("\n Recall score is:\n",recall_score(y_test,y_predicted))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predicted)))
# ----------------------------------------------------------------------------------------------------------------------
# C. Model Evaluation & Comparison (35 marks):
# Write a Function to detect the model’s Accuracy by applying the trained model
# on a testing dataset to find the predicted labels of Status. Was there overfitting?
accuracy = accuracy_score(y_test, y_predicted)
# Check for overfitting by comparing the accuracy of the model on the training dataset
# with the accuracy of the model on the testing dataset
train_accuracy = naive_bayes.score(X_train, y_train)
overfitting = train_accuracy - accuracy

print (overfitting)
##overfitting=-0.0024396091229852424 -> close to zero, so no overfitting neither underfitting

##If the overfitting value is positive, it indicates that the model may be overfitting
# to the training dataset. If the overfitting value is negative, it indicates that the
# model may be underfitting the training dataset. If the overfitting value is close to zero,
# it suggests that the model is performing well and not overfitting or underfitting the
# training dataset.

#------------------------
# Tune the model using GridSearchCV (5 marks)
print("Naive Bayes with GridSearchCV")
# Tune the model using GridSearchCV
param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
grid_search = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
best_model_ = grid_search.fit(X_train, y_train)
y_pred_2 = best_model_.predict(X_test)
print("Classification Report is:\n",classification_report(y_test,y_pred_2))
print("\n F1:\n",f1_score(y_test,y_pred_2))
print("\n Precision score is:\n",precision_score(y_test,y_pred_2))
print("\n Recall score is:\n",recall_score(y_test,y_pred_2))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_2)))

# ----------------------------------------------------------------------------------------------------------------------
# 3. Using the same data set partitioning method, evaluate the performance of
# a SVM and Decision tree classifier on the dataset. Compare the results of the
# Naïve Bayes classifier with SVM and
# Decision tree model according to the following criteria:
# Accuracy, Sensitivity, Specificity & F1 score.
# Identify the model that performed best and worst according to each criterion. (10 marks)
#
# # # #FIXME: Not working properly
print("SVM")
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0, verbose=True)
svm.fit(X_train, y_train)
svm_y_pred=svm.predict(X_test)
accuracy_svm=accuracy_score(y_test, svm_y_pred)
print('The accuracy of the SVM classifier on test data is', accuracy_svm)
print("Classification Report is:\n", classification_report(y_test, svm_y_pred))
print("\n F1:\n", f1_score(y_test, svm_y_pred))
print("\n Precision score is:\n", precision_score(y_test, svm_y_pred))
print("\n Recall score is:\n", recall_score(y_test, svm_y_pred))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, svm_y_pred)))


# model = SVC()
# kernel = ['poly', 'rbf', 'sigmoid']
# C = [50, 10, 1.0, 0.1, 0.01]
# gamma = ['scale']
# #%%
# # define grid search
# grid = dict(kernel=kernel,C=C,gamma=gamma)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',error_score=0)
#------------------------
print("Decision Tree")
# Decision tree
dt = DecisionTreeClassifier(random_state=2)
# Create the parameter grid based on the results of random search
params = {
    'max_depth': [5, 10, 20, 25],
    'min_samples_leaf': [10, 20, 50, 100, 120],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4, n_jobs=-1, verbose=1, scoring="f1")
best_model = grid_search.fit(X_train, y_train)
dt_pred = best_model.predict(X_test)
print("Classification Report is:\n", classification_report(y_test, dt_pred))
print("\n F1:\n", f1_score(y_test, dt_pred))
print("\n Precision score is:\n", precision_score(y_test, dt_pred))
print("\n Recall score is:\n", recall_score(y_test, dt_pred))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, dt_pred)))
#------------------------
# # Naïve Bayes
# # Accuracy: 0.7716
# # Sensitivity/Recall: 0.13614311704063067
# # Specificity: TN/N
# # F1 score: 0.19169601878535597
# # Precision score 0.32383699963937973
# Classification Report is:
#                precision    recall  f1-score   support
#
#            0       0.81      0.93      0.87     26561
#            1       0.32      0.14      0.19      6596
#
#     accuracy                           0.77     33157
#    macro avg       0.57      0.53      0.53     33157
# weighted avg       0.72      0.77      0.73     33157
#------------------------
# # Naïve Bayes with gridsearchCV
# # Accuracy: 0.8004
# # Sensitivity/Recall: 0.001061249241964827
# # Specificity: TN/N
# # F1 score: 0.0021109770808202654
# # Precision score 0.19444444444444445
#Classification Report is:
#                precision    recall  f1-score   support
#            0       0.80      1.00      0.89     26561
#            1       0.19      0.00      0.00      6596
#     accuracy                           0.80     33157
#    macro avg       0.50      0.50      0.45     33157
# weighted avg       0.68      0.80      0.71     33157
#------------------------
# # SVM
# # Accuracy: 0.8011
# # Sensitivity/Recall: 0
# # Specificity: TN/N
# # F1 score: 0
# # Precision score 0
# Classification Report is:
#                precision    recall  f1-score   support
#            0       0.80      1.00      0.89     26561
#            1       0.00      0.00      0.00      6596
#     accuracy                           0.80     33157
#    macro avg       0.40      0.50      0.44     33157
# weighted avg       0.64      0.80      0.71     33157
#------------------------
# # Decision tree model
# # Accuracy: 0.7670
# # Sensitivity/Recall: 0.2183141297756216
# # Specificity:
# # F1 score: 0.271518808334119
# # Precision score: 0.35901271503365745
# Classification Report is:
#                precision    recall  f1-score   support
#
#            0       0.82      0.90      0.86     26561
#            1       0.36      0.22      0.27      6596
#
#     accuracy                           0.77     33157
#    macro avg       0.59      0.56      0.57     33157
# weighted avg       0.73      0.77      0.74     33157
#------------------------
# Model accuracy score: 0.7670
# Naïve Bayes AUC: 0.532775447681115
# Naïve Bayes with GridSearchCV AUC: 0.4999847114400028
# Decision Tree AUC: 0.5607590377050994
# SVM AUC: 0.5
# # ----------------------------------------------------------------------------------------------------------------------
# # 4. Carry out a ROC analysis to compare the performance of the Naïve Bayes, SVM model with
# # the Decision Tree model. Plot the ROC graph of the models. (10 marks)

# calculate FPR, TPR, and threshold values for each model
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, y_predicted)
nb2_fpr, nb2_tpr, nb2_thresholds = roc_curve(y_test, y_pred_2)
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_pred)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_y_pred)

# plot the ROC curves for both models
plt.figure(figsize=(35, 20))
sns.set_style(style='whitegrid')

plt.subplot(2, 2, 1)
plt.plot(nb_fpr, nb_tpr, label='Naïve Bayes')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(nb2_fpr, nb2_tpr, label='Naïve Bayes with GridShearchCV')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(dt_fpr, dt_tpr, label='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(svm_fpr, svm_tpr, label='SVM')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# calculate the AUC for each model
nb_auc = roc_auc_score(y_test, y_predicted)
nb2_auc = roc_auc_score(y_test, y_pred_2)
dt_auc = roc_auc_score(y_test, dt_pred)
svm_auc = roc_auc_score(y_test, svm_y_pred)

# print the AUC values
print('Naïve Bayes AUC:', nb_auc)
print('Naïve Bayes with GridSearchCV AUC:', nb2_auc)
print('Decision Tree AUC:', dt_auc)
print('SVM AUC:', svm_auc)
plt.show()