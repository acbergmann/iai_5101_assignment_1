from sklearn.metrics import classification_report
import traceback
from time import perf_counter, time

import dill as dill
import numpy as np
from sklearn.metrics import RocCurveDisplay
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
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=20)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

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
naive_bayes.fit(X_train, Y_train)
# Predict on test data
Y_predicted = naive_bayes.predict(X_test)

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(Y_test, Y_predicted)))
print("Classification Report is:\n", classification_report(Y_test, Y_predicted))
print("\n F1:\n", f1_score(Y_test, Y_predicted))
print("\n Precision score is:\n", precision_score(Y_test, Y_predicted))
print("\n Recall score is:\n", recall_score(Y_test, Y_predicted))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(Y_test, Y_predicted)))
# ----------------------------------------------------------------------------------------------------------------------
# C. Model Evaluation & Comparison (35 marks):
# Write a Function to detect the model’s Accuracy by applying the trained model
# on a testing dataset to find the predicted labels of Status. Was there overfitting?
accuracy = accuracy_score(Y_test, Y_predicted)
# Check for overfitting by comparing the accuracy of the model on the training dataset
# with the accuracy of the model on the testing dataset
train_accuracy = naive_bayes.score(X_train, Y_train)
overfitting = train_accuracy - accuracy

print(overfitting)
##overfitting=-0.0024396091229852424 -> close to zero, so no overfitting neither underfitting

##If the overfitting value is positive, it indicates that the model may be overfitting
# to the training dataset. If the overfitting value is negative, it indicates that the
# model may be underfitting the training dataset. If the overfitting value is close to zero,
# it suggests that the model is performing well and not overfitting or underfitting the
# training dataset.

# ------------------------
# Tune the model using GridSearchCV (5 marks)
print("Naive Bayes with GridSearchCV")
# Tune the model using GridSearchCV
param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
grid_search = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
best_model_ = grid_search.fit(X_train, Y_train)
y_pred_2 = best_model_.predict(X_test)
print("Classification Report is:\n", classification_report(Y_test, y_pred_2))
print("\n F1:\n", f1_score(Y_test, y_pred_2))
print("\n Precision score is:\n", precision_score(Y_test, y_pred_2))
print("\n Recall score is:\n", recall_score(Y_test, y_pred_2))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(Y_test, y_pred_2)))

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
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0, verbose=True, shrinking=False)
# svm.fit(X_train, y_train)
# svm_y_pred=svm.predict(X_test)
# accuracy_svm=accuracy_score(y_test, svm_y_pred)
# print('The accuracy of the SVM classifier on test data is', accuracy_svm)
# print("Classification Report is:\n", classification_report(y_test, svm_y_pred))
# print("\n F1:\n", f1_score(y_test, svm_y_pred))
# print("\n Precision score is:\n", precision_score(y_test, svm_y_pred))
# print("\n Recall score is:\n", recall_score(y_test, svm_y_pred))
# print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, svm_y_pred)))
# Plot data points
# plt.plot(X_train, y_train, 'o')
#
# svm = SVC()

try:
    # Load the data from the file using dill.load()
    with open('fit_history.pkl', 'rb') as f:
        fit_history = dill.load(f)
        print(fit_history)
except Exception:
    fit_history = False
    print('Could not find or open result file')
    traceback.print_exc()

try:
    # Load the data from the file using dill.load()
    with open('svm_y_prediction.pkl', 'rb') as f:
        svm_y_prediction = dill.load(f)
        print(svm_y_prediction)
except Exception:
    svm_y_prediction = False
    print('Could not find or open result file')
    traceback.print_exc()

try:
    # Load the data from the file using dill.load()
    with open('svm_score.pkl', 'rb') as f:
        svm_score = dill.load(f)
        print(svm_score)
except Exception:
    svm_score = False
    print('Could not find or open result file')
    traceback.print_exc()


def fit_svm_data(x_train, x_test, y_train, y_test):
    print("Fitting process is now beginning...")
    tic = perf_counter()
    fit_history = svm.fit(x_train, y_train)
    toc = perf_counter()
    print(f"SVM Fitting took in {toc - tic:0.4f} seconds")

    print("Prediction process is now beginning...")
    tic_pred = perf_counter()
    svm_y_prediction = svm.predict(x_test)
    toc_pred = perf_counter()
    print(f"SVM Prediction took in {toc_pred - tic_pred:0.4f} seconds")

    print("Score process is now beginning...")
    tic_score = perf_counter()
    svm_score = svm.score(x_test, y_test)
    toc_score = perf_counter()
    print(f"SVM Score took in {toc_score - tic_score:0.4f} seconds")

    svm_fitting_result_data = [
        fit_history,
        svm_y_prediction,
        svm_score
    ]

    try:
        # # Apparently .npy does not work for complex data
        # np.save('fit_history_result.npy', svm_fitting_result_data)

        # Save the data to a file using dill.dump()
        with open('fit_history.pkl', 'wb') as f:
            dill.dump(fit_history, f)
    except Exception:
        print("Fit history save failed")
        traceback.print_exc()

    try:
        # # Apparently .npy does not work for complex data
        # np.save('fit_history_result.npy', svm_fitting_result_data)

        # Save the data to a file using dill.dump()
        with open('svm_y_prediction.pkl', 'wb') as f:
            dill.dump(svm_y_prediction, f)
    except Exception:
        print("Prediction save failed")
        traceback.print_exc()

    try:
        # # Apparently .npy does not work for complex data
        # np.save('fit_history_result.npy', svm_fitting_result_data)

        # Save the data to a file using dill.dump()
        with open('svm_score.pkl', 'wb') as f:
            dill.dump(svm_score, f)
    except Exception:
        print("Score save failed")
        traceback.print_exc()

    print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm_score))

    return svm_fitting_result_data


if fit_history == False or svm_y_prediction == False or svm_score == False:
    # # Mock/Test
    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = np.dot(X, np.array([1, 2])) + 3
    #
    # history = fit_svm_data(X, y)

    # Original
    svm_fitting_result_data_obj = fit_svm_data(X_train, X_test, Y_train, Y_test)
    fit_history = svm_fitting_result_data_obj[0]
    svm_y_prediction = svm_fitting_result_data_obj[1]
    svm_score = svm_fitting_result_data_obj[2]
else:
    while True:
        response = input("Do you want to load the previously generated results? (Y)es/(N)o:")
        if response.upper() == 'Y':
            print("Continuing...")
            break
        elif response.upper() == 'N':
            print("Exiting and beginning fitting...")

            # # Mock/Test
            # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            # y = np.dot(X, np.array([1, 2])) + 3
            #
            # history = fit_svm_data(X,y)

            # Original
            svm_fitting_result_data_obj = fit_svm_data(X_train, X_test, Y_train, Y_test)
            fit_history = svm_fitting_result_data_obj[0]
            svm_y_prediction = svm_fitting_result_data_obj[1]
            svm_score = svm_fitting_result_data_obj[2]

            break
        else:
            print("Invalid input. Please enter Y or N.")

print(svm_fitting_result_data_obj)
print('Fitting process completed')

# ------------------------
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
best_model = grid_search.fit(X_train, Y_train)
dt_pred = best_model.predict(X_test)
print("Classification Report is:\n", classification_report(Y_test, dt_pred))
print("\n F1:\n", f1_score(Y_test, dt_pred))
print("\n Precision score is:\n", precision_score(Y_test, dt_pred))
print("\n Recall score is:\n", recall_score(Y_test, dt_pred))
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(Y_test, dt_pred)))
# ------------------------
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
# ------------------------
# # Naïve Bayes with gridsearchCV
# # Accuracy: 0.8004
# # Sensitivity/Recall: 0.001061249241964827
# # Specificity: TN/N
# # F1 score: 0.0021109770808202654
# # Precision score 0.19444444444444445
# Classification Report is:
#                precision    recall  f1-score   support
#            0       0.80      1.00      0.89     26561
#            1       0.19      0.00      0.00      6596
#     accuracy                           0.80     33157
#    macro avg       0.50      0.50      0.45     33157
# weighted avg       0.68      0.80      0.71     33157
# ------------------------
# # SVM
# # Accuracy: 0.8004
# # Sensitivity/Recall: 0.008489993935718617
# # Specificity: TN/N
# # F1 score: 0.016644375092881556
# # Precision score 0.42105263157894735
# Classification Report is:
#                precision    recall  f1-score   support
#
#            0       0.80      1.00      0.89     26561
#            1       0.42      0.01      0.02      6596
#
#     accuracy                           0.80     33157
#    macro avg       0.61      0.50      0.45     33157
# weighted avg       0.73      0.80      0.72     33157
# ------------------------
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
# ------------------------
# Naïve Bayes AUC: 0.532775447681115
# Naïve Bayes with GridSearchCV AUC: 0.4999847114400028
# Decision Tree AUC: 0.5607590377050994
# SVM AUC: 0.5027955033493962
# # ----------------------------------------------------------------------------------------------------------------------
# # 4. Carry out a ROC analysis to compare the performance of the Naïve Bayes, SVM model with
# # the Decision Tree model. Plot the ROC graph of the models. (10 marks)

# calculate FPR, TPR, and threshold values for each model
nb_fpr, nb_tpr, nb_thresholds = roc_curve(Y_test, Y_predicted)
nb2_fpr, nb2_tpr, nb2_thresholds = roc_curve(Y_test, y_pred_2)
dt_fpr, dt_tpr, dt_thresholds = roc_curve(Y_test, dt_pred)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(Y_test, svm_y_prediction)

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
nb_auc = roc_auc_score(Y_test, Y_predicted)
nb2_auc = roc_auc_score(Y_test, y_pred_2)
dt_auc = roc_auc_score(Y_test, dt_pred)
svm_auc = roc_auc_score(Y_test, svm_y_prediction)

# print the AUC values
print('Naïve Bayes AUC:', nb_auc)
print('Naïve Bayes with GridSearchCV AUC:', nb2_auc)
print('Decision Tree AUC:', dt_auc)
print('SVM AUC:', svm_auc)
plt.savefig('roc_analysis.png')
plt.show()
