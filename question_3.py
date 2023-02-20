import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Read the data using csv
data = pd.read_csv('my_clean_data.csv')
print(data)

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
# 3. Using the same data set partitioning method, evaluate the performance of
# a SVM and Decision tree classifier on the dataset. Compare the results of the
# Naïve Bayes classifier with SVM and
# Decision tree model according to the following criteria:
# Accuracy, Sensitivity, Specificity & F1 score.
# Identify the model that performed best and worst according to each criterion. (10 marks)
#------------------------
#
# # # #FIXME: Not working properly
# print("SVM")
# svm = SVC()
# svm.fit(X_train, y_train)
# svm_y_pred=svm.predict_proba(X_test)[:,1]
# print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test, y_test)))

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
# # Accuracy: 0.7939
# # Sensitivity/Recall: 0.04639175257731959
# # Specificity: TN/N
# # F1 score: 0.08221386351423966
# # Precision score 0.3608490566037736
# Classification Report is:
#                precision    recall  f1-score   support
#            0       0.81      0.98      0.88     26561
#            1       0.36      0.05      0.08      6596
#     accuracy                           0.79     33157
#    macro avg       0.58      0.51      0.48     33157
# weighted avg       0.72      0.79      0.72     33157
#------------------------
# # SVM
# # Accuracy:
# #Sensitivity/Recall:
# # Specificity &
# # F1 score.
#------------------------
# # Decision tree model
# # Accuracy: 0.7873
# # Sensitivity/Recall: 0.06898120072771377
# # Specificity:
# # F1 score: 0.11429289123335845
# # Precision score: 0.3330893118594436
# Classification Report is:
#                precision    recall  f1-score   support
#            0       0.81      0.97      0.88     26561
#            1       0.33      0.07      0.11      6596
#     accuracy                           0.79     33157
#    macro avg       0.57      0.52      0.50     33157
# weighted avg       0.71      0.79      0.73     33157
# # ----------------------------------------------------------------------------------------------------------------------
# # 4. Carry out a ROC analysis to compare the performance of the Naïve Bayes, SVM model with
# # the Decision Tree model. Plot the ROC graph of the models. (10 marks)







label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)
label_binarizer.transform(["No_show_1"])

class_of_interest = "No_show_1"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)
class_id


print("ROC Naive Bayes")
RocCurveDisplay.from_predictions(
    y_test[:, class_id],
    y_predicted[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

print("Decison Tree")
RocCurveDisplay.from_predictions(
    y_test[:, class_id],
    dt_pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()