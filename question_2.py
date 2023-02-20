from question_1 import clean_data_
# ----------------------------------------------------------------------------------------------------------------------
# B. Model Development (20 marks):
# 1. Develop a Naïve Bayes classifier to predict the outcome of the test data using Python.
# The performance of the classifier should be evaluated by partitioning the dataset into a train dataset (70%)
# and test dataset (30%).
# Use the train dataset to build the Naïve Bayes and the test dataset to
# evaluate how well the model generalizes to future results.
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Splitting data into train and test
target_name = 'No_show_1'
# given predictions - training data
y = clean_data_[target_name]
# dropping the Outcome column and keeping all other columns as X
X = clean_data_.drop(target_name, axis=1)

# splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#
# Calling the Class
naive_bayes = GaussianNB()
# Fitting the data to the classifier
naive_bayes.fit(X_train, y_train)
# Predict on test data
y_predicted = naive_bayes.predict(X_test)

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predicted)))
print("Classification Report is:\n", classification_report(y_test, y_predicted))
print("\n F1:\n", f1_score(y_test, y_predicted))
print("\n Precision score is:\n", precision_score(y_test, y_predicted))
print("\n Recall score is:\n", recall_score(y_test, y_predicted))
