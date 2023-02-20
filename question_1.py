import json

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Read the data using csv
data = pd.read_csv('MedicalCentre.csv')
print(data.shape)
# See initial 5 records
print(data.head(5))
# ----------------------------------------------------------------------------------------------------------------------
# A-1. Prepare the data for downstream processes, e.g., deal with missing values, duplicates:
# Missing values:
# Count missing values in DataFrame
missing = pd.isnull(data).sum()
print(missing)
##Only Age is missing - 3 rows
# Verify null info in Age
bool_series = pd.isnull(data["Age"])
# filtering data and displaying data only with age = NaN
print(data[bool_series])
# Drop all the missing values/null values
data.dropna(inplace=True)
# Check the information of DataFrame
missing_1 = pd.isnull(data).sum()
print(missing_1)
##No more missing values
# --------------------------------------------------
# Duplicates:
print(data.shape)
duplicates = data.drop_duplicates()
print(data.shape)
##Didn't find any duplicates
# ----------------------------------------------------------------------------------------------------------------------
# A.2. Determine the frequency of distinct values in each feature set:
# Frequency plots:
plt.figure(figsize=(35, 20))
sns.set_style(style='whitegrid')
# --------------------------------------------------------------
# Gender
plt.subplot(3, 4, 1)
# create the data
gender_counts = pd.Series(data['Gender']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(gender_counts)), gender_counts.values, align='center')
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Gender Rating Distribution")
# --------------------------------------------------------------
# Appointment Day
plt.subplot(3, 4, 2)
# create the data
appoint_counts = pd.Series(data['AppointmentDay']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(appoint_counts)), appoint_counts.values, align='center')
plt.xlabel("Appointment Day")
plt.ylabel("Frequency")
plt.title("Appointment Day Rating Distribution")
# --------------------------------------------------------------
# Age
plt.subplot(3, 4, 3)
# create the data
age_counts = pd.Series(data['Age']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(age_counts)), age_counts.values, align='center')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Rating Distribution")
# --------------------------------------------------------------
# Neighbourhood
plt.subplot(3, 4, 4)
# create the data
neigh_counts = pd.Series(data['Neighbourhood']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(neigh_counts)), neigh_counts.values, align='center')
plt.xlabel("Neighbourhood")
plt.ylabel("Frequency")
plt.title("Neighbourhood Rating Distribution")
# --------------------------------------------------------------
# Scholarship
plt.subplot(3, 4, 5)
# create the data
scholar_counts = pd.Series(data['Scholarship']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(scholar_counts)), scholar_counts.values, align='center')
plt.xlabel("Scholarship")
plt.ylabel("Frequency")
plt.title("Scholarship Rating Distribution")
# --------------------------------------------------------------
# Hypertension
plt.subplot(3, 4, 6)
# create the data
hyper_counts = pd.Series(data['Hypertension']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(hyper_counts)), hyper_counts.values, align='center')
plt.xlabel("Hypertension")
plt.ylabel("Frequency")
plt.title("Hypertension Rating Distribution")
# --------------------------------------------------------------
# Diabetes
plt.subplot(3, 4, 7)
# create the data
diab_counts = pd.Series(data['Diabetes']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(diab_counts)), diab_counts.values, align='center')
plt.xlabel("Diabetes")
plt.ylabel("Frequency")
plt.title("Diabetes Rating Distribution")
# --------------------------------------------------------------
# Alcoholism
plt.subplot(3, 4, 8)
# create the data
alco_counts = pd.Series(data['Alcoholism']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(alco_counts)), alco_counts.values, align='center')
plt.xlabel("Alcoholism")
plt.ylabel("Frequency")
plt.title("Alcoholism Rating Distribution")
# --------------------------------------------------------------
# Handicap
plt.subplot(3, 4, 9)
# create the data
handi_counts = pd.Series(data['Handicap']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(handi_counts)), handi_counts.values, align='center')
plt.xlabel("Handicap")
plt.ylabel("Frequency")
plt.title("Handicap Rating Distribution")
# --------------------------------------------------------------
# SMS
plt.subplot(3, 4, 10)
# create the data
SMS_counts = pd.Series(data['SMS_received']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(SMS_counts)), SMS_counts.values, align='center')
plt.xlabel("SMS")
plt.ylabel("Frequency")
plt.title("SMS Rating Distribution")
# --------------------------------------------------------------
# No_Show
plt.subplot(3, 4, 11)
# create the data
noshow_counts = pd.Series(data['No-show']).value_counts()
# Plot the data/Add X Label on X-axis/Add X Label on X-axis/Add a title to graph/Show the plot
plt.bar(range(len(noshow_counts)), noshow_counts.values, align='center')
plt.xlabel("No-show")
plt.ylabel("Frequency")
plt.title("No-show Rating Distribution")
plt.show()
####Explain why not frequency???????????????? (PatientID, AppointmentID, ScheduledDay)
# ----------------------------------------------------------------------------------------------------------------------
# # A.3. Initialize a function to plot relevant features within the dataset to visualize for outliers:
# Use box plot to check for outliers, deal with them using scaling
plt.figure(figsize=(35, 20))
sns.set_style(style='whitegrid')
plt.subplot(3, 4, 1)
plt.xlabel("Gender")
sns.boxplot(data=gender_counts)
plt.subplot(3, 4, 2)
plt.xlabel("Appointment")
sns.boxplot(data=appoint_counts)
plt.subplot(3, 4, 3)
plt.xlabel("Age")
sns.boxplot(data=age_counts)
plt.subplot(3, 4, 3)
plt.xlabel("Neighborhood")
sns.boxplot(data=neigh_counts)
plt.subplot(3, 4, 4)
plt.xlabel("Scholarship")
sns.boxplot(data=scholar_counts)
plt.subplot(3, 4, 5)
plt.xlabel("Hypertension")
sns.boxplot(data=hyper_counts)
plt.subplot(3, 4, 6)
plt.xlabel("Diabets")
sns.boxplot(data=diab_counts)
plt.subplot(3, 4, 7)
plt.xlabel("Alcoholism")
sns.boxplot(data=alco_counts)
plt.subplot(3, 4, 8)
plt.xlabel("Handicap")
sns.boxplot(data=handi_counts)
plt.subplot(3, 4, 9)
plt.xlabel("SMS")
sns.boxplot(data=SMS_counts)
plt.subplot(3, 4, 10)
plt.xlabel("No-show")
sns.boxplot(data=noshow_counts)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# A4. Count the frequency of negative Age feature observations, and remove them
# count negatives
count = 0
for Age in data['Age']:
    if Age < 0:
        count += 1
print("There is ", count, " negative Age feature observation.")

# remove negatives
data = data[~(data['Age'] < 0)]
count = 0
for Age in data['Age']:
    if Age < 0:
        count += 1
print("There is ", count, " negative Age feature observation.")
# ----------------------------------------------------------------------------------------------------------------------
# A5. The values within AwaitingTime are negative, transform them into positive values
# There is no variable named AwaitingTime. (Are we supposed to calculate it?)

data["AwaitingTime"]=(pd.to_datetime(data['AppointmentDay'], format='%Y-%m-%d'))-(pd.to_datetime(data['ScheduledDay'], format='%Y-%m-%d'))
data["AwaitingTime"]=abs(((data["AwaitingTime"].dt.total_seconds() / 60)/60)/24)
print (data["AwaitingTime"])
#Now we have a new variable which is waiting time in days

# ----------------------------------------------------------------------------------------------------------------------
# 6. ML algorithm requires the variables to be coded into its equivalent integer codes.
# (Gender, Neighbourhood, No-show)
# Dummy encoding -> Gender
encoded_data_gen = pd.get_dummies(data['Gender'])
# Join the encoded_data with original dataframe
data = data.join(encoded_data_gen)
# Dummy encoding -> No-show
encoded_data_noshow_ = pd.get_dummies(data['No-show'])
# Join the encoded_data with original dataframe
data = data.join(encoded_data_noshow_)
# Instantiate the Label Encoder Object
label_encoder = LabelEncoder()
# Fit and transform the column
encoded_data_nei = label_encoder.fit_transform(data['Neighbourhood'])
numpy_nei = np.array(encoded_data_nei)
# Join the encoded _data with original dataframe
data['enconded_neighbourhood'] = numpy_nei.tolist()
print(data.info())
# PatientID, Age, Gender(F,M), No_show(No, Yes) -> int;
data['Age'] = data['Age'].astype(int)
data['PatientID'] = data['PatientID'].astype(int)
data['F'] = data['F'].astype(int)
data['M'] = data['M'].astype(int)
data['No'] = data['No'].astype(int)
data['Yes'] = data['Yes'].astype(int)
print(data.dtypes)
# ----------------------------------------------------------------------------------------------------------------------
# A7. Separate the date features into date components
data['ScheduledDay_'], data['ScheduledHour_'] = data['ScheduledDay'].str.split(pat='T', n=1).str
data['AppointmentDay_'], data['AppointmentHour_'] = data['AppointmentDay'].str.split(pat='T', n=1).str
clean_data = data[
    ["PatientID", "AppointmentID", "F", "ScheduledDay_", "AppointmentDay_", "Age", "enconded_neighbourhood",
     "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap", "SMS_received", "AwaitingTime", "Yes"]].copy()
clean_data.rename(columns={"F": "Gender", "enconded_neighbourhood": "Neighbourhood", "Yes": "No_show_1"}, inplace=True)
print(clean_data.info())
# ----------------------------------------------------------------------------------------------------------------------
# A8. ML algorithms work best when the input data are scaled to a narrow range around zero.
# Rescale the age feature with a normalizing (e.g., min_max normalization) or standardization (e.g., z_score standardization) function.
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

scaler = MinMaxScaler()
# To scale data
scaler.fit(clean_data['Age'].values.reshape(-1, 1))
clean_data['Age'] = scaler.transform(clean_data['Age'].values.reshape(-1, 1))
print(clean_data.info())
# ----------------------------------------------------------------------------------------------------------------------
# A9. Conduct variability comparison between features using a correlation matrix & drop correlated features
# Determine correlated features using a heatmap
plt.figure(figsize=(15, 10))
sns.set_style(style='whitegrid')
plt.subplot(1, 1, 1)
clean_data_ = clean_data[
    ["Gender", "Age", "Neighbourhood", "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap",
     "SMS_received", "AwaitingTime", "No_show_1"]].copy()
corrmat = clean_data_.corr()
print(corrmat)
sns.heatmap(corrmat, annot=True)
plt.show()

clean_data_.to_csv('my_clean_data.csv', index=False)