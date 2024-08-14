''' Machine Learning (assignment-1)
    Name: Bhargav
    Roll no. 21dcs022 '''
    
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('air_quality.csv')

print('Loading\n')


sorted(set(data.pollutant_id))

# Setting arbitrary levels : There are 5 pollutants, therefore dictionary is created for dangerous threshold,
# if lower, it is acceptable, else it is not acceptable

dangerous_thresholds = {
 "NO2": 25,
 "CO": 10,
 "NH3": 80,
 "OZONE": 20,
 "PM10": 30,

 "PM2.5": 65,
 "SO2": 50
}

# Adding column is Acceptable

for index,row in data.iterrows():
 if not np.isnan(row["pollutant_avg"]):
    isAcceptable = "No"
 if row["pollutant_avg"] < dangerous_thresholds[row["pollutant_id"]]:
    isAcceptable = "Yes"
 if np.isnan(row["pollutant_avg"]): isAcceptable = "nan"
 
 data.at[index,'isAcceptable'] = isAcceptable
 
 
 
# Data visualization
 
# Bar chart  
plt.figure(figsize=(10, 6))
data['isAcceptable'].value_counts().plot(kind='bar')
plt.title('Count of Acceptable and Unacceptable Air Quality')
plt.xlabel('isAcceptable')
plt.ylabel('Count')
plt.savefig('outputs/acceptable_air_quality_bar_chart.png')  # Save the bar chart as a PNG file
plt.close()  # Close the figure
print ('Saved bar chart')

# Pie chart
plt.figure(figsize=(8, 8))
data['isAcceptable'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Acceptable and Unacceptable Air Quality')
plt.savefig('outputs/acceptable_air_quality_pie_chart.png')  # Save the pie chart as a PNG file
plt.close()  # Close the figure
print('Saved pie chart')

#Data preprocessing

# Removing all nan values
data_without_nan = data.dropna(subset=["pollutant_avg"])
print("data size:" + str(data.shape[0]))
print("data without nan size:" + str(data_without_nan.shape[0]))

print('\npreprocessing :removed nan values')

# Encoding categorical features

encoder = OneHotEncoder()
# Fit and transform the 'pollutant_id' column
encoded_pollutant_id = encoder.fit_transform(data_without_nan[['pollutant_id']])
print(encoded_pollutant_id.shape[0])
print(data_without_nan.shape[0])

# Create a DataFrame from the encoded features
encoded_data = pd.DataFrame(encoded_pollutant_id.toarray(),
 columns=encoder.get_feature_names_out(['pollutant_id']))

# Concatenate the encoded features with the original DataFrame
data_without_nan.reset_index(drop=True, inplace=True)
encoded_data.reset_index(drop=True, inplace=True)
data_encoded = pd.concat([data_without_nan, encoded_data], axis=1)
data_encoded.drop(columns=["pollutant_id"], inplace=True)


# Scatter plot for latitude vs. pollutant_avg
plt.figure(figsize=(8, 6))
sns.scatterplot(x='latitude', y='pollutant_avg', hue='isAcceptable', data=data_encoded)
plt.title('Latitude vs. Pollutant Average')
plt.xlabel('Latitude')
plt.ylabel('Pollutant Average')
plt.savefig('outputs/pollutant_avg') # Save the plot as png file
plt.close()  # Close the figure
print('saved pollutant_avg')

# Scatter plot for longitude vs. pollutant_avg
plt.figure(figsize=(8, 6))
sns.scatterplot(x='longitude', y='pollutant_avg', hue='isAcceptable', data=data_encoded)
plt.title('Longitude vs. Pollutant Average')
plt.xlabel('Longitude')
plt.ylabel('Pollutant Average')
plt.savefig('outputs/pollutant_avg_1')    #Save the plot as png file   
plt.close()  # Close the figure
print('saved pollutant_avg_1')



# Initialize the label encoder
label_encoder = LabelEncoder()

# Columns that are not needed for model training
not_needed_columns = ["country", "state", "city", "station","last_update","isAcceptable"]

# Feature matrix X (all columns except not_needed_columns)
X = data_encoded.drop(columns=not_needed_columns)

# Target vector y (encoded 'isAcceptable' column)
y = label_encoder.fit_transform(data_encoded["isAcceptable"])


# Checkpoint
print("\ncheckpoint : checking datatype inconsistency and unhandled nan entries")
print(X.dtypes)             # Check for non-numeric data in X
print(X.isnull().sum())     # Check for missing values



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print sorted set of 'isAcceptable' values
print("\nisAcceptable values: ")
print(sorted(set(data_encoded["isAcceptable"])))

# Initialize the K-NN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

print( '\nknn applied to training set')
# Fit the model to the training data
knn.fit(X_train, y_train)



# hperparameter optimization
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print ("\nbest k:")
print(best_k)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(roc_curve(y_test, y_pred))

#confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('outputs/confusion_matrix')
print('saved confusion matrix')
plt.close()  # Close the figure
