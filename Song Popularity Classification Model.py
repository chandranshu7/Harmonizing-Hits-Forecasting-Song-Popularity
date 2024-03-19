#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score


# # Importing the dataset

# In[18]:


df = pd.read_csv("song_data.csv")

df.head()


# # Data Description
# 
# The dataset contains information about songs, with features like 'song_popularity', 'key', 'time_signature', and others.
# The target variable is 'is_popular', which is derived based on the mean popularity of songs.

# In[19]:


#Creating "is_popular" column using mean of popularity
mean_popularity = df['song_popularity'].mean()
df['is_popular'] = np.where(df['song_popularity'] >= mean_popularity, 1, 0)

df.head()


# # Exploratory Data Analysis (EDA)

# In[20]:


# Visualize the distribution of the target variable
plt.figure(figsize=(4, 2))
df['is_popular'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Popularity Classes')
plt.xlabel('Popularity Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[21]:


#Correlation Matrix of the data
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Create correlation matrix
corr_matrix = df[numeric_columns].corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[22]:


# Select numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns

import math

n = 5

# Create a list of colors for the histograms
clr = ['r', 'g', 'b', 'g', 'b', 'r']

# Create histograms
plt.figure(figsize=[15, 4 * math.ceil(len(numeric_columns) / n)])
for i in range(len(numeric_columns)):
    plt.subplot(math.ceil(len(numeric_columns) / n), n, i + 1)
    sns.histplot(df[numeric_columns[i]], color=clr[i % len(clr)], bins=10, edgecolor="black", linewidth=2)
    plt.title(f'Distribution of {numeric_columns[i]}')  # Added title
plt.tight_layout()
plt.show()

# Create boxplots
plt.figure(figsize=[15, 4 * math.ceil(len(numeric_columns) / n)])
for i in range(len(numeric_columns)):
    plt.subplot(math.ceil(len(numeric_columns) / n), n, i + 1)
    sns.boxplot(x=df[numeric_columns[i]], color=clr[i % len(clr)])
    plt.title(f'Boxplot of {numeric_columns[i]}')  # Added title
plt.tight_layout()
plt.show()


# # Data Preprocessing

# In[23]:


#Dropping duplicates
df.drop_duplicates(inplace=True)
df = df.sort_values('song_popularity', ascending=False)
df.count()


# In[24]:


#Checking nulls
df.isnull().sum()


# In[25]:


# Define features and target
X = df.drop(['song_popularity', 'is_popular','key','time_signature','song_name','audio_mode','tempo','speechiness'], axis=1)
Y = df['is_popular']

X.info()


# In[26]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)


# In[33]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[34]:


# Modeling Section
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=50),
    'Logistic Regression': LogisticRegression(C=1.0, random_state=50),
    'SVC': SVC(C=1.0, kernel='rbf', random_state=50),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=50)
}


# In[35]:


classifier_names = []
accuracies = []


# In[36]:


# Train and evaluate each classifier
for classifier_name, classifier in classifiers.items():
    # Train the model
    classifier.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{classifier_name} Accuracy: {accuracy*100:.2f}")
    accuracies.append(accuracy)
    classifier_names.append(classifier_name)

    # Display classification report
    print(f"\n{classifier_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy_train = accuracy_score(y_train, classifier.predict(X_train_scaled))
    accuracy_test = accuracy_score(y_test, y_pred)


    print(f"Training Accuracy: {accuracy_train:.3f}")
    print(f"Test Accuracy: {accuracy_test:.3f}")
    
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
        print(f"\n{classifier_name} Feature Importance:")
        for feature, importance in zip(X_train.columns, feature_importances):
            print(f"{feature}: {importance:.4f}")

    # Display confusion matrix
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{classifier_name} Confusion Matrix")
    plt.show()


# In[37]:


# Results and Conclusion
fig, ax = plt.subplots()

bar_width = 0.30
index = np.arange(len(classifier_names))

bar1 = ax.bar(index, accuracies, bar_width)

# Add labels, title, and legend
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Comparison')
ax.set_xticks(index)
ax.set_xticklabels(classifier_names)


for i, v in enumerate(accuracies):
    ax.text(i, v + 0.01, f'{v*100:.2f}%', ha='center', va='bottom')

# Display the bar graph
plt.show()

