# Harmonizing Hits: Forecasting Song Popularity

## Abstract

Predicting song popularity is crucial in the dynamic music industry, requiring a deep understanding of influencing factors. This project leverages the Song Popularity Dataset, encompassing audio features and metadata for thousands of songs, to employ various classification algorithms for predicting popularity. The analysis focuses on identifying the most influential features within the dataset, providing valuable insights for music industry professionals and enhancing understanding of predictive algorithms driving a song's success.

## Introduction

This project aims to predict song popularity by employing machine learning algorithms and analyzing diverse song characteristics. Understanding the determinants of song popularity is vital due to the substantial impact of the music industry on human culture and its significant revenue generation. This research contributes to the evolving field of "Hit Song Science," utilizing machine learning to predict song popularity. Various algorithms including SVC, logistic regression, Random Forest, and Gradient Boosting Classifier are explored to determine whether a song will achieve popularity without relying on the popularity score itself.

## Related Work

Previous research has extensively investigated the social impact on a song's popularity and utilized machine learning techniques for popularity prediction. This project builds upon previous studies by integrating metadata and audio features, aiming to enhance prediction accuracy. Notable works include those by Bertin-Mahieux et al., Koenignstein et al., Ni et al., Pachet and Roy, and Salganik, Dodds, and Watts.

## Dataset and Features

The study utilizes the Song Popularity Dataset, comprising audio attributes and metadata for nearly thirteen thousand songs. Various features such as duration, acousticness, danceability, energy, instrumentalness, key, loudness, and others are included in the dataset. A new column "is_popular" is introduced, derived from the "song_popularity" feature, to classify songs based on popularity. Features are divided into training and testing sets for robust analysis.

## Methods

### Feature Selection

Feature selection is crucial for effective model training. A correlation matrix is utilized to identify relationships between features. Features finalized for prediction include song duration, acousticness, danceability, energy, instrumentalness, liveness, loudness, and audio valence.

### Classification

Various classifiers including Random Forest, Logistic Regression, Support Vector Classifier (SVC), and Gradient Boosting Classifier are employed for predicting song popularity. Each classifier offers unique strengths in capturing non-linear relationships and feature interactions.

## Experimental Results

Comprehensive analysis reveals varied performances of different classifiers. Random Forest emerges as the top performer with an accuracy of 59.27%, showcasing its ability to capture complex patterns in the data. Logistic Regression, SVC, and Gradient Boosting Classifier also demonstrate competitive accuracies, providing valuable insights into song popularity prediction.

## Conclusion

This project evaluates different classifiers for predicting song popularity, identifying Random Forest as the most accurate model. The findings contribute significantly to music recommendation systems and offer valuable insights for stakeholders in the music industry. Further optimization of the Random Forest model may enhance its performance, highlighting the potency of machine learning in uncovering intricate patterns in music data.

## Dataset Source

[Song Popularity Dataset](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset)

## Supplementary Material

[Pham, J., Kyauk, E., & Park, E. (2015). Predicting Song Popularity](https://cs229.stanford.edu/proj2015/140_report.pdf)
