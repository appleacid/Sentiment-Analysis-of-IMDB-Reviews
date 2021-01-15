# Sentiment-Analysis-of-IMDB-Reviews
Sentiment Analysis of IMDB Reviews

Description:
This project is to conduct a sentiment analysis for the online imdb review data in the text format. 

Dataset:
It contains 25000 reviews that are labeled as positive and 25000 reviews that are labeled as negative. The dataset describe the online movie review forum imdb and use a criteria of any star review that has 7 or more stars is marked as positive and any reviews that receive 4 or less than 4 reviews are marked as negative. Those are reviewed are polarized reviews, this project is not dealing with neutral reviews. The dataset is downloaded from Stanford AI lab website: https://ai.stanford.edu/~amaas/data/sentiment/

Objective:
The goal of the project is to build a supervised learning classifer that is able to predict the sentiment (positive or negative) from the text-based review contents. In order to carry out this goal, the task is split into two stages. Stage 1 is to clean, and preprocess the data. Stage 2 is to use various basic classification and deep learning models to build a classifier which can predict the binary results. 

Pipeline and steps:
In stage 1, I used text preprocessing methods to remove the non-alphabetical words, html tags, and non-text characters from the reviews. Also, I removed the english stopwords. Following that, I used the tf-idf method to tokenize the texts and obtain the feature vector matrix, and transform the text into vectors. After cleaning and preprocessing the data, I conducted exloratory data analysis to view the top10 most frequencly appreared words in postive and negative review datasets, respectively. Then, moving on to stage 2, I first split the train and test data sets into 0.5:0.5 ratio, and conducted my baseline model, the logistic regression model. Since the dataset is balanced, the logistic regression model is able to achieve an accuracy score at 88.98%. I then tested different basic classification model, including SVM, Naive Bayes, and Random Forest. I also used a RNN-LSTM deep learning model to train and test the datasets. During stage 2, I used adjusting the epochs number, adding layers into lstm model to improve the model performance score. 

Modeling results:
In the end, the basic classic models and the rnn-lstm model were able to give an average accuracy sore of over 85%. All the results are reported in the format of confusion matrix and model report. To summarize, the accuracy scores ranging from high to low is as following, with the classifcation model it applies: Logistic Regression (0.8898), SVM(0.8874), Random Forest(0.8684), Bayes(0.8668), RNN-LSTM(0.8273). 
