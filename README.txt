CIS 600 Datamining
Team 12
Prediction of the economic trend from Twitter

General information:
Our goal is using machine learning method classify the attitude of a tweet.
Collect data using Twitter API (Streaming) on the trend of economic trend.
Loading streaming data with label tweets in positive and negative.
Creating word clouds on positive and negative tweets.
Constructing training and testing sets, using SVM, Naive Bayes, Singular-layer CNN and TextCNN to train the data.

Technologies Used:

Programming tool:
Python

API:
Twitter API

Libraries:
OS
pandas
seaborn
matplotlib
nltk
re
sklearn
jieba
keras
wordcloud
collections
functools

Setup:
1.Install Python in your computer
2.Import all the Python packages above correctly
3.Download the datafiles and put them in the same fold with Python file. When running the code, the excel files will be read

Run:
There are three Python files.
1.Mediadata.py is used for crawing data from tweets applied streaming API. Before running it, please use correct Keys. 
2.data_process.py is data processing file. This is the main file to handle the data using machine learning algorithms. When applying this file, please first download the data files
and saved in the same folder with data_process.py file. 
3.data_cloud.py is a file which can be used to visualized the word frequency. 


Features:
Analysis on data
Visiual the frequency WordCloud
Machine Learning models - SVM, Naive-Bayes, Single-layerCNN, TextCNN

Room for improvement:
There are still some limitations need to be done in the future:
Crawing more data to improve the accuracy on the machine learning models.
Try more advanced Machine Learning models to get more precise results.
Be more careful to classify the positive tweets and negative tweets. 

