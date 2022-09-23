#CIS 600 datamining
#team 12

##########################   Data loading   ############################################
import os
import pandas as pd
print(os.getcwd())
data_dir = 'C:/Users/jinbo/Desktop/fishc/venv/'  # note to modify directory if changes
files = os.listdir(data_dir)
print(f'data files: {files}')
dfs = []
for file in files:
    try:
        if file.endswith(".xlsx"):
            df = pd.read_excel(data_dir + file, header=None)
        elif file.endswith(".csv"):
            df = pd.read_csv(data_dir + file, header=None)
        dfs.append(df)
    except Exception as e:
        print(file, e)


df = pd.concat(dfs, axis=0)
# First col is label, second col is text
df.columns = ['label', 'comment']

# Remove missing values, duplicate values and abnormal samples
df = df.dropna()
df = df.drop_duplicates()
df = df[df['label'].isin([0.0, '-1', -1.0, 1.0, '0', '1'])]
df['label'] = df['label'].astype('int')
df['comment'] = df['comment'].astype('str')
print(f"Number of labels in the training set:\n {df['label'].value_counts()}")
#if there are label 0 change it to 1
df['label'] = df['label'].replace(0, 1)
print(f"The labels after combination:\n{df['label'].value_counts()}")
df


##########################   Text Data cleaning   ############################################
from nltk.corpus import stopwords
import string
import re
from nltk import *

# load stopwords
stop_words = stopwords.words('english')
stop_words.append("https")
stop_words.append('\'s')

porter = PorterStemmer()
def remove_noise(text):
    tokens = word_tokenize(text) # participle
    tokens_filtered = [w for w in tokens if w.lower() not in stop_words and w.lower() not in string.punctuation] # filter abnormal words
    return tokens_filtered


df['comment_pre'] = df['comment'].apply(remove_noise)
df.head()
print(df.head())

##########################   Negative Word Cloud   ################################################
from functools import reduce
df_negetive = df[df['label']==-1]
df_negetive.head()
texts= df_negetive['comment_pre'].tolist()

word_list = reduce(lambda x,y:x+y,texts)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

from collections import Counter
word_frequence = dict(Counter(word_list).most_common(1000))
word_frequence
wordcloud=WordCloud(background_color="white",max_font_size=80)
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()
##########################   Positive Word Cloud   ################################################
from functools import reduce
df_positive = df[df['label']==1]
print(df_positive.head())
texts= df_positive['comment_pre'].tolist()

word_list = reduce(lambda x,y:x+y,texts)
word_list[0:50]

from wordcloud import WordCloud
import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

from collections import Counter
word_frequence = dict(Counter(word_list).most_common(1000))
word_frequence
wordcloud=WordCloud(background_color="white",max_font_size=80)
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()