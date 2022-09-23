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
    if file.endswith(".xlsx"):
        df = pd.read_excel(data_dir + file, header=None)
    elif file.endswith(".csv"):
        df = pd.read_csv(data_dir + file, engine='python', header=None)
    dfs.append(df)

df = pd.concat(dfs, axis=0)
# First col is label, second col is text
df.columns = ['label', 'comment']

# Remove missing values, duplicate values and abnormal samples
df = df.dropna()
df = df.drop_duplicates()
df = df[df['label'].isin([0.0, '-1', -1.0, 1.0, '0', '1'])]
df['label'] = df['label'].astype('int')
print(f"Number of labels in the training set:\n {df['label'].value_counts()}")
#if there are label 0 change it to 1
df['label'] = df['label'].replace(0, 1)
print(f"The labels after combination:\n{df['label'].value_counts()}")
df


##########################   Data Expolring   ################################################
# header defaultly presents 5 lines
df.head()
print(df.head())

# View the distribution of samples
count = len(df['label'])
count_negative_one = df.loc[df['label'] == -1]
number_of_negative_ones = len(count_negative_one)
count_one = df.loc[df['label'] == 1]
number_of_ones = len(count_one)

print(' Total amount of examples:', count, "\n", 'Total number of label=1: ',
      number_of_ones, "\n", "Total number of label=-1: ", number_of_negative_ones)

##########################   Visualize the distribution   ####################################
import seaborn as sns
sns.countplot(x='label', data=df).set_title('label distribution')

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
numbers =  [number_of_ones, number_of_negative_ones]
explode = None
labels= ['label=1', 'label=-1']
plt.pie(numbers, explode =explode, labels = labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('label distribution')
#show the diagram
#plt.show()

##########################   Text data cleaning   ###############################################
from nltk.corpus import stopwords
import string
import re
from nltk import *

# Load stopwords
stop_words = stopwords.words('english')
# Add some other words
#stop_words.append("https")

porter = PorterStemmer()
def remove_noise(text):
    # Participate
    tokens = word_tokenize(text)
    # stemmer
    tokens =[porter.stem(t) for t in tokens]
    # Filter abnoarmal words
    tokens_filtered = [w for w in tokens if w.lower() not in stop_words and w.lower() not in string.punctuation]
    return " ".join(tokens_filtered)

df['comment_pre'] = df['comment'].apply(remove_noise)
df.head()
print(df.head())

##########################   Construct Training and Testing sets   ###############################################
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X, y = df['comment_pre'].to_numpy(), df['label'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f'X_train size:{X_train.shape[0]}, X_test size:{X_test.shape[0]}')

# manually construct text features, construct Tfidf statistical features
vect = TfidfVectorizer(ngram_range=(2, 2), max_features=5000)

X_train_Tfidf = vect.fit_transform(X_train)
X_test_Tfidf = vect.transform(X_test)
X_train_Tfidf.shape

#################################   SVM MODEL   ################################################################
print("#####   SVM MODEL   #######")
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

# Create a svm Classifier
clf = svm.SVC(kernel='rbf')  # rbf Kernel
# Train the model using the training sets
clf.fit(X_train_Tfidf, y_train)
# Predict the class for train dataset
predict_train = clf.predict(X_train_Tfidf)

print('---------------------Training Set--------------------')
print(f"Confusion matrix:\n {confusion_matrix(y_train, predict_train)}")
print(f"Indicator statistics:\n {classification_report(y_train, predict_train)}")
print()
print('---------------------Testing Set---------------------')
# Test the performance on testing set
predict_test = clf.predict(X_test_Tfidf)
print(f"Confusion matrix:\n {confusion_matrix(y_test, predict_test)}")
print(f"Indicator statistics:\n {classification_report(y_test, predict_test)}")


#################################   Naive Bayes MODEL   ############################################################
print("#####   Naive Bayes MODEL   #######")
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

# Create a naive_bayes Classifier
clf = MultinomialNB()
# Train the model using the training sets
clf.fit(X_train_Tfidf.toarray(), y_train)
# Predict the class for train dataset
predict_train = clf.predict(X_train_Tfidf.toarray())

print('---------------------Training Set--------------------')
print(f"Confusion matrix:\n {confusion_matrix(y_train, predict_train)}")
print(f"Indicator statistics:\n {classification_report(y_train, predict_train)}")
print()
print('---------------------Testing Set---------------------')
# Test the performance on testing set
predict_test = clf.predict(X_test_Tfidf)
print(f"Confusion matrix:\n {confusion_matrix(y_test, predict_test)}")
print(f"Indicator statistics:\n {classification_report(y_test, predict_test)}")


#################################   Single-layer CNN MODEL   ############################################################
print("#####   Single-layer CNN MODEL   ######")
import jieba
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

text = df['comment_pre'].tolist()
labels = df['label'].tolist()

MAX_FEATURES = 5000 #Max dict
MAX_DOCUMENT_LENGTH = 20 #max doc length

# Convert text to number
tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
X = pad_sequences(sequences, maxlen=MAX_DOCUMENT_LENGTH)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print("X_train:", X_train)
print("y_train:", y_train)

from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Input, MaxPooling1D, Flatten, concatenate, Embedding, Dropout

BATCH_SIZE = 128
NUM_CLASSES = 3
EPOCH = 15
# CNN parameters
embedding_dims = 20
filters = 20
kernel_size = 3
hidden_dims = 250

model = Sequential()
# word embedding layer
model.add(Embedding(MAX_FEATURES, embedding_dims))

# Convolutional layer
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# pooling layer
model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))

# fully connected layer
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCH,
          validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test,batch_size=BATCH_SIZE)
print("loss: {}, accuracy:{}".format(loss, accuracy))

# plot loss and accuracy with epoch
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
def plot_learning_curves(history):
    pd.DataFrame(history.history, index=range(1,EPOCH+1)).plot(figsize=(10, 6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

#################################   TextCNN MODEL   ############################################################
print("###   TextCNN MODEL   #######")
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Input, MaxPooling1D, Flatten, concatenate, Embedding, Dropout
from keras import regularizers

BATCH_SIZE = 128
NUM_CLASSES = 3
EPOCH = 15

# CNN parameters
embedding_dims = 50
filters = 10

# Inputs
input = Input(shape=[MAX_DOCUMENT_LENGTH])

# Embeddings layers
x = Embedding(MAX_FEATURES, embedding_dims)(input)

# conv layers
convs = []
for filter_size in [3, 4, 5]:
    l_conv = Conv1D(filters=filters, kernel_size=filter_size, activation='relu')(x)
    l_pool = MaxPooling1D()(l_conv)
    l_pool = Flatten()(l_pool)
    convs.append(l_pool)

merge = concatenate(convs, axis=1)

out = Dropout(0.5)(merge)

out = Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01))(out)

output = Dense(units=NUM_CLASSES, activation='softmax')(out)

# Output Layer
model = Model([input], output)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCH,
          validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test,batch_size=BATCH_SIZE)
print("loss: {}, accuracy:{}".format(loss, accuracy))

# plot loss and accuracy with epoch

import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
def plot_learning_curves(history):
    pd.DataFrame(history.history, index=range(1,EPOCH+1)).plot(figsize=(10, 6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)


