import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from vncorenlp import VnCoreNLP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import linear_model
import pickle 
import streamlit as st

vn_stopwords = []
with open('data/vietnamese_stopwords.txt', encoding="utf8") as file:
    for line in file.read().splitlines():
        vn_stopwords.append(line.strip())

X_train_vect = pickle.load(open('data/X_train_vect.sav','rb'))
y_train = pickle.load(open('data/y_train.sav','rb'))

vectorization = TfidfVectorizer()
X_train = vectorization.fit_transform(X_train_vect)

def preprocessing(X_df):
    for index, test_str in enumerate(X_df):
        # Remove http links
        filtered_str = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", test_str[0])
        # Remove all number in string
        filtered_str = ''.join([i for i in filtered_str if not i.isdigit()])
        # Remove all special characters and punctuation
        filtered_str = re.sub('\W+',' ', filtered_str)
        filtered_str = filtered_str.strip()
        X_df[index][0] = filtered_str.lower()

    X_df = X_df.flatten()

    # Remove stop words
    for index, test_str in enumerate(X_df):
        sw_removed = [w for w in test_str.split(' ') if not w in vn_stopwords]
        X_df[index] = ' '.join(sw_removed)

    return X_df

def tokennize_text(X_df):
    '''
    Tokenize the sentences into words
    return a numpy array of preprocessed text, each item in array is a paragraph after remove stopwords, special characters, ...
    '''
    annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
    #annotator = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar")
    with annotator:
        for index, filtered_text in enumerate(X_df):
            token_text = annotator.tokenize(filtered_text)[0]
            X_df[index] = ' '.join(token_text)
    
    return X_df

def create_DTM(X_df):
    cv = CountVectorizer(analyzer='word')
    data = cv.fit_transform(X_df).todense()
    return data

def hash_data(X_df):
    vectorization = TfidfVectorizer()
    return vectorization.fit(X_df)

#def vectorize_tfidf(X_df):
#    X = hash_data(X_df)
#    return X.transform(X_df)

def vectorize_tfidf(X_df):
    vectorization = TfidfVectorizer()
    return vectorization.fit_transform(X_df).todense()

#hash_data_learn = hash_data(X_df)

#xv_train = vectorize_tfidf(X_df)
#X_train, X_test, y_train, y_test = train_test_split(xv_train, Y_target, test_size=0.2, random_state=0)

def prediction(model, news):
    test_dict = {"text": [news]}
    test_df = pd.DataFrame(test_dict)

    text = test_df.iloc[:,:].values

    processed_text = tokennize_text(preprocessing(text))
    vector_text = vectorization.transform(processed_text)

    if model == 'Logistic Regression':
        LR = LogisticRegression()
        LR.fit(X_train, y_train)
        return LR.predict(vector_text)

    elif model == 'Decision Tree':
        dtc = DecisionTreeClassifier(criterion='entropy')
        dtc.fit(X_train, y_train)
        return dtc.predict(vector_text)

    elif model == 'Gradient Booting Classifier':
        GBC = GradientBoostingClassifier(random_state=0)
        GBC.fit(X_train, y_train)
        return GBC.predict(vector_text)

    elif model == 'Linear Regession':
        REG = linear_model.LinearRegression()
        REG.fit(X_train, y_train)
        y_pred = REG.predict(vector_text)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        return y_pred

    elif model == 'Nearest Centroid Classifier':
        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        return clf.predict(vector_text)

authors='''
19120525 - Lê Minh Hữu
19120533 - Ninh Duy Huy
19120544 - Cao Thanh Khiết
19120547 - Nguyễn Tuấn Khoa
19120557 - Trần Tuấn Kiệt'''

def main():
    st.title('VIETNAMESE FAKE NEWS DETECTION')
    st.text(authors)

    news = st.text_input('Write your news here')

    model = st.selectbox('Select your model',options=['Logistic Regression','Decision Tree',
                                                     'Gradient Booting Classifier','Linear Regession',
                                                     'Nearest Centroid Classifier'])

    if st.button('Detect'):
        predict = prediction(model, news)
        if predict[0] == 0:
            st.success("This is fake news.")
        else:
            st.success("This is not fake news.")


if __name__ == '__main__':
    main()