import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from joblib import dump

df = pd.read_csv('dataset/dataset.csv')

def train_model(df: pd.DataFrame):
    df = df.fillna("")
    df['Symptom'] = ""

    for i in range(1,18):
        df['s'] = df["Symptom_{}".format(i)]
        df['Symptom'] = df['Symptom'] + df['s']

    for i in range(1,18):
        df = df.drop("Symptom_{}".format(i),axis=1)

    df = df.drop("s",axis=1)

    X = df['Symptom']
    y = df['Disease']

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.25, shuffle=True, random_state=44)

    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)


    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)

    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC()),])

    text_clf.fit(X_train, y_train)

    predictions = text_clf.predict(X_test)

    print(metrics.confusion_matrix(y_test,predictions))
    print(metrics.classification_report(y_test,predictions))
    print("Accuracy score: {}".format(metrics.accuracy_score(y_test,predictions)))

    dump(text_clf, 'models/disease_prediction_model.joblib')

    print("Model saved to 'disease_prediction_model.joblib'")

train_model(df)
