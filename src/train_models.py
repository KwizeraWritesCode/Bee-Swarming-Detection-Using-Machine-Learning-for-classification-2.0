import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB


def train_classifier(model, X_train, y_train, X_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    predictions = model.predict(X_test)
    test_time = time.time() - start

    return predictions, train_time, test_time


def get_models():
    return {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gaussian NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB()
    }