from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    return acc, report, cm