from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, clean_data, split_features_labels, encode_labels
from src.train_models import get_models, train_classifier
from src.evaluate import evaluate_model

# Load
df = load_data("data/processed/features.csv")
df = clean_data(df)

# Split
X, y = split_features_labels(df)
y_encoded, encoder = encode_labels(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train + Evaluate
models = get_models()

for name, model in models.items():
    preds, train_time, test_time = train_classifier(model, X_train, y_train, X_test)
    acc, report, cm = evaluate_model(y_test, preds)

    print(f"\n{name}")
    print(f"Accuracy: {acc}")
    print(report)