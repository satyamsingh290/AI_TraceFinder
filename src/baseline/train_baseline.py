import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

CSV_PATH = "processed_data/metadata_features.csv"
MODEL_DIR = "models"


def train_models():
    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Features and labels
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")

    # Train SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    joblib.dump(svm, f"{MODEL_DIR}/svm.pkl")

    # Save scaler
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("✅ Models trained and saved successfully!")


if __name__ == "__main__":
    train_models()
