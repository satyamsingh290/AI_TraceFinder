import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# =============================
# ✅ PATH CONFIG
# =============================
CSV_PATH = r"C:\Users\SATYAM\Desktop\AI image tracer\AI_TraceFinder\processed_data\metadata_features.csv"
MODEL_DIR = r"C:\Users\SATYAM\Desktop\AI image tracer\AI_TraceFinder\models"

# =============================
# ✅ TRAINING FUNCTION
# =============================
def train_models():
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print(f"📦 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

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

    # ==========================
    # 🌲 Random Forest
    # ==========================
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"✅ Random Forest Accuracy: {rf_acc:.3f}")
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))

    # ==========================
    # ⚡ Fast Linear SVM (with prob.)
    # ==========================
    print("\n⚡ Training Linear SVM (Calibrated for probabilities)...")
    base_svm = LinearSVC(max_iter=10000)
    svm = CalibratedClassifierCV(base_svm)
    svm.fit(X_train, y_train)
    svm_acc = svm.score(X_test, y_test)
    print(f"✅ SVM Accuracy: {svm_acc:.3f}")
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.pkl"))

    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("\n🎯 Training complete!")
    print(f"📁 Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    train_models()
