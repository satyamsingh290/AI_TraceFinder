import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

# ==============================
# ✅ PATH CONFIGURATION
# ==============================
BASE_DIR = r"C:\Users\SATYAM\Desktop\AI image tracer"
PROCEED_DATA = os.path.join(BASE_DIR, "proceed_data")

RES_PATH = os.path.join(PROCEED_DATA, "official_wiki_residuals.pkl")
FP_PATH = os.path.join(PROCEED_DATA, "official_wiki_fingerprints.pkl")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "hybrid_cnn_model.keras")
ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
HISTORY_SAVE_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

# ==============================
# ✅ VERIFY FILES EXIST
# ==============================
required_files = [RES_PATH, FP_PATH]
for p in required_files:
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ Missing file: {p}\nPlease make sure it exists before training.")

print("✅ All input files found — starting training...\n")

# ==============================
# ✅ LOAD DATA
# ==============================
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

with open(FP_PATH, "rb") as f:
    fingerprints_data = pickle.load(f)

print(f"✅ Data loaded successfully! Type of fingerprints_data: {type(fingerprints_data)}\n")

# ==============================
# ✅ CONVERT TO ARRAYS
# ==============================
X_res = []
y_res = []

for dataset_name, scanners in residuals_dict.items():
    for scanner_name, dpi_dict in scanners.items():
        for dpi, residual_list in dpi_dict.items():
            for residual in residual_list:
                feat_flat = np.array(residual).flatten()
                X_res.append(feat_flat)
                y_res.append(scanner_name)

X_res = np.array(X_res, dtype=np.float32)
print(f"✅ Residual array shape: {X_res.shape}")

# --- Fingerprint conversion ---
X_fp = []

# Handle both dictionary and numpy array cases
if isinstance(fingerprints_data, dict):
    for key, value in fingerprints_data.items():
        fp = np.array(value).flatten()
        X_fp.append(fp)
elif isinstance(fingerprints_data, (list, np.ndarray)):
    for fp in fingerprints_data:
        X_fp.append(np.array(fp).flatten())
else:
    raise TypeError(f"❌ Unsupported fingerprints data type: {type(fingerprints_data)}")

X_fp = np.array(X_fp, dtype=np.float32)
print(f"✅ Fingerprint array shape: {X_fp.shape}")

# --- Ensure equal lengths ---
min_len = min(len(X_res), len(X_fp))
X_res = X_res[:min_len]
X_fp = X_fp[:min_len]
y_res = y_res[:min_len]

# --- Combine features ---
X_combined = np.concatenate((X_res, X_fp), axis=1)
print(f"✅ Combined feature shape: {X_combined.shape}\n")

# ==============================
# ✅ LABEL ENCODING + SCALING
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_res)
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# ==============================
# ✅ TRAIN/TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)
print(f"✅ Train set: {X_train.shape}, Test set: {X_test.shape}\n")

# ==============================
# ✅ MODEL ARCHITECTURE
# ==============================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================
# ✅ TRAIN MODEL
# ==============================
print("\n🚀 Training started...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    verbose=1
)
import pickle
import os

# =============================================================
# ✅ SAVE TRAINED CNN MODEL
# =============================================================
save_dir = "C:/Users/SATYAM/Desktop/AI image tracer/src/cnn_model/saved_models"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "fast_hybrid_final.keras")
model.save(model_path)
print(f"✅ Model saved successfully at: {model_path}")

# =============================================================
# ✅ SAVE LABEL ENCODER
# =============================================================
try:
    with open(os.path.join(save_dir, "hybrid_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print("✅ Label encoder saved successfully.")
except Exception as e:
    print(f"⚠️ Label encoder not found: {e}")

# =============================================================
# ✅ SAVE FEATURE SCALER (optional, if you used one)
# =============================================================
try:
    with open(os.path.join(save_dir, "hybrid_feat_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Feature scaler saved successfully.")
except Exception as e:
    print(f"⚠️ Feature scaler not found: {e}")

# =============================================================
# ✅ SAVE TRAINING HISTORY (optional)
# =============================================================
try:
    with open(os.path.join(save_dir, "hybrid_training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)
    print("✅ Training history saved successfully.")
except Exception as e:
    print(f"⚠️ Training history not found: {e}")

# ==============================
# ✅ SAVE ALL ARTIFACTS
# ==============================
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_SAVE_PATH)
with open(ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(encoder, f)
with open(SCALER_SAVE_PATH, "wb") as f:
    pickle.dump(scaler, f)
with open(HISTORY_SAVE_PATH, "wb") as f:
    pickle.dump(history.history, f)

print("\n🎯 Training complete and all artifacts saved successfully!")
print(f"✅ Model: {MODEL_SAVE_PATH}")
print(f"✅ Encoder: {ENCODER_SAVE_PATH}")
print(f"✅ Scaler: {SCALER_SAVE_PATH}")
print(f"✅ History: {HISTORY_SAVE_PATH}")
