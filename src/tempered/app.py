import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="AI TraceFinder",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA", "🧠 CNN Model", "📈 Model Performance", "🔍 Prediction", "ℹ️ About"]
)

# =====================================================
# 🏠 HOME PAGE
# =====================================================
# =====================================================
# 🏠 HOME PAGE
# =====================================================
# 🏠 HOME PAGE
# =====================================================
if menu == "🏠 Home":
    import os
    st.title("🧠 AI TraceFinder – Digital Scanner Forensics")

    st.markdown("""
    **AI TraceFinder** is a powerful forensic tool designed to **identify the source scanner**
    of a scanned document and detect **tampering or manipulation** using AI-driven
    noise and pattern analysis.
    """)

    # --- Robust Image Loader ---
    possible_paths = [
        "image.jpg",
        "image.png",
        os.path.join("app", "image.jpg"),
        os.path.join("app", "image.png"),
        os.path.join("assets", "image.jpg"),
        os.path.join("assets", "image.png"),
        os.path.join(os.path.dirname(__file__), "image.jpg"),
        os.path.join(os.path.dirname(__file__), "image.png"),
    ]

    app_image = None
    for path in possible_paths:
        if os.path.exists(path):
            app_image = path
            break

    if app_image:
        st.image(app_image, caption="🖼️ AI TraceFinder Application Interface", use_container_width=True)
    else:
        st.warning(f"⚠️ App image not found. Please add your interface image as `image.jpg` or `image.png` in the project root folder.\n\nExpected paths:\n{possible_paths}")

    st.markdown("---")
    st.subheader("🎯 What It Does")
    st.markdown("""
    - Detects the **scanner device** used to produce scanned copies.  
    - Identifies **forged or tampered** documents.  
    - Helps in **forensic and legal investigations**.  
    - Assists **researchers and law enforcement** with AI-based scanner identification.
    """)

    st.markdown("---")
    st.subheader("💡 Real-World Applications")
    st.markdown("""
    - 🧾 **Document Verification** – Check authenticity of scanned documents.  
    - 🕵️‍♂️ **Forensic Analysis** – Identify scanners used in evidence production.  
    - 🧠 **AI Security Research** – Study scanner fingerprints and artifacts.  
    - 🧰 **Tampering Detection** – Detect local or global image manipulation.  
    """)

    st.markdown("---")
    st.info("➡️ Use the sidebar to explore **EDA**, **Model Performance**, and **Live Prediction** modules.")

# =====================================================
# 📊 EDA PAGE
# =====================================================
elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("Analyze extracted features from your enhanced dataset for insights and patterns.")

    path = "proceed_data/enhanced_features.pkl"

    if not os.path.exists(path):
        st.error("❌ File not found: `proceed_data/enhanced_features.pkl`")
    else:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Expect dict with 'features' and 'labels'
            if isinstance(data, dict) and "features" in data and "labels" in data:
                X = np.array(data["features"])
                y = np.array(data["labels"])
                df = pd.DataFrame(X)
                df["label"] = y

                st.subheader("🔍 Feature Overview")
                st.write(f"**Shape:** {df.shape[0]} samples × {df.shape[1]-1} features")
                st.dataframe(df.head(), use_container_width=True)

                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.subheader("🔥 Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
                    st.pyplot(fig)

                    st.subheader("📉 Feature Distributions")
                    num_cols = numeric_df.columns[:4]
                    cols = st.columns(2)
                    for i, col in enumerate(num_cols):
                        with cols[i % 2]:
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], bins=30, kde=True, ax=ax)
                            ax.set_title(f"Distribution of {col}")
                            st.pyplot(fig)
                else:
                    st.warning("⚠️ No numeric columns found for EDA visualization.")

                st.subheader("🏷️ Label Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x="label", data=df, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Optional banner image
                banner_path = "assets/tracefinder_banner.png"
                if os.path.exists(banner_path):
                    st.image(banner_path, caption="AI TraceFinder – Enhanced Feature Visualization", use_container_width=True)
                else:
                    st.info("🖼️ Add your banner image at: `assets/tracefinder_banner.png`")

            else:
                st.error("⚠️ Unsupported data format. Expected dictionary with 'features' and 'labels' keys.")
        except Exception as e:
            st.error(f"Error loading EDA data: {e}")

# ----------------------------
# PAGE: CNN MODEL
# ----------------------------
elif menu == "🧠 CNN Model":
    import pickle
    import matplotlib.pyplot as plt

    st.title("🧠 CNN Model Training Visualization")
    st.write("Visualize how the CNN model learned during training and validation over epochs.")

    # Possible locations for the training history file
    history_paths = [
        "hybrid_training_history.pkl",
        os.path.join("app", "hybrid_training_history.pkl"),
        os.path.join("saved_models", "training_history.pkl")
    ]

    history_file = next((p for p in history_paths if os.path.exists(p)), None)

    if history_file:
        try:
            with open(history_file, "rb") as f:
                history = pickle.load(f)

            if isinstance(history, dict):
                acc = history.get("accuracy", [])
                val_acc = history.get("val_accuracy", [])
                loss = history.get("loss", [])
                val_loss = history.get("val_loss", [])

                # --- Accuracy Plot ---
                st.subheader("📊 CNN Model Accuracy")
                fig_acc, ax1 = plt.subplots()
                ax1.plot(acc, label="Training Accuracy", marker='o')
                if val_acc:
                    ax1.plot(val_acc, label="Validation Accuracy", marker='x')
                ax1.set_xlabel("Epochs")
                ax1.set_ylabel("Accuracy")
                ax1.set_title("CNN Training vs Validation Accuracy")
                ax1.legend()
                st.pyplot(fig_acc)

                # --- Loss Plot ---
                st.subheader("📉 CNN Model Loss")
                fig_loss, ax2 = plt.subplots()
                ax2.plot(loss, label="Training Loss", marker='o', color='orange')
                if val_loss:
                    ax2.plot(val_loss, label="Validation Loss", marker='x', color='red')
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("Loss")
                ax2.set_title("CNN Training vs Validation Loss")
                ax2.legend()
                st.pyplot(fig_loss)

                st.success("✅ CNN training visualization loaded successfully!")

            else:
                st.warning("⚠️ Invalid file format. Expected a dictionary with keys like 'accuracy' and 'loss'.")
        except Exception as e:
            st.error(f"❌ Error loading training history: {e}")
    else:
        st.info("📂 Add your training history file (`hybrid_training_history.pkl` or `training_history.pkl`) to view CNN model performance graphs.")

# ===========================================================
# 📊 MODEL PERFORMANCE PAGE (FIXED)
# ===========================================================
# ------------------- Model Performance Section -------------------
elif menu == "Model Performance":
    st.title("📊 Model Performance Overview")

    try:
        MODEL_DIR = r"C:\Users\SATYAM\Desktop\AI image tracer\cnn_files"
        history_path = os.path.join(MODEL_DIR, "hybrid_training_history.pkl")

        if not os.path.exists(history_path):
            st.error("❌ Training history file not found.")
        else:
            with open(history_path, "rb") as f:
                history = pickle.load(f)

            acc = history.get("accuracy", [])
            val_acc = history.get("val_accuracy", [])
            loss = history.get("loss", [])
            val_loss = history.get("val_loss", [])
            epochs = range(1, len(acc) + 1)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Accuracy Curve")
                fig1, ax1 = plt.subplots()
                ax1.plot(epochs, acc, label="Training Accuracy", color="blue")
                ax1.plot(epochs, val_acc, label="Validation Accuracy", color="orange")
                ax1.set_xlabel("Epochs")
                ax1.set_ylabel("Accuracy")
                ax1.legend()
                st.pyplot(fig1, bbox_inches="tight")
                plt.close(fig1)

            with col2:
                st.subheader("Loss Curve")
                fig2, ax2 = plt.subplots()
                ax2.plot(epochs, loss, label="Training Loss", color="red")
                ax2.plot(epochs, val_loss, label="Validation Loss", color="green")
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("Loss")
                ax2.legend()
                st.pyplot(fig2, bbox_inches="tight")
                plt.close(fig2)

            st.success("✅ Model performance loaded successfully!")

    except Exception as e:
        st.error(f"🚨 Error displaying model performance: {e}")
        st.code(traceback.format_exc())

# =====================================================
# 🔍 PREDICTION PAGE
# =====================================================
elif menu == "🔍 Prediction":
    st.title("🔍 Live Prediction")
    st.write("Upload a scanned document to identify the **source scanner** and detect possible **tampering**.")

    import tempfile, os, pickle, cv2, pywt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    from skimage.feature import local_binary_pattern

    # -----------------------------
    # 📁 Correct Folder Paths
    # -----------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Corrected model paths (pointing to cnn_files folder)
    MODEL_DIR = os.path.join("cnn_files")
    SCN_MODEL_PATH = os.path.join(MODEL_DIR, "scanner_hybrid_final.keras")

    SCN_LE_PATH     = os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl")
    SCN_SCALER_PATH = os.path.join(MODEL_DIR, "hybrid_feat_scaler.pkl")
    SCN_FP_PATH     = os.path.join(MODEL_DIR, "scanner_fingerprints.pkl")
    SCN_FP_KEYS     = os.path.join(MODEL_DIR, "fp_keys.npy")

    IMG_SIZE = (256, 256)

    # -----------------------------
    # ⚙️ Load Resources
    # -----------------------------
    @st.cache_resource
    def load_tf_model(path):
        return tf.keras.models.load_model(path, compile=False)

    @st.cache_resource
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @st.cache_resource
    def load_numpy_list(path):
        return np.load(path, allow_pickle=True).tolist()

    # Load models safely
    try:
        hyb_model   = load_tf_model(SCN_MODEL_PATH)
        le_sc       = load_pickle(SCN_LE_PATH)
        sc_sc       = load_pickle(SCN_SCALER_PATH)
        scanner_fps = load_pickle(SCN_FP_PATH)
        fp_keys     = load_numpy_list(SCN_FP_KEYS)
        st.success("✅ Model and supporting files loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading scanner models: {e}")
        hyb_model = None

    # -----------------------------
    # 🧩 Feature Extraction Helpers
    # -----------------------------
    def preprocess_residual_pywt(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Cannot read input image.")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
        cH.fill(0); cV.fill(0); cD.fill(0)
        den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
        return (img - den).astype(np.float32)

    def corr2d(a, b):
        a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        bins = np.linspace(0, r.max() + 1e-6, K + 1)
        feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i + 1])])) for i in range(K)]
        return feats

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - np.min(img)) / (rng + 1e-8)
        codes = local_binary_pattern((g * 255).astype(np.uint8), P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P + 3), density=True)
        return hist.astype(np.float32).tolist()

    def make_scanner_feats_from_res(res):
        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
        v_fft = fft_radial_energy(res, 6)
        v_lbp = lbp_hist_safe(res, 8, 1.0)
        feat = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
        return sc_sc.transform(feat)

    # -----------------------------
    # 🔮 Prediction Function
    # -----------------------------
    def predict_scanner(image_path):
        res = preprocess_residual_pywt(image_path)
        x_img = np.expand_dims(res, axis=(0, -1))
        x_ft = make_scanner_feats_from_res(res)
        prob = hyb_model.predict([x_img, x_ft], verbose=0).ravel()
        idx = int(np.argmax(prob))
        label = le_sc.classes_[idx]
        conf = float(prob[idx] * 100.0)
        return label, conf

    # -----------------------------
    # 📤 Upload & Predict
    # -----------------------------
    uploaded = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1] or ".tif"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.close()

        try:
            if hyb_model is None:
                st.error("Model not loaded. Please check model paths.")
            else:
                label, conf = predict_scanner(tmp.name)
                img = Image.open(tmp.name).convert("RGB")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(img, caption=f"Predicted Scanner: {label}", use_container_width=True)
                with c2:
                    st.metric("Scanner Prediction Confidence", f"{conf:.2f}%")

                st.success(f"✅ **Predicted Scanner:** {label} ({conf:.2f}%)")

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

        finally:
            try:
                os.remove(tmp.name)
            except:
                pass

# =====================================================
# ℹ️ ABOUT PAGE
# =====================================================
elif menu == "ℹ️ About":
    import base64
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    st.title("ℹ️ About AI TraceFinder")
    st.markdown("---")

    st.markdown("""
    ## 🧠 Project Summary
    **AI TraceFinder** is an advanced forensic system designed to **identify the source scanner**
    of a scanned document and detect **tampering or forgery** using AI-based pattern analysis.

    It combines **Deep Learning (CNN)** and **Machine Learning (Random Forest & SVM)** 
    for hybrid prediction — extracting both **texture-level** and **feature-level** fingerprints 
    from scanned images.  
    """)

    st.markdown("---")
    st.subheader("💡 Key Features")
    st.markdown("""
    - 📄 Detects the **scanner device** used to produce scanned documents.  
    - 🧠 Identifies **forged or tampered** copies using residual noise patterns.  
    - 📊 Integrates **EDA, Model Performance, and Live Prediction** modules.  
    - ⚙️ Built with **AI, ML, and Forensic Imaging principles**.  
    """)

    st.markdown("---")
    st.subheader("🧰 Technologies Used")
    st.markdown("""
    | Component | Description |
    |------------|-------------|
    | **Streamlit** | For interactive UI and modular app design |
    | **TensorFlow / Keras** | CNN model training and prediction |
    | **Scikit-learn** | Random Forest & SVM model implementation |
    | **OpenCV + PyWavelets** | Image processing and residual analysis |
    | **NumPy / Pandas** | Data handling and preprocessing |
    | **Matplotlib / Seaborn** | Visualization and plotting |
    """)

    st.markdown("---")
    st.subheader("🌍 Project Repository")
    st.markdown("""
    The full project source code, trained models, and documentation are available on GitHub.
    """)

    # --- GitHub Button (UPDATED) ---
    github_url = "https://github.com/satyamsingh290/AI_Image_Tracer"
    st.markdown(f"""
        <a href="{github_url}" target="_blank">
            <button style="
                background-color:#24292e;
                color:white;
                border:none;
                border-radius:8px;
                padding:10px 20px;
                font-size:16px;
                cursor:pointer;">
                🔗 View on GitHub
            </button>
        </a>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ----------------------------
    # 📄 Downloadable PDF Report
    # ----------------------------
    st.subheader("📥 Download Project Report")

    def create_pdf_report():
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(180, height - 80, "AI TraceFinder - Project Report")

        pdf.setFont("Helvetica", 12)
        text = pdf.beginText(50, height - 120)
        text.textLines("""
AI TraceFinder is a hybrid forensic intelligence system that identifies
the source scanner of a document image and detects tampering using
noise, frequency, and texture-based artifacts.

----------------------------
🧠 Model Summary
----------------------------
- CNN Model (TensorFlow/Keras)
- Random Forest (Scikit-learn)
- SVM Model (Scikit-learn)
- Hybrid feature-based + deep feature analysis

----------------------------
💡 Key Highlights
----------------------------
- Noise residual extraction using DWT
- PRNU and correlation-based fingerprinting
- Texture feature extraction (LBP, FFT)
- Scanner classification and forgery detection

----------------------------
🧰 Technologies Used
----------------------------
Streamlit · TensorFlow · Scikit-learn · OpenCV · PyWavelets · Matplotlib

----------------------------
👨‍💻 Developer
----------------------------
Satyam Singh
CSE Student | AI-ML Enthusiast
GitHub: https://github.com/satyamsingh290/AI_Image_Tracer
""")
        pdf.drawText(text)
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = create_pdf_report()
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="AI_TraceFinder_Report.pdf"><button style="background-color:#0078D7;color:white;padding:10px 20px;border:none;border-radius:8px;cursor:pointer;">⬇️ Download PDF Report</button></a>'
    st.markdown(href, unsafe_allow_html=True)

    st.markdown("---")
    st.success("✅ About section loaded successfully with GitHub link and downloadable report!")
