"""
processing.py
-------------
Preprocessing pipeline for Flatfield, Official, and Wikipedia datasets.
Generates residual images, fingerprints (feature vectors), and labels.
Outputs:
- official_wiki_residuals.pkl
- flatfield_residuals.pkl
- official_wiki_fingerprints.pkl
- official_wiki_labels.pkl
"""

import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pywt
from skimage.restoration import denoise_wavelet
from scipy.signal import wiener as scipy_wiener
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 1) Global Parameters
# ---------------------------
IMG_SIZE = (256, 256)
DENOISE_METHOD = "wavelet"  # or "wiener"
MAX_WORKERS = 8
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS

BASE_DIR = r"C:\Users\SATYAM\Desktop\AI image tracer\proceed_data"

# ---------------------------
# 2) Helper Functions
# ---------------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def denoise_wavelet_img(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

def preprocess_image(fpath, method=DENOISE_METHOD):
    """Read, gray, resize, normalize, denoise, compute residual."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_gray(img)
    img = resize_to(img)
    img = normalize_img(img)
    if method == "wiener":
        den = scipy_wiener(img, mysize=(5,5))
    elif method == "wavelet":
        den = denoise_wavelet_img(img)
    else:
        raise ValueError(f"Unknown denoise method: {method}")
    residual = (img - den).astype(np.float32)
    return residual

def extract_fingerprint(residual):
    """Extract simple texture-based fingerprint using Local Binary Pattern."""
    lbp = local_binary_pattern(residual, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def parallel_process_images(file_list):
    """Process images in parallel and return list of residuals."""
    residuals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(preprocess_image, f) for f in file_list]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                residuals.append(res)
    return residuals

def process_folder(base_dir, use_dpi_subfolders=True):
    """
    Process folder to compute residuals grouped by scanner/dpi.
    Returns: dict[scanner][dpi] = list_of_residuals
    """
    residuals_dict = {}
    scanners = sorted(os.listdir(base_dir))

    for scanner in tqdm(scanners, desc=f"Processing scanners in {os.path.basename(base_dir)}"):
        scanner_dir = os.path.join(base_dir, scanner)
        if not os.path.isdir(scanner_dir):
            continue

        residuals_dict[scanner] = {}

        if use_dpi_subfolders:
            for dpi in os.listdir(scanner_dir):
                dpi_dir = os.path.join(scanner_dir, dpi)
                if not os.path.isdir(dpi_dir):
                    continue
                files = []
                for root, _, fs in os.walk(dpi_dir):
                    for f in fs:
                        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                            files.append(os.path.join(root, f))
                residuals_dict[scanner][dpi] = parallel_process_images(files)
        else:
            files = []
            for root, _, fs in os.walk(scanner_dir):
                for f in fs:
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                        files.append(os.path.join(root, f))
            residuals_dict[scanner] = parallel_process_images(files)

    return residuals_dict

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved: {path}")

# ---------------------------
# 3) Main Execution
# ---------------------------
if __name__ == "__main__":
    # ----- Official + Wikipedia -----
    datasets = ["Official", "Wikipedia"]
    official_wiki_residuals = {}

    all_fingerprints = []
    all_labels = []

    for dataset in datasets:
        print(f"\n🔄 Processing {dataset} dataset...")
        dataset_dir = os.path.join(BASE_DIR, dataset)
        residuals = process_folder(dataset_dir, use_dpi_subfolders=True)
        official_wiki_residuals[dataset] = residuals

        # Extract fingerprints and labels
        for scanner, dpi_dict in residuals.items():
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    fp = extract_fingerprint(res)
                    all_fingerprints.append(fp)
                    all_labels.append(scanner)  # label by scanner

    # Save Official + Wikipedia residuals
    official_wiki_residuals_path = os.path.join(BASE_DIR, "official_wiki_residuals.pkl")
    save_pickle(official_wiki_residuals, official_wiki_residuals_path)

    # Encode labels numerically
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(all_labels)

    # Save fingerprints & labels
    fingerprints_path = os.path.join(BASE_DIR, "official_wiki_fingerprints.pkl")
    labels_path = os.path.join(BASE_DIR, "official_wiki_labels.pkl")
    save_pickle(np.array(all_fingerprints, dtype=np.float32), fingerprints_path)
    save_pickle(y_encoded, labels_path)
    print(f"✅ Generated fingerprints: {len(all_fingerprints)} samples, {len(set(all_labels))} unique scanners.")

    # ----- Flatfield -----
    print("\n🔄 Processing Flatfield dataset...")
    flatfield_dir = os.path.join(BASE_DIR, "Flatfield")
    flatfield_residuals = process_folder(flatfield_dir, use_dpi_subfolders=False)

    flatfield_out_path = os.path.join(BASE_DIR, "flatfield_residuals.pkl")
    save_pickle(flatfield_residuals, flatfield_out_path)

    print("\n🎯 Preprocessing complete!")
    print(f"Files generated in: {BASE_DIR}")
    print("""
    - official_wiki_residuals.pkl
    - flatfield_residuals.pkl
    - official_wiki_fingerprints.pkl
    - official_wiki_labels.pkl
    """)
