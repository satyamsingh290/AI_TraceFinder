import os, cv2

BASE_DIR = r"C:\Users\SATYAM\Desktop\AI image tracer\proceed_data\Official"

for scanner in os.listdir(BASE_DIR):
    scanner_dir = os.path.join(BASE_DIR, scanner)
    if not os.path.isdir(scanner_dir):
        continue
    for dpi in os.listdir(scanner_dir):
        dpi_dir = os.path.join(scanner_dir, dpi)
        if not os.path.isdir(dpi_dir):
            continue
        files = [f for f in os.listdir(dpi_dir) if f.lower().endswith((".tif",".tiff",".png",".jpg",".jpeg"))]
        print(scanner, dpi, len(files))
        for f in files[:3]:  # check first 3 images
            img = cv2.imread(os.path.join(dpi_dir, f), cv2.IMREAD_UNCHANGED)
            print("  ", f, "->", None if img is None else img.shape)
