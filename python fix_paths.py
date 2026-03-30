import pickle
from pathlib import Path

# load paths lama dari Colab
with open("models/image_paths.pkl", "rb") as f:
    old_paths = pickle.load(f)

# ganti prefix Colab dengan prefix lokal
LOCAL_IMAGES_DIR = "data/images"

new_paths = []
for p in old_paths:
    if p is None:
        new_paths.append(None)
    elif p.startswith("WIFI_PRODUCT::"):
        # path WiFi tidak perlu diubah
        new_paths.append(p)
    else:
        # ambil hanya nama filenya, sambungkan ke path lokal
        filename = Path(p).name
        new_paths.append(str(Path(LOCAL_IMAGES_DIR) / filename))

# simpan paths baru
with open("models/image_paths.pkl", "wb") as f:
    pickle.dump(new_paths, f)

print(f"Total paths   : {len(new_paths)}")
print(f"Contoh path baru : {new_paths[0]}")