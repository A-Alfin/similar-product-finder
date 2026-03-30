# Similar Product Finder
**Visual Similarity Search · ResNet50 + FAISS**

Sistem image similarity search untuk produk e-commerce. Upload foto produk apapun — sistem akan menampilkan produk paling mirip dari database menggunakan deep feature extraction.

---

## Demo

> Upload foto produk WiFi merek apapun → sistem merekomendasikan produk HIFI di hasil teratas.

---

## Highlights

- **No fine-tuning required** — ResNet50 pretrained ImageNet digunakan langsung sebagai feature extractor. Fitur visual seperti tekstur, warna, dan bentuk sudah cukup untuk similarity search tanpa training tambahan.
- **Custom product injection** — Foto produk HIFI ditambahkan ke FAISS index dengan teknik augmentasi (flip, rotasi, brightness, contrast, crop) untuk memperkuat representasi di feature space.
- **Production-realistic** — Pipeline inferensi konsisten antara Colab (eksperimen) dan lokal (deployment): preprocessing identik, L2 normalization, FAISS exact search.

---

## Architecture

```
Upload Image
     │
     ▼
ResNet50 pretrained (ImageNet)
  └─ FC layer dihapus → output avgpool = 2048-dim feature vector
     │
     ▼
L2 Normalization
  └─ cosine similarity = dot product → efisien di FAISS
     │
     ▼
FAISS IndexFlatIP
  └─ exact inner product search
     │
     ▼
Top-K Similar Products
```

---

## Dataset

**Fashion Product Images (Small)** — Kaggle (paramaggarwal)
- ~44.000 produk fashion e-commerce
- Metadata: nama produk, kategori, subkategori, warna, gender
- Disampling 2.000 gambar secara stratified by `masterCategory`

**Custom product — HIFI**
- ~20 foto produk dari berbagai sudut
- Di-augmentasi menjadi ~200 variasi (flip, rotasi, brightness, contrast, crop)
- Ditambahkan langsung ke FAISS index tanpa retraining model

---

## Tech Stack

| Komponen | Library |
|---|---|
| Feature extraction | PyTorch · TorchVision ResNet50 |
| Similarity search | FAISS IndexFlatIP |
| Augmentasi | TorchVision Transforms Functional |
| Web app | Streamlit |
| Data processing | Pandas · NumPy |
| Eksperimen | Google Colab |

---

## Project Structure

```
similar-product-finder/
├── app.py                 
├── fix_paths.py            
├── requirements.txt
├── README.md
├── .gitignore
├── models/                 
│   ├── resnet50_extractor.pth
│   ├── faiss.index
│   └── image_paths.pkl
└── data/
    ├── images/             
    └── wifi_product/       
```

---

## Setup & Run

```bash
# clone repo
git clone https://github.com/A-Alfin/similar-product-finder
cd similar-product-finder

# buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt

# jalankan app
streamlit run app.py
```

> Model files (`resnet50_extractor.pth`, `faiss.index`, `image_paths.pkl`) akan otomatis ter-download dari Hugging Face saat pertama kali app dijalankan.

---

## ML Concepts

| Konsep | Implementasi |
|---|---|
| Transfer learning | ResNet50 pretrained ImageNet, zero fine-tuning |
| Feature extraction | FC layer dihapus, avgpool output = 2048-dim embedding |
| L2 normalization | Cosine similarity = dot product, efisien di FAISS |
| Data augmentation | 10 variasi per foto untuk memperkuat representasi produk custom |
| Similarity search | FAISS IndexFlatIP, exact search, O(n·d) complexity |

---

## Possible Improvements

- Fine-tuning dengan triplet loss untuk embedding lebih domain-specific
- FAISS IVFFlat untuk approximate search pada dataset 100k+ produk
- CLIP embedding untuk multi-modal query (teks + gambar)
- Metadata re-ranking — filter by kategori sebelum similarity search
- MLflow untuk experiment tracking iterasi model

---

*Built by Alfin — Portfolio Project 2026*