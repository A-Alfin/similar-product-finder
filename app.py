import pickle
import numpy as np
import faiss
import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

# ── config ───────────────────────────────────────────────────────────────────
HF_REPO_ID  = "A-Alfin/similar-product-finder-models"
MODELS_DIR  = Path("models")
INDEX_PATH  = MODELS_DIR / "faiss.index"
PATHS_PKL   = MODELS_DIR / "image_paths.pkl"
MODEL_PATH  = MODELS_DIR / "resnet50_extractor.pth"
WIFI_DIR    = Path("data/wifi_product")
TOP_K       = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Similar Product Finder",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* header */
.header-wrap {
    background: linear-gradient(135deg, #FF6B35 0%, #F7C59F 50%, #EFEFD0 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.header-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0;
}
.header-sub {
    font-size: 0.9rem;
    color: #444;
    margin-top: 0.3rem;
}

/* stat chips */
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.stat-chip {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #333;
}

/* upload zone */
.upload-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.5rem;
}

/* product card */
.product-card {
    background: white;
    border-radius: 12px;
    border: 1.5px solid #f0f0f0;
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.2s;
    margin-bottom: 1rem;
}
.product-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.card-body {
    padding: 10px 12px;
}
.card-category {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #FF6B35;
    margin-bottom: 3px;
}
.card-name {
    font-size: 0.82rem;
    color: #1a1a1a;
    font-weight: 500;
    line-height: 1.3;
}
.card-sim {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
}
.sim-bar-outer {
    flex: 1;
    height: 4px;
    background: #f0f0f0;
    border-radius: 999px;
}
.sim-bar-inner {
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(to right, #FF6B35, #f7c59f);
}
.sim-pct {
    font-size: 0.72rem;
    font-weight: 600;
    color: #FF6B35;
    min-width: 36px;
    text-align: right;
}

/* wifi badge */
.wifi-badge {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 4px;
}

/* search button */
.stButton > button {
    background: #FF6B35 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 1rem;
}
.result-count {
    font-size: 0.82rem;
    color: #888;
    font-weight: 400;
    margin-left: 6px;
}

/* empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #bbb;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-text { font-size: 0.95rem; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── download model files ─────────────────────────────
def download_models():
    MODELS_DIR.mkdir(exist_ok=True)

    # download model files
    for fname in ["resnet50_extractor.pth", "faiss.index", "image_paths.pkl"]:
        dest = MODELS_DIR / fname
        if not dest.exists():
            with st.spinner(f"Menyiapkan {fname}..."):
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=fname,
                    local_dir=str(MODELS_DIR)
                )

    # download dan extract gambar sampel
    images_dir = Path("data/images")
    if not images_dir.exists() or not any(images_dir.iterdir()):
        import zipfile
        zip_dest = Path("data/images_sample.zip")
        zip_dest.parent.mkdir(parents=True, exist_ok=True)

        with st.spinner("Menyiapkan gambar produk (26MB)..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="images_sample.zip",
                local_dir="data"
            )

        with st.spinner("Mengekstrak gambar..."):
            with zipfile.ZipFile(zip_dest, "r") as z:
                z.extractall(images_dir)
            zip_dest.unlink()  # hapus zip setelah extract

        st.success("Gambar produk siap!")

download_models()

# debug — hapus setelah masalah solved
import os
from pathlib import Path

st.write("=== DEBUG INFO ===")
st.write(f"data/images exists: {Path('data/images').exists()}")
if Path('data/images').exists():
    files = list(Path('data/images').glob('*.jpg'))
    st.write(f"Jumlah jpg di data/images: {len(files)}")
    if files:
        st.write(f"Contoh path: {files[0]}")
else:
    st.write("Folder data/images tidak ada")

st.write(f"data/images_sample.zip exists: {Path('data/images_sample.zip').exists()}")
st.write(f"Image paths sample: {image_paths[:3]}")
st.write("=== END DEBUG ===")

# ── load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = models.resnet50(weights=None)
    model = nn.Sequential(*list(model.children())[:-1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    index = faiss.read_index(str(INDEX_PATH))

    with open(PATHS_PKL, "rb") as f:
        image_paths = pickle.load(f)

    return model, index, image_paths

@st.cache_data
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])

def extract_query_feature(model, img):
    tensor = get_transform()(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(tensor).squeeze(-1).squeeze(-1).cpu().numpy()
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat.astype("float32")

# ── load ──────────────────────────────────────────────────────────────────────
model, index, image_paths = load_resources()

# debug
model, index, image_paths = load_resources()

st.write("=== DEBUG ===")
st.write(f"Contoh image_path[0]: {image_paths[0]}")
st.write(f"Path exists: {Path(image_paths[0]).exists() if image_paths[0] else 'None'}")

# coba cari file berdasarkan nama file saja
sample_path = image_paths[0]
if sample_path:
    fname = Path(sample_path).name
    local_path = Path("data/images") / fname
    st.write(f"Nama file: {fname}")
    st.write(f"Local path: {local_path}")
    st.write(f"Local path exists: {local_path.exists()}")
st.write("=== END ===")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-wrap">
    <p class="header-title">🛍️ Similar Product Finder</p>
    <p class="header-sub">Upload foto produk — temukan produk paling mirip dari database secara instan</p>
    <div class="stat-row">
        <span class="stat-chip">ResNet50</span>
        <span class="stat-chip">FAISS IndexFlatIP</span>
        <span class="stat-chip">2048-dim embedding</span>
        <span class="stat-chip">{index.ntotal:,} produk</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.8], gap="large")

with col_left:
    st.markdown('<p class="upload-label">Upload gambar produk</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "jpg / png / webp",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        st.image(query_img, use_container_width=True,
                 caption="Query image")

        top_k = st.slider("Jumlah hasil", 4, 12, TOP_K, step=4)

        if st.button("🔍 Cari Produk Mirip"):
            with st.spinner("Sedang mencari..."):
                feat               = extract_query_feature(model, query_img)
                scores, idx_result = index.search(feat, top_k + 1)

            st.session_state["results"] = [
                (int(i), float(s))
                for i, s in zip(idx_result[0], scores[0])
            ][:top_k]

with col_right:
    if "results" in st.session_state:
        results = st.session_state["results"]

        st.markdown(
            f'<p class="section-title">Produk Serupa <span class="result-count">{len(results)} hasil</span></p>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)
        for i, (idx, score) in enumerate(results):
            path    = image_paths[idx]
            is_wifi = isinstance(path, str) and path.startswith("WIFI_PRODUCT::")
            col     = cols[i % 4]
            sim_pct = round(score * 100, 1)

            with col:
                # tampilkan gambar
                if is_wifi:
                    fname   = path.split("::")[1].split("_aug")[0]
                    matches = list(WIFI_DIR.glob(f"{fname}.*"))
                    if matches:
                        img = Image.open(matches[0]).convert("RGB")
                        st.image(img, use_container_width=True)
                    else:
                        st.image(
                            Image.new("RGB", (300, 300), "#e8f5e9"),
                            use_container_width=True
                        )
                elif path and Path(path).exists():
                    st.image(Image.open(path).convert("RGB"),
                             use_container_width=True)
                else:
                    # placeholder dengan warna sesuai similarity
                    placeholder = Image.new("RGB", (300, 300), "#fafafa")
                    st.image(placeholder, use_container_width=True)

                # card info
                if is_wifi:
                    st.markdown(f"""
                    <div class="card-body">
                        <div class="card-category">Rekomendasi</div>
                        <div class="card-name">Produk WiFi Perusahaan</div>
                        <span class="wifi-badge">✓ Produk Kami</span>
                        <div class="card-sim">
                            <div class="sim-bar-outer">
                                <div class="sim-bar-inner" style="width:{min(sim_pct,100)}%"></div>
                            </div>
                            <span class="sim-pct">{sim_pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card-body">
                        <div class="card-category">Similar Product</div>
                        <div class="card-name">Produk #{idx}</div>
                        <div class="card-sim">
                            <div class="sim-bar-outer">
                                <div class="sim-bar-inner" style="width:{min(sim_pct,100)}%"></div>
                            </div>
                            <span class="sim-pct">{sim_pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🛍️</div>
            <div class="empty-text">Upload gambar produk untuk mulai mencari</div>
        </div>
        """, unsafe_allow_html=True)