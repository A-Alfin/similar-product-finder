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
HF_REPO_ID    = "A-Alfin/similar-product-finder-models"
MODELS_DIR    = Path("models")
INDEX_PATH    = MODELS_DIR / "faiss.index"
PATHS_PKL     = MODELS_DIR / "image_paths.pkl"
MODEL_PATH    = MODELS_DIR / "resnet50_extractor.pth"
WIFI_DIR      = Path("data/wifi_product")
TOP_K         = 8
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── download model files ────────────────────
def download_models():
    MODELS_DIR.mkdir(exist_ok=True)
    files = [
        "resnet50_extractor.pth",
        "faiss.index",
        "image_paths.pkl",
    ]
    for fname in files:
        dest = MODELS_DIR / fname
        if not dest.exists():
            with st.spinner(f"Downloading {fname} dari Hugging Face..."):
                path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=fname,
                    local_dir=str(MODELS_DIR)
                )
            st.success(f"{fname} siap")

download_models()

# ── load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # load model
    model = models.resnet50(weights=None)
    model = nn.Sequential(*list(model.children())[:-1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    # load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # load image paths
    with open(PATHS_PKL, "rb") as f:
        image_paths = pickle.load(f)

    return model, index, image_paths

# ── preprocessing ─────────────────────────────────────────────────────────────
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

def extract_query_feature(model, img: Image.Image) -> np.ndarray:
    transform = get_transform()
    tensor    = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(tensor)
        feat = feat.squeeze(-1).squeeze(-1)
        feat = feat.cpu().numpy()

    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat.astype("float32")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Similar Product Finder",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Similar Product Finder")
st.caption("ResNet50 + FAISS · Portfolio Project · Alfin 2026")
st.divider()

# ── load ──────────────────────────────────────────────────────────────────────
model, index, image_paths = load_resources()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Info")
    st.metric("Produk dalam database", index.ntotal)
    st.metric("Feature dimension", "2048-dim")
    st.metric("Model", "ResNet50")
    st.metric("Search engine", "FAISS IndexFlatIP")
    st.divider()
    top_k = st.slider("Jumlah hasil", min_value=4, max_value=12, value=TOP_K, step=4)

# ── main ──────────────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 2.5])

with col_upload:
    st.markdown("#### Upload gambar produk")
    uploaded = st.file_uploader(
        "jpg / png / webp",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        st.image(query_img, caption="Query image", use_container_width=True)

        if st.button("Cari produk mirip", use_container_width=True, type="primary"):
            with st.spinner("Mencari..."):
                query_feat          = extract_query_feature(model, query_img)
                scores, indices_res = index.search(query_feat, top_k + 1)

            results = [
                (int(idx), float(score))
                for idx, score in zip(indices_res[0], scores[0])
            ][:top_k]

            st.session_state["results"] = results

with col_results:
    if "results" in st.session_state:
        st.markdown("#### Produk serupa ditemukan")

        cols = st.columns(4)
        for i, (idx, score) in enumerate(st.session_state["results"]):
            path    = image_paths[idx]
            is_wifi = isinstance(path, str) and path.startswith("WIFI_PRODUCT::")
            col     = cols[i % 4]

            with col:
                if is_wifi:
                    fname   = path.split("::")[1].split("_aug")[0]
                    matches = list(WIFI_DIR.glob(f"{fname}.*"))
                    if matches:
                        st.image(Image.open(matches[0]).convert("RGB"),
                                 use_container_width=True)
                    else:
                        st.info("WiFi product")
                    st.success("WIFI PERUSAHAAN")
                    st.caption(f"sim: {score:.3f}")
                else:
                    if path and Path(path).exists():
                        st.image(Image.open(path).convert("RGB"),
                                 use_container_width=True)
                    else:
                        st.image(
                            Image.new("RGB", (224, 224), color=(240, 240, 240)),
                            use_container_width=True
                        )
                        st.caption("Gambar belum tersedia")
                    st.caption(f"sim: {score:.3f}")
    else:
        st.info("Upload gambar dan klik 'Cari produk mirip' untuk mulai.")
