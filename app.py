import pickle
import numpy as np
import faiss
import zipfile
import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "A-Alfin/similar-product-finder-models"
MODELS_DIR = Path("models")
INDEX_PATH = MODELS_DIR / "faiss.index"
PATHS_PKL  = MODELS_DIR / "image_paths.pkl"
MODEL_PATH = MODELS_DIR / "resnet50_extractor.pth"
WIFI_DIR   = Path("data/wifi_product")
TOP_K      = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Finder · HIFI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0f172a;
}

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1d4ed8 100%);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 1rem;
}
.hero-title span { color: #60a5fa; }
.hero-desc {
    font-size: 1rem;
    color: #cbd5e1;
    max-width: 600px;
    line-height: 1.7;
    margin-bottom: 1.8rem;
}
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── section ── */
.section { margin: 2.5rem 0; }
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #1d4ed8;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 0.75rem;
    line-height: 1.2;
}
.section-desc {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.75;
    max-width: 680px;
}

/* ── problem cards ── */
.problem-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 1.5rem;
}
.problem-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
}
.problem-icon { font-size: 1.6rem; margin-bottom: 0.75rem; }
.problem-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.4rem;
}
.problem-text { font-size: 0.82rem; color: #64748b; line-height: 1.6; }

/* ── keunggulan ── */
.keunggulan-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 1.5rem;
}
.keunggulan-card {
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    position: relative;
}
.keunggulan-card.highlight {
    border-color: #1d4ed8;
    background: #eff6ff;
}
.highlight-badge {
    position: absolute;
    top: -10px; right: 14px;
    background: #1d4ed8;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 999px;
}
.keunggulan-icon { font-size: 1.5rem; margin-bottom: 0.6rem; }
.keunggulan-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.3rem;
}
.keunggulan-text { font-size: 0.8rem; color: #64748b; line-height: 1.6; }

/* ── how it works ── */
.step-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 1.5rem;
}
.step-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
}
.step-number {
    width: 28px; height: 28px;
    background: #1d4ed8;
    color: white;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.75rem;
}
.step-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.step-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.3rem;
}
.step-text { font-size: 0.75rem; color: #64748b; line-height: 1.5; }

/* ── tech stack ── */
.tech-grid {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.tech-chip {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #334155;
}
.tech-chip span { color: #1d4ed8; }

/* ── divider ── */
.divider {
    height: 1px;
    background: #e2e8f0;
    margin: 2.5rem 0;
}

/* ── demo section ── */
.demo-wrap {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
}
.upload-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.5rem;
}

/* ── result card ── */
.card-body { padding: 8px 10px 10px; }
.card-cat {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #1d4ed8;
    margin-bottom: 2px;
}
.card-name {
    font-size: 0.78rem;
    font-weight: 500;
    color: #0f172a;
    line-height: 1.3;
}
.card-sim {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 7px;
}
.sim-outer {
    flex: 1; height: 3px;
    background: #e2e8f0;
    border-radius: 999px;
}
.sim-inner {
    height: 3px;
    border-radius: 999px;
    background: #1d4ed8;
}
.sim-pct {
    font-size: 0.68rem;
    font-weight: 700;
    color: #1d4ed8;
    min-width: 32px;
    text-align: right;
}
.hifi-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.65rem;
    font-weight: 700;
    margin-top: 4px;
}

/* ── button ── */
.stButton > button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── empty state ── */
.empty-wrap {
    text-align: center;
    padding: 4rem 1rem;
    color: #94a3b8;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-text { font-size: 0.9rem; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── download dari HF ──────────────────────────────────────────────────────────
def download_models():
    MODELS_DIR.mkdir(exist_ok=True)
    for fname in ["resnet50_extractor.pth", "faiss.index", "image_paths.pkl"]:
        if not (MODELS_DIR / fname).exists():
            with st.spinner(f"Menyiapkan {fname}..."):
                hf_hub_download(repo_id=HF_REPO_ID, filename=fname,
                                local_dir=str(MODELS_DIR))

    images_dir = Path("data/images")
    if not images_dir.exists() or not any(images_dir.iterdir()):
        zip_dest = Path("data/images_sample.zip")
        zip_dest.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Menyiapkan gambar produk..."):
            hf_hub_download(repo_id=HF_REPO_ID, filename="images_sample.zip",
                            local_dir="data")
        with st.spinner("Mengekstrak gambar..."):
            with zipfile.ZipFile(zip_dest, "r") as z:
                z.extractall(images_dir)
            zip_dest.unlink()

download_models()


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def extract_query_feature(model, img):
    tensor = get_transform()(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(tensor).squeeze(-1).squeeze(-1).cpu().numpy()
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat.astype("float32")

model, index, image_paths = load_resources()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Portfolio Project · Deep Learning · Computer Vision</div>
    <h1 class="hero-title">Temukan Produk Serupa<br><span>Hanya dengan Satu Foto</span></h1>
    <p class="hero-desc">
        Sistem pencarian produk berbasis kemiripan visual menggunakan deep feature extraction.
        Terinspirasi dari Google Lens dan TikTok Shop — cukup foto produknya, sistem langsung
        menemukan produk paling mirip dari ribuan katalog dalam hitungan detik.
    </p>
    <div class="badge-row">
        <span class="badge">ResNet50</span>
        <span class="badge">FAISS IndexFlatIP</span>
        <span class="badge">2048-dim Embedding</span>
        <span class="badge">{index.ntotal:,} Produk dalam Database</span>
        <span class="badge">Streamlit · PyTorch</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section">
    <div class="section-label">Latar Belakang</div>
    <div class="section-title">Mengapa Visual Search?</div>
    <div class="section-desc">
        Cara orang mencari produk sudah berubah. Di TikTok Shop, user bisa klik produk yang dipakai
        seseorang di video dan langsung melihat produk serupa. Di Google Lens, cukup arahkan kamera
        ke suatu barang dan hasil pencarian muncul seketika. Kata kunci tidak lagi jadi satu-satunya
        jalan — <strong>gambar adalah query baru.</strong>
    </div>
    <div class="problem-grid">
        <div class="problem-card">
            <div class="problem-icon">🔍</div>
            <div class="problem-title">Pencarian Kata Kunci Terbatas</div>
            <div class="problem-text">
                User sering tidak tahu nama produk yang ingin dicari. Foto jauh lebih intuitif
                dan akurat dibanding mengetik deskripsi.
            </div>
        </div>
        <div class="problem-card">
            <div class="problem-icon">🛒</div>
            <div class="problem-title">Rekomendasi Tidak Relevan</div>
            <div class="problem-text">
                Sistem rekomendasi berbasis kategori atau tag sering menghasilkan produk yang
                tidak benar-benar mirip secara visual.
            </div>
        </div>
        <div class="problem-card">
            <div class="problem-icon">⚡</div>
            <div class="problem-title">Ekspektasi User Makin Tinggi</div>
            <div class="problem-text">
                Setelah terbiasa dengan Google Lens dan TikTok Shop, user mengharapkan
                pengalaman visual search di platform manapun.
            </div>
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KEUNGGULAN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section">
    <div class="section-label">Keunggulan</div>
    <div class="section-title">Apa yang Membuat Sistem Ini Berbeda?</div>
    <div class="section-desc">
        Lebih dari sekadar image similarity — sistem ini dirancang dengan pendekatan
        <em>business-aware</em> yang bisa diprioritaskan untuk produk tertentu tanpa fine-tuning ulang.
    </div>
    <div class="keunggulan-grid">
        <div class="keunggulan-card highlight">
            <div class="highlight-badge">Fitur Utama</div>
            <div class="keunggulan-icon">📡</div>
            <div class="keunggulan-title">Prioritas Produk HIFI</div>
            <div class="keunggulan-text">
                Produk WiFi HIFI ditambahkan ke dalam index dengan teknik augmentasi data —
                flip, rotasi, perubahan brightness dan contrast — menghasilkan ~200 variasi
                dari 20 foto asli. Ini memperkuat representasi produk HIFI di feature space
                sehingga muncul di hasil teratas saat user mencari produk WiFi apapun.
            </div>
        </div>
        <div class="keunggulan-card">
            <div class="keunggulan-icon">🧠</div>
            <div class="keunggulan-title">Transfer Learning tanpa Fine-Tuning</div>
            <div class="keunggulan-text">
                ResNet50 pretrained ImageNet digunakan langsung sebagai feature extractor.
                FC layer dihapus, output avgpool (2048-dim) menjadi representasi visual produk.
                Tidak perlu data berlabel atau proses training — efisien dan langsung siap pakai.
            </div>
        </div>
        <div class="keunggulan-card">
            <div class="keunggulan-icon">⚡</div>
            <div class="keunggulan-title">Pencarian Instan dengan FAISS</div>
            <div class="keunggulan-text">
                FAISS IndexFlatIP melakukan exact cosine similarity search pada ribuan vektor
                dalam milidetik. Vektor di-L2 normalize sehingga inner product setara cosine
                similarity — efisien tanpa kehilangan akurasi.
            </div>
        </div>
        <div class="keunggulan-card">
            <div class="keunggulan-icon">🔧</div>
            <div class="keunggulan-title">Mudah Dikustomisasi</div>
            <div class="keunggulan-text">
                Produk baru bisa ditambahkan ke index tanpa melatih ulang model dari nol.
                Cukup extract feature produk baru dan append ke FAISS index yang sudah ada.
                Cocok untuk katalog produk yang terus berkembang.
            </div>
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section">
    <div class="section-label">Cara Kerja</div>
    <div class="section-title">Pipeline Sistem</div>
    <div class="section-desc">
        Dari foto yang diupload hingga hasil rekomendasi, seluruh proses berjalan
        dalam satu pipeline yang konsisten antara tahap eksperimen dan deployment.
    </div>
    <div class="step-grid">
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-icon">📸</div>
            <div class="step-title">Upload Foto Produk</div>
            <div class="step-text">
                User mengupload foto produk dalam format JPG, PNG, atau WebP.
                Foto bisa dari kamera, screenshot, atau hasil download.
            </div>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-icon">🧬</div>
            <div class="step-title">Ekstraksi Fitur</div>
            <div class="step-text">
                Foto di-resize ke 224×224, dinormalisasi, lalu di-forward pass
                melalui ResNet50. Output avgpool menghasilkan vektor 2048-dim
                yang merepresentasikan visual produk.
            </div>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-icon">🔎</div>
            <div class="step-title">Similarity Search</div>
            <div class="step-text">
                Vektor query di-L2 normalize lalu dibandingkan dengan seluruh
                vektor di FAISS index menggunakan cosine similarity.
                Top-K vektor terdekat diambil sebagai hasil.
            </div>
        </div>
        <div class="step-card">
            <div class="step-number">4</div>
            <div class="step-icon">🛍️</div>
            <div class="step-title">Tampilkan Hasil</div>
            <div class="step-text">
                Produk paling mirip ditampilkan beserta similarity score.
                Produk HIFI yang ada di hasil ditandai secara khusus
                sebagai rekomendasi prioritas.
            </div>
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TECH STACK
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section">
    <div class="section-label">Teknologi</div>
    <div class="section-title">Tech Stack</div>
    <div class="tech-grid">
        <div class="tech-chip"><span>Model</span> · ResNet50 pretrained ImageNet</div>
        <div class="tech-chip"><span>Framework</span> · PyTorch + TorchVision</div>
        <div class="tech-chip"><span>Search</span> · FAISS IndexFlatIP</div>
        <div class="tech-chip"><span>Augmentasi</span> · TorchVision Transforms Functional</div>
        <div class="tech-chip"><span>Web App</span> · Streamlit</div>
        <div class="tech-chip"><span>Model Storage</span> · Hugging Face Hub</div>
        <div class="tech-chip"><span>Eksperimen</span> · Google Colab</div>
        <div class="tech-chip"><span>Dataset</span> · Fashion Product Images · Kaggle</div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DEMO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-label">Coba Sekarang</div>
<div class="section-title">Demo Visual Search</div>
<div class="section-desc">
    Upload foto produk apapun — baju, sepatu, tas, atau perangkat WiFi —
    dan sistem akan menemukan produk paling mirip dari database secara instan.
    Coba upload foto router atau modem WiFi untuk melihat produk HIFI muncul di hasil.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="demo-wrap">', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2.5], gap="large")

with col_left:
    st.markdown('<p class="upload-label">Upload gambar produk (JPG / PNG / WebP)</p>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        st.image(query_img, use_container_width=True, caption="Query image")
        top_k = st.slider("Jumlah hasil", 4, 12, TOP_K, step=4)

        if st.button("🔍  Cari Produk Mirip"):
            with st.spinner("Sedang memproses..."):
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
            f'<p style="font-size:0.85rem;font-weight:600;color:#334155;margin-bottom:1rem">'
            f'Menampilkan {len(results)} produk serupa</p>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)
        for i, (idx, score) in enumerate(results):
            path    = image_paths[idx]
            is_hifi = isinstance(path, str) and path.startswith("WIFI_PRODUCT::")
            col     = cols[i % 4]
            sim_pct = round(score * 100, 1)

            with col:
                if is_hifi:
                    fname   = path.split("::")[1].split("_aug")[0]
                    matches = list(WIFI_DIR.glob(f"{fname}.*"))
                    if matches:
                        st.image(Image.open(matches[0]).convert("RGB"),
                                 use_container_width=True)
                    else:
                        st.image(Image.new("RGB", (300, 300), "#dbeafe"),
                                 use_container_width=True)
                else:
                    local_path = Path("data/images") / Path(path).name if path else None
                    if local_path and local_path.exists():
                        st.image(Image.open(local_path).convert("RGB"),
                                 use_container_width=True)
                    else:
                        st.image(Image.new("RGB", (300, 300), "#f8fafc"),
                                 use_container_width=True)

                if is_hifi:
                    st.markdown(f"""
                    <div class="card-body">
                        <div class="card-cat">Rekomendasi Utama</div>
                        <div class="card-name">Produk WiFi HIFI</div>
                        <span class="hifi-badge">✓ Produk HIFI</span>
                        <div class="card-sim">
                            <div class="sim-outer">
                                <div class="sim-inner" style="width:{min(sim_pct,100)}%"></div>
                            </div>
                            <span class="sim-pct">{sim_pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card-body">
                        <div class="card-cat">Produk Serupa</div>
                        <div class="card-name">Produk #{idx}</div>
                        <div class="card-sim">
                            <div class="sim-outer">
                                <div class="sim-inner" style="width:{min(sim_pct,100)}%"></div>
                            </div>
                            <span class="sim-pct">{sim_pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-icon">🔍</div>
            <div class="empty-text">Upload gambar produk untuk mulai mencari</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;color:#94a3b8;font-size:0.78rem">
    Dibangun oleh <strong style="color:#475569">Alfin</strong> · Portfolio Project 2026 ·
    ResNet50 + FAISS + Streamlit
</div>
""", unsafe_allow_html=True)