"""Aplikasi klasifikasi gambar CIFAR-10 – TK2 Kelompok 2.
Jalankan lokal:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model_cifar10_terbaik.keras"
INFO_PATH = MODEL_DIR / "model_info.json"

st.set_page_config(page_title="Tebak Gambar CIFAR-10", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    import tensorflow as tf  # import di sini supaya halaman tampil lebih cepat
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


def prepare_image(img: Image.Image, size: int = 32) -> np.ndarray:
    """Harus identik dengan prepare_image di notebook."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return np.asarray(img, dtype="float32") / 255.0


# ---------------------------------------------------------------- cek file model
if not MODEL_PATH.exists() or not INFO_PATH.exists():
    st.error(
        "File model belum ada. Jalankan notebook TK2 di Colab, lalu salin "
        "`model_cifar10_terbaik.keras` dan `model_info.json` ke folder `app/model/`."
    )
    st.stop()

info = load_info()
model = load_model()
NAMES = info["class_names"]
NAMES_ID = info.get("class_names_id", NAMES)
LABEL = [f"{id_} ({en})" for id_, en in zip(NAMES_ID, NAMES)]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Tentang model")
    st.markdown(
        f"**{info['model_name']}**  \n"
        f"Akurasi data uji CIFAR-10: **{info['test_accuracy']:.1%}**  \n"
        f"Macro F1: {info['macro_f1']:.3f}"
    )
    threshold = st.slider(
        "Batas keyakinan", 0.0, 1.0, 0.5, 0.05,
        help="Di bawah nilai ini aplikasi menandai prediksi sebagai 'ragu-ragu'.",
    )
    st.markdown("**10 kelas yang dikenali:**")
    st.markdown("  \n".join(f"• {l}" for l in LABEL))
    st.caption("TK2 Kelompok 2 · Artificial Intelligence")

# ---------------------------------------------------------------- halaman utama
st.title("🔎 Tebak Gambar CIFAR-10")
st.write(
    "Unggah foto (atau ambil dari kamera), lalu model CNN akan menebak isinya ke salah satu "
    "dari 10 kelas CIFAR-10. Hasil paling bagus kalau objeknya ada di tengah dan memenuhi foto."
)

tab_up, tab_cam = st.tabs(["📁 Unggah gambar", "📷 Kamera"])
images = []
with tab_up:
    ups = st.file_uploader("Pilih satu atau beberapa gambar", type=["jpg", "jpeg", "png", "webp"],
                           accept_multiple_files=True)
    for u in ups or []:
        images.append((u.name, Image.open(u)))
with tab_cam:
    shot = st.camera_input("Ambil foto")
    if shot is not None:
        images.append(("foto kamera", Image.open(shot)))

if not images:
    st.info("Belum ada gambar. Coba foto kucing, mobil, atau kapal di sekitarmu.")
    st.stop()

batch = np.stack([prepare_image(img) for _, img in images])
probs = model.predict(batch, verbose=0)

for (name, img), p in zip(images, probs):
    st.divider()
    order = p.argsort()[::-1]
    top, conf = order[0], float(p[order[0]])

    c1, c2, c3 = st.columns([1.2, 0.8, 2])
    with c1:
        st.image(ImageOps.exif_transpose(img).convert("RGB"), caption=name, use_container_width=True)
    with c2:
        small = Image.fromarray((prepare_image(img) * 255).astype("uint8")).resize((160, 160), Image.NEAREST)
        st.image(small, caption="Yang dilihat model (32×32 px)")
    with c3:
        if conf >= threshold:
            st.success(f"### {LABEL[top]}\nKeyakinan: **{conf:.1%}**")
        else:
            st.warning(
                f"### Ragu-ragu: mungkin {LABEL[top]}\nKeyakinan hanya **{conf:.1%}**. "
                "Bisa jadi objeknya bukan salah satu dari 10 kelas, atau terlalu kecil di foto."
            )
        df = pd.DataFrame({"kelas": [LABEL[i] for i in order[:5]], "probabilitas": p[order[:5]]})
        st.bar_chart(df, x="kelas", y="probabilitas", horizontal=True, sort="-probabilitas", height=220)

with st.expander("Kenapa kadang hasilnya aneh?"):
    st.write(
        "Model hanya mengenal 10 kelas, jadi foto laptop atau orang pun tetap akan dipaksa masuk "
        "ke salah satu kelas itu. Selain itu, model dilatih dengan gambar 32×32 piksel; detail "
        "yang hilang saat foto diperkecil (lihat kolom tengah) tidak bisa dipakai model."
    )
