# Aplikasi Tebak Gambar CIFAR-10 (TK2 Kelompok 2)

Aplikasi Streamlit untuk mengunggah foto dan menampilkan prediksi model CNN hasil TK2.

## Isi folder

```
app/
├── app.py              # aplikasi Streamlit
├── requirements.txt    # GANTI dengan requirements.txt dari hasil_tk2.zip (versi TF = versi Colab)
├── README.md
└── model/
    ├── model_cifar10_terbaik.keras   # dari hasil_tk2.zip
    └── model_info.json               # dari hasil_tk2.zip
```

## 1. Coba di laptop dulu

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
Browser terbuka di `http://localhost:8501`.

## 2. Deploy ke Streamlit Community Cloud (gratis, paling mudah)

1. Buat repo GitHub baru (public), upload seluruh isi folder `app/` (termasuk folder `model/`).
   Model MobileNetV2 sekitar 10 MB, masih aman untuk GitHub.
2. Buka https://share.streamlit.io → **Create app** → pilih repo tadi, branch `main`, file `app.py`.
3. Di **Advanced settings**, pilih Python **3.11** atau **3.12**.
4. Klik **Deploy**. Build pertama ±3–5 menit karena harus memasang TensorFlow.
5. Salin link `https://<nama-app>.streamlit.app` ke laporan.

> Kalau muncul error saat memuat model, hampir selalu karena versi TensorFlow di `requirements.txt` beda dengan di Colab. Pakai `requirements.txt` yang dibuat otomatis oleh notebook.

## 3. Alternatif: Hugging Face Spaces

Buat Space baru, pilih template **Streamlit**, lalu upload `app.py`, `requirements.txt`, dan folder `model/`. Langkah lainnya sama.

## Catatan teknis

- Input model: gambar 32×32×3, nilai 0–1. Preprocessing di `prepare_image()` sama persis dengan notebook
  (perbaiki rotasi EXIF → potong tengah jadi persegi → perkecil ke 32×32 → bagi 255).
- Augmentasi, resize ke 96×96, dan normalisasi MobileNetV2 sudah ada di dalam model, jadi aplikasi tidak perlu mengurusnya.
- Slider "batas keyakinan" di sidebar hanya mengubah tampilan (hijau/kuning), bukan prediksinya.
