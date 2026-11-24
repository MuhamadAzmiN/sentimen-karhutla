import json

# Code for Google Colab that is unique, elegant, tailored for Karhutla, and distinct from the friend's template
distinct_colab_code = """# ==============================================================================
# PROYEK ANALISIS SENTIMEN ISU KARHUTLA (KEBAKARAN HUTAN & LAHAN)
# KLASIFIKASI TEKS MENGGUNAKAN ALGORITMA NAIVE BAYES & CHATBOT INTERAKTIF
# ==============================================================================

!pip install pandas scikit-learn emoji matplotlib seaborn

import os
import re
import string
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# TAHAP 1: MEMUAT DATASET KOMENTAR KARHUTLA
# ------------------------------------------------------------------------------
file_path = "komentar_karhutla_200.csv"
if not os.path.exists(file_path):
    if os.path.exists("dataset_karhutla_200.csv"):
        file_path = "dataset_karhutla_200.csv"

df_karhutla = pd.read_csv(file_path, encoding="utf-8")

print("--- Cuplikan 5 Data Teratas ---")
print(df_karhutla.head())

print(f"\\nTotal Data Komentar : {len(df_karhutla)}")
print("\\nSebaran Kategori Sentimen (Hanya Marah & Sedih):")
print(df_karhutla["sentimen"].value_counts())

# ------------------------------------------------------------------------------
# TAHAP 2: PRA-PEMROSESAN TEKS (TEXT PREPROCESSING)
# ------------------------------------------------------------------------------
stopword_id = {
    "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "untuk",
    "dengan", "nya", "sih", "aja", "deh", "banget", "pada", "adalah",
    "karena", "oleh", "namun", "akan", "kita", "dia", "mereka", "juga"
}

def bersihkan_komentar(kalimat):
    # 1. Lowercasing / Case folding
    teks = str(kalimat).lower()
    
    # 2. Hapus emoji & karakter khusus
    teks = emoji.replace_emoji(teks, replace=" ")
    
    # 3. Hapus seluruh tanda baca
    tabel_hapus = str.maketrans(string.punctuation, " " * len(string.punctuation))
    teks = teks.translate(tabel_hapus)
    
    # 4. Hapus angka / digit
    teks = re.sub(r"\\d+", " ", teks)
    
    # 5. Filtering kata umum (stopwords)
    token_bersih = [kata for kata in teks.split() if kata not in stopword_id and len(kata) > 1]
    
    # 6. Normalisasi spasi
    return " ".join(token_bersih)

# Terapkan preprocessing
df_karhutla["komentar_bersih"] = df_karhutla["komentar"].apply(bersihkan_komentar)

print("\\n--- Perbandingan Komentar Sebelum vs Sesudah Preprocessing ---")
print(df_karhutla[["komentar", "komentar_bersih", "sentimen"]].head(5))

# ------------------------------------------------------------------------------
# TAHAP 3: VISUALISASI DISTRIBUSI SENTIMEN (DIAGRAM 1)
# ------------------------------------------------------------------------------
sebaran_sentimen = df_karhutla["sentimen"].value_counts()

plt.figure(figsize=(7, 4.5))
warna_sentimen = ["#D32F2F", "#1976D2"] # Merah untuk Marah, Biru untuk Sedih
bars1 = plt.bar(sebaran_sentimen.index.str.upper(), sebaran_sentimen.values, color=warna_sentimen, width=0.45, edgecolor="black", linewidth=1.2)

plt.title("Visualisasi 1: Distribusi Sentimen Masyarakat Mengenai Karhutla", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Label Sentimen", fontweight="bold")
plt.ylabel("Jumlah Ulasan", fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars1:
    tinggi = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, tinggi + 1.5, f"{int(tinggi)} ({tinggi/len(df_karhutla)*100:.0f}%)", 
             ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.ylim(0, max(sebaran_sentimen.values) + 15)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# TAHAP 4: ANALISIS & VISUALISASI FREKUENSI KATA TERBANYAK (DIAGRAM 2)
# ------------------------------------------------------------------------------
gabungan_kata = " ".join(df_karhutla["komentar_bersih"]).split()
frekuensi_kata = Counter(gabungan_kata)

# Mengambil 10 kata teratas
top_10 = frekuensi_kata.most_common(10)
list_kata = [item[0] for item in top_10]
list_jumlah = [item[1] for item in top_10]

print("\\nCounter Kata Terbanyak:")
print(dict(top_10))

plt.figure(figsize=(9, 4.5))
bars2 = plt.bar(list_kata, list_jumlah, color="#E65100", edgecolor="#BF360C", width=0.55)

plt.title("Visualisasi 2: Top 10 Kata Kunci yang Paling Sering Muncul", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Kata Kunci", fontweight="bold")
plt.ylabel("Frekuensi Kemunculan", fontweight="bold")
plt.xticks(rotation=30, ha="right", fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars2:
    tinggi = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, tinggi + 1, f"{int(tinggi)}", ha="center", va="bottom", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# TAHAP 5: PEMISAHAN DATA (TRAIN-TEST SPLIT 80:20)
# ------------------------------------------------------------------------------
X_teks = df_karhutla["komentar_bersih"]
y_label = df_karhutla["sentimen"]

X_latih, X_uji, y_latih, y_uji = train_test_split(
    X_teks, y_label, test_size=0.2, random_state=42, stratify=y_label
)

print(f"\\nJumlah Data Training (80%) : {len(X_latih)} ulasan")
print(f"Jumlah Data Testing  (20%) : {len(X_uji)} ulasan")

# ------------------------------------------------------------------------------
# TAHAP 6: EKSTRAKSI FITUR (COUNTVECTORIZER)
# ------------------------------------------------------------------------------
vektor_fitur = CountVectorizer()
X_latih_vec = vektor_fitur.fit_transform(X_latih)
X_uji_vec = vektor_fitur.transform(X_uji)

print(f"Bentuk Matriks Data Training : {X_latih_vec.shape}")
print(f"Jumlah Kosakata Terbentuk    : {len(vektor_fitur.vocabulary_)} kata unik")

# ------------------------------------------------------------------------------
# TAHAP 7: PELATIHAN & EVALUASI MODEL MULTINOMIAL NAIVE BAYES
# ------------------------------------------------------------------------------
klasifikator_nb = MultinomialNB()
klasifikator_nb.fit(X_latih_vec, y_latih)

prediksi_uji = klasifikator_nb.predict(X_uji_vec)
skor_akurasi = accuracy_score(y_uji, prediksi_uji)

print("\\n" + "="*50)
print(f"HASIL EVALUASI MODEL NAIVE BAYES: {skor_akurasi * 100:.2f}%")
print("="*50)
print("\\nLaporan Klasifikasi Presisi & Recall:")
print(classification_report(y_uji, prediksi_uji, zero_division=0))

# ------------------------------------------------------------------------------
# TAHAP 8: CHATBOT ANALISIS OPINI KARHUTLA & RESPON OTOMATIS
# ------------------------------------------------------------------------------
# Vektorisasi seluruh korpus untuk bot
vektor_bot = CountVectorizer()
X_full = vektor_bot.fit_transform(X_teks)
model_bot = MultinomialNB()
model_bot.fit(X_full, y_label)

def jalankan_chatbot():
    print("\\n" + "="*60)
    print("      🔥 ASISTEN BOT ANALISIS SENTIMEN KARHUTLA 🔥")
    print("      Hanya Mengklasifikasikan: [MARAH] & [SEDIH]")
    print("      (Ketik 'keluar' untuk mengakhiri interaksi)")
    print("="*60)
    
    # Contoh kalimat bawaan dari data karhutla untuk simulasi
    contoh_karhutla = [
        "Asap karhutla membuat udara sulit dihirup, kasihan anak-anak sekolah sesak napas.",
        "Perusahaan yang membakar lahan harus dihukum seberat-beratnya, jangan ada pembiaran!"
    ]
    
    print("\\n[Simulasi Uji Otomatis dari Data Karhutla]:")
    for sample in contoh_karhutla:
        sample_clean = bersihkan_komentar(sample)
        sample_vec = vektor_bot.transform([sample_clean])
        pred_label = model_bot.predict(sample_vec)[0]
        
        if pred_label == "marah":
            balasan = "Kritik dan kemarahan Anda sangat berdasar. Tindak pidana pembakaran hutan harus segera diusut tuntas oleh aparat penegak hukum dan izin perusahaan terkait wajib dicabut!"
        else: # sedih
            balasan = "Kami turut merasakan duka dan keprihatinan atas dampak kabut asap ini. Prioritas saat ini adalah bantuan masker N95, posko udara bersih, serta perlindungan bagi warga dan anak-anak."
            
        print(f"\\nInput   : \\"{sample}\\"")
        print(f"Prediksi: [{pred_label.upper()}]")
        print(f"Bot     : {balasan}")
        
    print("\\n" + "-"*60)
    print("Sekarang giliran Anda, silakan masukkan komentar:")
    
    while True:
        teks_masuk = input("\\nKetik Komentar Anda: ")
        
        if teks_masuk.strip().lower() == "keluar":
            print("\\nTerima kasih atas partisipasi Anda dalam analisis isu Karhutla.")
            break
            
        teks_siap = bersihkan_komentar(teks_masuk)
        if not teks_siap:
            print("Harap ketik komentar berupa kalimat teks.")
            continue
            
        vec_masuk = vektor_bot.transform([teks_siap])
        hasil_prediksi = model_bot.predict(vec_masuk)[0]
        
        # Respon cerdas hanya untuk MARAH dan SEDIH
        if hasil_prediksi == "marah":
            tanggapan = "Kritik dan kemarahan Anda sangat berdasar. Tindak pidana pembakaran hutan harus segera diusut tuntas oleh aparat penegak hukum dan izin perusahaan terkait wajib dicabut!"
        else:
            tanggapan = "Kami turut merasakan duka dan keprihatinan atas dampak kabut asap ini. Prioritas saat ini adalah bantuan masker N95, posko udara bersih, serta perlindungan bagi warga dan anak-anak."
            
        print(f"-> Sentimen Terdeteksi : [{hasil_prediksi.upper()}]")
        print(f"-> Respon Bot          : {tanggapan}")

# Menjalankan fungsi chatbot
jalankan_chatbot()
"""

notebook_distinct = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [line + "\n" for line in distinct_colab_code.split("\n")]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("komentar_karhutla_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_distinct, f, indent=1, ensure_ascii=False)

print("komentar_karhutla_colab.ipynb successfully updated with distinct structure!")
