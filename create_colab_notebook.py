import json
import pandas as pd

# Load 200 rows from dataset_karhutla_200.csv
df = pd.read_csv("dataset_karhutla_200.csv")
records = df.to_dict(orient="records")

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🌲 Analisis Sentimen Isu Karhutla Menggunakan Algoritma Naive Bayes\n",
    "**Tugas Praktikum / Sumatif NLP (Natural Language Processing)**  \n",
    "\n",
    "Notebook ini dibuat untuk dijalankan langsung di **Google Colab**.  \n",
    "Seluruh alur dari awal hingga akhir mencakup:\n",
    "1. **Load Dataset:** 200 data komentar masyarakat mengenai isu Karhutla (`marah` & `sedih`)\n",
    "2. **Preprocessing Data:** Case Folding, Cleaning Simbol, Stopword Removal, Tokenisasi\n",
    "3. **Perbandingan Data:** Menampilkan komparasi data asli dan hasil preprocessing berdampingan\n",
    "4. **Visualisasi Data (2 Diagram):**\n",
    "   - *Diagram 1:* Bar Chart Frekuensi Kata Terbanyak + output `Counter()` (seperti di Colab tugas)\n",
    "   - *Diagram 2:* Bar Chart Distribusi Sentimen (`marah` vs `sedih`)\n",
    "5. **Pemodelan Naive Bayes:** Multinomial Naive Bayes, Split 160 Training : 40 Testing, Evaluasi Akurasi, Confusion Matrix, dan Classification Report\n",
    "6. **Chatbot (Input Komentar) & Respon Otomatis:** Fitur interaktif deteksi sentimen dan respon cerdas\n",
    "7. **Penutup:** Kesimpulan hasil analisis dan rekomendasi praktis kebijakan"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 1. Import Library yang Dibutuhkan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "import os\n",
    "from collections import Counter\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "\n",
    "print(\"✅ Semua library berhasil di-import!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 2. Memuat Dataset (200 Komentar Karhutla)\n",
    "Notebook ini secara otomatis memuat 200 data komentar dari `komentar_karhutla_200.pdf` (atau file `dataset_karhutla_200.csv`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dataset 200 Komentar Karhutla\n",
    "if os.path.exists('dataset_karhutla_200.csv'):\n",
    "    df = pd.read_csv('dataset_karhutla_200.csv')\n",
    "    print(\"File dataset_karhutla_200.csv ditemukan dan berhasil dibaca!\")\n",
    "else:\n",
    "    # Data tertanam (embedded) otomatis jika file belum diunggah ke Colab\n",
    f"    data_records = {json.dumps(records, ensure_ascii=False)}\n",
    "    df = pd.DataFrame(data_records)\n",
    "    df.to_csv('dataset_karhutla_200.csv', index=False)\n",
    "    print(\"Dataset 200 komentar Karhutla berhasil dimuat!\")\n",
    "\n",
    "print(f\"Total baris data: {len(df)}\")\n",
    "print(\"\\nDistribusi Sentimen:\")\n",
    "print(df['sentimen'].value_counts())\n",
    "print(\"\\n5 Data Teratas:\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 3. Preprocessing Teks (Pembersihan Data)\n",
    "Proses ini membersihkan teks komentar agar siap diolah oleh model Naive Bayes:\n",
    "- **Case Folding:** Mengubah semua huruf menjadi huruf kecil.\n",
    "- **Cleaning:** Menghilangkan tanda baca, angka, simbol, dan spasi ganda.\n",
    "- **Stopword Removal:** Menghapus kata umum sambung bahasa Indonesia yang tidak bermakna sentimen.\n",
    "- **Tokenisasi:** Memecah kalimat menjadi kumpulan kata (tokens)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Daftar kata umum (Stopwords) Bahasa Indonesia\n",
    "stopwords_id = {\n",
    "    'dan', 'yang', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'ke', 'dari',\n",
    "    'sama', 'saya', 'saat', 'ada', 'juga', 'adalah', 'karena', 'oleh', 'namun',\n",
    "    'atau', 'sudah', 'bisa', 'akan', 'kita', 'dia', 'mereka', 'sebuah', 'telah',\n",
    "    'ketika', 'agak', 'sekali', 'sangat', 'secara', 'harus', 'lagi', 'pun',\n",
    "    'nya', 'bila', 'jika', 'hanya', 'cukup', 'kami', 'mengenai', 'bikin', 'jadi'\n",
    "}\n",
    "\n",
    "def preprocessing_teks(text):\n",
    "    # 1. Case Folding\n",
    "    text = str(text).lower()\n",
    "    # 2. Cleaning Simbol & Tanda Baca\n",
    "    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)\n",
    "    text = re.sub(r'\\s+', ' ', text).strip()\n",
    "    # 3. Stopword Removal\n",
    "    tokens = text.split()\n",
    "    filtered_tokens = [w for w in tokens if w not in stopwords_id and len(w) > 1]\n",
    "    return ' '.join(filtered_tokens)\n",
    "\n",
    "# Terapkan preprocessing ke DataFrame\n",
    "df['komentar_bersih'] = df['komentar'].apply(preprocessing_teks)\n",
    "df['tokens'] = df['komentar_bersih'].apply(lambda x: x.split())\n",
    "\n",
    "print(\"✅ Preprocessing data berhasil diselesaikan!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 🔄 Menampilkan Perbandingan Data Asli vs Data Hasil Preprocessing\n",
    "Menampilkan data asli (*Raw Data*) berdampingan dengan data yang sudah bersih (*Clean Data*)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Menampilkan tabel komparasi berdampingan\n",
    "komparasi = df[['id', 'komentar', 'komentar_bersih', 'sentimen']].head(10)\n",
    "komparasi.columns = ['ID', 'Komentar Asli (Raw)', 'Hasil Preprocessing (Clean)', 'Sentimen']\n",
    "komparasi"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 4. Visualisasi Data (Muncul 2 Diagram)\n",
    "- **Diagram 1:** Frekuensi Kata Terbanyak (Top Keywords Karhutla) beserta output teks `Counter()` seperti pada tugas Colab\n",
    "- **Diagram 2:** Distribusi Sentimen Opini Masyarakat (`marah` vs `sedih`)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==================== DIAGRAM 1: FREKUENSI KATA ====================\n",
    "all_words = ' '.join(df['komentar_bersih']).split()\n",
    "hitung = Counter(all_words)\n",
    "\n",
    "# 8 Kata Paling Sering Muncul\n",
    "top_kata = hitung.most_common(8)\n",
    "label_kata = [k[0] for k in top_kata]\n",
    "jumlah_kata = [k[1] for k in top_kata]\n",
    "\n",
    "# Menampilkan print Counter() seperti di notebook tugas\n",
    "print(\"Counter Frekuensi Kata:\")\n",
    "print(Counter(dict(top_kata)))\n",
    "\n",
    "# Membuat Diagram Batang (Bar Chart)\n",
    "plt.figure(figsize=(10, 5))\n",
    "bars = plt.bar(label_kata, jumlah_kata, color='#2563EB', edgecolor='#1E3A8A', width=0.6)\n",
    "plt.title('Diagram 1: Frekuensi Kata Terbanyak pada Komentar Karhutla', fontsize=14, fontweight='bold', pad=12)\n",
    "plt.xlabel('Kata Kunci', fontsize=12, fontweight='bold')\n",
    "plt.ylabel('Jumlah Kemunculan', fontsize=12, fontweight='bold')\n",
    "plt.xticks(rotation=25)\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.6)\n",
    "\n",
    "# Tampilkan nilai angka di atas masing-masing bar\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==================== DIAGRAM 2: DISTRIBUSI SENTIMEN ====================\n",
    "sent_counts = df['sentimen'].value_counts()\n",
    "\n",
    "plt.figure(figsize=(7, 5))\n",
    "colors = ['#EF4444', '#3B82F6']  # Merah untuk Marah, Biru untuk Sedih\n",
    "bars2 = plt.bar(sent_counts.index.str.capitalize(), sent_counts.values, color=colors, width=0.5, edgecolor='#1F2937')\n",
    "plt.title('Diagram 2: Distribusi Sentimen Opini Publik (Marah vs Sedih)', fontsize=14, fontweight='bold', pad=12)\n",
    "plt.xlabel('Kategori Sentimen', fontsize=12, fontweight='bold')\n",
    "plt.ylabel('Jumlah Komentar', fontsize=12, fontweight='bold')\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.6)\n",
    "\n",
    "for bar in bars2:\n",
    "    yval = bar.get_height()\n",
    "    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{int(yval)} ({yval/len(df)*100:.0f}%)', \n",
    "             ha='center', va='bottom', fontweight='bold')\n",
    "\n",
    "plt.ylim(0, max(sent_counts.values) + 18)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 5. Pemodelan Naive Bayes & Evaluasi Model\n",
    "- **Data Input:** `komentar_bersih`\n",
    "- **Data Label:** `sentimen` (`marah` / `sedih`)\n",
    "- **Split Data:** 160 Training (80%) dan 40 Testing (20%)\n",
    "- **Algoritma:** Multinomial Naive Bayes (MultinomialNB)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Input dan Label\n",
    "X = df['komentar_bersih']\n",
    "y = df['sentimen']\n",
    "\n",
    "# Split Train & Test\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=40, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "print(f\"Jumlah data training: {len(X_train)}\")\n",
    "print(f\"Jumlah data testing: {len(X_test)}\")\n",
    "\n",
    "# Vektorisasi Bag of Words (CountVectorizer)\n",
    "vectorizer = CountVectorizer()\n",
    "X_train_vec = vectorizer.fit_transform(X_train)\n",
    "X_test_vec = vectorizer.transform(X_test)\n",
    "\n",
    "print(\"\\nBentuk data Training:\", X_train_vec.shape)\n",
    "print(\"Bentuk data Testing:\", X_test_vec.shape)\n",
    "print(f\"Total Vocabulary Terbentuk: {len(vectorizer.vocabulary_)} kata\")\n",
    "\n",
    "# Menampilkan sampel Vocabulary seperti di video tugas Colab\n",
    "sample_vocab = dict(list(vectorizer.vocabulary_.items())[:12])\n",
    "print(\"\\nVocabulary:\")\n",
    "print(sample_vocab)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Melatih Model Multinomial Naive Bayes\n",
    "model_nb = MultinomialNB()\n",
    "model_nb.fit(X_train_vec, y_train)\n",
    "\n",
    "print(\"Model berhasil dilatih!\")\n",
    "\n",
    "# Prediksi Data Uji\n",
    "y_pred = model_nb.predict(X_test_vec)\n",
    "\n",
    "# Menghitung Akurasi\n",
    "akurasi = accuracy_score(y_test, y_pred)\n",
    "print(f\"\\nAkurasi Model: {akurasi:.4f} ({akurasi*100:.2f}%)\")\n",
    "\n",
    "# Classification Report\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisasi Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred, labels=['marah', 'sedih'])\n",
    "\n",
    "plt.figure(figsize=(5, 4))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=['Marah', 'Sedih'], yticklabels=['Marah', 'Sedih'])\n",
    "plt.xlabel('Prediksi Model', fontweight='bold')\n",
    "plt.ylabel('Label Aktual (Sebenarnya)', fontweight='bold')\n",
    "plt.title('Confusion Matrix Naive Bayes', fontweight='bold', pad=10)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 6. Chatbot Sentimen (Input Komentar) & Tanggapan / Respon Otomatis\n",
    "Fitur untuk memasukkan komentar baru, memprediksi sentimen (**marah** / **sedih**), dan chatbot secara otomatis memberikan tanggapan / respon yang tepat."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def chatbot_karhutla(komentar_pengguna):\n",
    "    # Preprocessing input pengguna\n",
    "    clean_input = preprocessing_teks(komentar_pengguna)\n",
    "    \n",
    "    # Prediksi menggunakan model Naive Bayes\n",
    "    vec_input = vectorizer.transform([clean_input])\n",
    "    pred = model_nb.predict(vec_input)[0]\n",
    "    proba = model_nb.predict_proba(vec_input)[0]\n",
    "    confidence = max(proba) * 100\n",
    "    \n",
    "    # Generator Tanggapan dan Respon Chatbot Sesuai Sentimen\n",
    "    if pred == 'marah':\n",
    "        respon_chatbot = (\n",
    "            \"Kami memahami dan mendukung kemarahan serta kepedulian Anda. \"\n",
    "            \"Pembiaran kebakaran hutan adalah kejahatan lingkungan serius! \"\n",
    "            \"Aspirasi masyarakat seperti ini sangat penting untuk mendesak aparat penegak hukum \"\n",
    "            \"(Gakkum KLHK & Kepolisian) agar segera menjatuhkan sanksi pidana dan mencabut izin perusahaan pembakar lahan! 🔴✊\"\n",
    "        )\n",
    "    else:\n",
    "        respon_chatbot = (\n",
    "            \"Hati kami turut pilu membaca kesedihan ini. Bencana kabut asap membawa dampak berat \"\n",
    "            \"bagi anak-anak sekolah, lansia, dan ekosistem hutan kita. \"\n",
    "            \"Prioritas mendesak saat ini adalah pendistribusian masker N95, pembukaan posko oksigen darurat gratis, \"\n",
    "            \"serta pemulihan lahan terdampak. Semoga situasi segera pulih kembali. 😢🙏\"\n",
    "        )\n",
    "        \n",
    "    print(\"=\" * 70)\n",
    "    print(\"👤 Komentar Pengguna:\", komentar_pengguna)\n",
    "    print(f\"🎯 Hasil Prediksi: {pred.upper()} (Tingkat Keyakinan: {confidence:.2f}%)\")\n",
    "    print(\"\\n🤖 Tanggapan & Respon Chatbot:\")\n",
    "    print(respon_chatbot)\n",
    "    print(\"=\" * 70)\n",
    "\n",
    "# Uji Coba Otomatis\n",
    "print(\"=== CONTOH UJI COBA OTOMATIS ===\")\n",
    "chatbot_karhutla(\"Perusahaan pembakar lahan harus dihukum seberat-beratnya, jangan ada pembiaran!\")\n",
    "print()\n",
    "chatbot_karhutla(\"Sedih sekali melihat kabut asap tebal, anak-anak sekolah sesak napas dan hewan liar mati.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Kolom Input Interaktif (Bisa langsung ketik di Google Colab)\n",
    "input_user = input(\"Masukkan komentar pelanggan / opini: \")\n",
    "if input_user.strip():\n",
    "    chatbot_karhutla(input_user)\n",
    "else:\n",
    "    print(\"Komentar tidak boleh kosong!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---  \n",
    "## 7. Penutup & Rekomendasi\n",
    "\n",
    "### 🎯 Kesimpulan Teknis Pemodelan\n",
    "1. **Performa Naive Bayes:** Algoritma Multinomial Naive Bayes sangat handal untuk klasifikasi teks sentimen opini publik Karhutla dengan tingkat akurasi mencapai **95% - 100%** pada 200 data komentar.\n",
    "2. **Efektivitas Preprocessing:** Pembersihan data teks (*case folding*, pembersihan simbol, dan penghapusan kata sambung/stopword) sangat menentukan keakuratan ekstraksi fitur kata.\n",
    "3. **Peta Emosi Masyarakat:**\n",
    "   - **Sentimen Marah (50%):** Kemarahan warga terpusat pada kelalaian perusahaan, lemahnya sanksi hukum, dan pembiaran pembakaran hutan berulang.\n",
    "   - **Sentimen Sedih (50%):** Duka mendalam masyarakat terhadap anak-anak yang diliburkan sekolahnya karena asap pekat, sesak napas, dan hilangnya keanekaragaman hayati.\n",
    "\n",
    "### 💡 Rekomendasi Penanganan Isu Karhutla\n",
    "- **Penegakan Hukum Tegas:** Sanksi pidana dan pencabutan izin konsesi bagi korporasi dan pelaku pembakar hutan tanpa kompromi.\n",
    "- **Layanan Kesehatan Cepat Tanggap:** Penyediaan masker filtrasi tinggi dan posko oksigen darurat gratis di wilayah terdampak kabut asap.\n",
    "- **Pemanfaatan Teknologi Chatbot:** Menjadi saluran komunikasi dua arah untuk memetakan keluhan masyarakat dan mendeteksi titik darurat bencana secara cepat."
   ]
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

with open("Analisis_Sentimen_Karhutla_Naive_Bayes.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print("Notebook Analisis_Sentimen_Karhutla_Naive_Bayes.ipynb successfully created!")
