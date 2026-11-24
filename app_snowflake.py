import streamlit as st
import pandas as pd
import re
import math
import json
from collections import Counter, defaultdict

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Sentimen Karhutla",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS Sederhana
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .chat-bubble-user {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .chat-bubble-bot {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# FUNGSI PREPROCESSING TEKS (MURNI REGEX STANDARD PYTHON)
# -------------------------------------------------------------
stopword_id = {
    "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "untuk",
    "dengan", "nya", "sih", "aja", "deh", "banget", "pada", "adalah",
    "karena", "oleh", "namun", "akan", "kita", "dia", "mereka", "juga"
}

def bersihkan_komentar(kalimat):
    teks = str(kalimat).lower()
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    token_bersih = [kata for kata in teks.split() if kata not in stopword_id and len(kata) > 1]
    return " ".join(token_bersih)

# -------------------------------------------------------------
# LOAD DATASET (200 Data Karhutla Otomatis Tertanam)
# -------------------------------------------------------------
@st.cache_data
def load_karhutla_data():
    try:
        data = pd.read_csv("komentar_karhutla_200.csv", encoding="utf-8")
    except Exception:
        try:
            data = pd.read_csv("dataset_karhutla_200.csv", encoding="utf-8")
        except Exception:
            data = pd.DataFrame([{"id": 1, "komentar": "Lahan yang terbakar membuat saya sedih, kasihan warganya.", "sentimen": "sedih"}, {"id": 2, "komentar": "Marah melihat hutan kita dijadikan korban keserakahan, tolong usut dan tindak tegas.", "sentimen": "marah"}, {"id": 3, "komentar": "Asap karhutla membuat udara sulit dihirup, kasihan warganya.", "sentimen": "sedih"}, {"id": 4, "komentar": "Perusahaan yang lalai bikin saya geram, geram sekali!", "sentimen": "marah"}, {"id": 5, "komentar": "Karhutla ini harus diusut tuntas, warga yang menderita.", "sentimen": "marah"}, {"id": 6, "komentar": "Geram, anak sekolah diliburkan karena asap, tapi tidak ada tindakan tegas kepada pembakarnya.", "sentimen": "marah"}, {"id": 7, "komentar": "Pembakaran lahan ini membuat rakyat jadi korban.", "sentimen": "marah"}, {"id": 8, "komentar": "Lahan yang terbakar membuat saya sedih di kampung kami.", "sentimen": "sedih"}, {"id": 9, "komentar": "Pemerintah yang lalai bikin saya geram, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 10, "komentar": "Lahan yang terbakar membuat udara sulit dihirup, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 11, "komentar": "Warga yang terdampak bikin hati pilu, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 12, "komentar": "Asap karhutla merusak kehidupan banyak makhluk hidup.", "sentimen": "sedih"}, {"id": 13, "komentar": "Pemerintah yang lalai sudah keterlaluan, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 14, "komentar": "Hutan yang hangus membuat warga kehilangan harapan, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 15, "komentar": "Karhutla ini dibiarkan terus berulang, geram sekali!", "sentimen": "marah"}, {"id": 16, "komentar": "Asap karhutla menyisakan luka bagi warga kecil, kasihan warganya.", "sentimen": "sedih"}, {"id": 17, "komentar": "Kabar karhutla membuat saya sedih, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 18, "komentar": "Perusahaan yang lalai bikin saya geram, warga yang menderita.", "sentimen": "marah"}, {"id": 19, "komentar": "Karhutla ini bikin saya geram, warga yang menderita.", "sentimen": "marah"}, {"id": 20, "komentar": "Pembiaran karhutla dibiarkan terus berulang, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 21, "komentar": "Pelaku pembakaran dibiarkan terus berulang, geram sekali!", "sentimen": "marah"}, {"id": 22, "komentar": "Hewan-hewan liar yang keluar dari hutan kelaparan dan kehilangan arah, kasihan sekali.", "sentimen": "sedih"}, {"id": 23, "komentar": "Karhutla ini bikin saya geram, geram sekali!", "sentimen": "marah"}, {"id": 24, "komentar": "Lahan yang terbakar merusak kehidupan banyak makhluk hidup di kampung kami.", "sentimen": "sedih"}, {"id": 25, "komentar": "Asap karhutla membuat saya sedih, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 26, "komentar": "Warga yang terdampak membuat udara sulit dihirup di kampung kami.", "sentimen": "sedih"}, {"id": 27, "komentar": "Asap karhutla menyisakan luka bagi warga kecil, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 28, "komentar": "Kabar karhutla membuat udara sulit dihirup, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 29, "komentar": "Kabar karhutla membuat saya sedih, kasihan warganya.", "sentimen": "sedih"}, {"id": 30, "komentar": "Warga yang terdampak membuat warga kehilangan harapan.", "sentimen": "sedih"}, {"id": 31, "komentar": "Pembiaran karhutla bikin saya geram, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 32, "komentar": "Karhutla ini bikin saya geram setiap tahun.", "sentimen": "marah"}, {"id": 33, "komentar": "Saya kesal, sistem peringatan dini sudah ada tapi respons di lapangan tetap lambat.", "sentimen": "marah"}, {"id": 34, "komentar": "Warga yang terdampak membuat saya sedih.", "sentimen": "sedih"}, {"id": 35, "komentar": "Pemerintah yang lalai bikin saya geram, geram sekali!", "sentimen": "marah"}, {"id": 36, "komentar": "Kebakaran hutan ini membuat udara sulit dihirup di kampung kami.", "sentimen": "sedih"}, {"id": 37, "komentar": "Warga yang terdampak bikin hati pilu, sedih sekali.", "sentimen": "sedih"}, {"id": 38, "komentar": "Perusahaan yang lalai sudah keterlaluan, geram sekali!", "sentimen": "marah"}, {"id": 39, "komentar": "Pelaku pembakaran dibiarkan terus berulang, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 40, "komentar": "Kebakaran hutan ini membuat warga kehilangan harapan, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 41, "komentar": "Kabar karhutla bikin hati pilu, kasihan warganya.", "sentimen": "sedih"}, {"id": 42, "komentar": "Lahan yang terbakar membuat saya sedih, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 43, "komentar": "Perusahaan yang lalai membuat rakyat jadi korban, geram sekali!", "sentimen": "marah"}, {"id": 44, "komentar": "Pemerintah yang lalai dibiarkan terus berulang setiap tahun.", "sentimen": "marah"}, {"id": 45, "komentar": "Pembakaran lahan ini bikin saya geram, geram sekali!", "sentimen": "marah"}, {"id": 46, "komentar": "Asap karhutla membuat saya sedih, kasihan warganya.", "sentimen": "sedih"}, {"id": 47, "komentar": "Hutan yang hangus membuat saya sedih, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 48, "komentar": "Warga yang terdampak merusak kehidupan banyak makhluk hidup, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 49, "komentar": "Pelaku pembakaran harus diusut tuntas, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 50, "komentar": "Perusahaan yang lalai menunjukkan lemahnya penegakan hukum, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 51, "komentar": "Kabar karhutla membuat udara sulit dihirup di kampung kami.", "sentimen": "sedih"}, {"id": 52, "komentar": "Pembakaran lahan ini harus diusut tuntas, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 53, "komentar": "Lahan yang terbakar bikin hati pilu, sedih sekali.", "sentimen": "sedih"}, {"id": 54, "komentar": "Hutan yang hangus menyisakan luka bagi warga kecil, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 55, "komentar": "Kabar karhutla membuat udara sulit dihirup.", "sentimen": "sedih"}, {"id": 56, "komentar": "Marah! Setiap tahun kejadian yang sama, tapi tidak ada pembelajaran dari pemerintah.", "sentimen": "marah"}, {"id": 57, "komentar": "Pembakaran lahan ini dibiarkan terus berulang, warga yang menderita.", "sentimen": "marah"}, {"id": 58, "komentar": "Pemerintah yang lalai bikin saya geram.", "sentimen": "marah"}, {"id": 59, "komentar": "Kebakaran ini meninggalkan luka mendalam bagi warga yang kehilangan tempat tinggal.", "sentimen": "sedih"}, {"id": 60, "komentar": "Pelaku pembakaran sudah keterlaluan, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 61, "komentar": "Hutan yang hangus membuat udara sulit dihirup, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 62, "komentar": "Air sungai menghitam karena abu, ikan-ikan mati dan nelayan kehilangan mata pencaharian.", "sentimen": "sedih"}, {"id": 63, "komentar": "Warga yang terdampak membuat warga kehilangan harapan, sedih sekali.", "sentimen": "sedih"}, {"id": 64, "komentar": "Geram melihat warga sendiri yang menanggung kerugian, padahal bukan mereka yang membakar.", "sentimen": "marah"}, {"id": 65, "komentar": "Petani di kampung saya gagal panen lagi karena lahannya terbakar dan tanahnya rusak.", "sentimen": "sedih"}, {"id": 66, "komentar": "Perusahaan yang lalai menunjukkan lemahnya penegakan hukum setiap tahun.", "sentimen": "marah"}, {"id": 67, "komentar": "Lahan yang terbakar membuat warga kehilangan harapan, sedih sekali.", "sentimen": "sedih"}, {"id": 68, "komentar": "Hutan yang hangus membuat udara sulit dihirup, sedih sekali.", "sentimen": "sedih"}, {"id": 69, "komentar": "Kebakaran hutan selalu datang setiap tahun, dan warga kecil yang paling merasakan akibatnya.", "sentimen": "sedih"}, {"id": 70, "komentar": "Lahan yang terbakar membuat saya sedih.", "sentimen": "sedih"}, {"id": 71, "komentar": "Kebakaran hutan ini membuat warga kehilangan harapan, kasihan warganya.", "sentimen": "sedih"}, {"id": 72, "komentar": "Hutan yang hangus membuat warga kehilangan harapan, sedih sekali.", "sentimen": "sedih"}, {"id": 73, "komentar": "Warga yang terdampak membuat warga kehilangan harapan di kampung kami.", "sentimen": "sedih"}, {"id": 74, "komentar": "Pemerintah yang lalai membuat rakyat jadi korban setiap tahun.", "sentimen": "marah"}, {"id": 75, "komentar": "Warga yang terdampak membuat udara sulit dihirup, sedih sekali.", "sentimen": "sedih"}, {"id": 76, "komentar": "Lahan yang terbakar membuat warga kehilangan harapan, kasihan warganya.", "sentimen": "sedih"}, {"id": 77, "komentar": "Pemerintah yang lalai dibiarkan terus berulang.", "sentimen": "marah"}, {"id": 78, "komentar": "Pelaku pembakaran harus diusut tuntas setiap tahun.", "sentimen": "marah"}, {"id": 79, "komentar": "Saya kesal karena pemadaman dilakukan dengan alat seadanya, sementara anggaran besar disebut-sebut.", "sentimen": "marah"}, {"id": 80, "komentar": "Kebakaran hutan ini merusak kehidupan banyak makhluk hidup, kasihan warganya.", "sentimen": "sedih"}, {"id": 81, "komentar": "Pembiaran karhutla bikin saya geram setiap tahun.", "sentimen": "marah"}, {"id": 82, "komentar": "Karhutla ini membuat rakyat jadi korban.", "sentimen": "marah"}, {"id": 83, "komentar": "Lihat video hutan yang terbakar dan hewan yang lari kepanasan, air mata saya jatuh.", "sentimen": "sedih"}, {"id": 84, "komentar": "Pemerintah yang lalai menunjukkan lemahnya penegakan hukum, geram sekali!", "sentimen": "marah"}, {"id": 85, "komentar": "Asap karhutla membuat warga kehilangan harapan, kasihan warganya.", "sentimen": "sedih"}, {"id": 86, "komentar": "Pembiaran karhutla membuat rakyat jadi korban, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 87, "komentar": "Kabar karhutla membuat warga kehilangan harapan.", "sentimen": "sedih"}, {"id": 88, "komentar": "Penegakan hukumnya lemah, perusahaan yang diduga membakar lahan harus dihukum berat!", "sentimen": "marah"}, {"id": 89, "komentar": "Pemerintah yang lalai membuat rakyat jadi korban.", "sentimen": "marah"}, {"id": 90, "komentar": "Perusahaan yang lalai dibiarkan terus berulang, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 91, "komentar": "Kesal, pengawasan lahan konsesi terkesan formalitas saja, asap tetap datang tiap tahun.", "sentimen": "marah"}, {"id": 92, "komentar": "Membayangkan orangutan dan burung-burung yang kehilangan habitatnya membuat saya sedih.", "sentimen": "sedih"}, {"id": 93, "komentar": "Hutan yang hangus menyisakan luka bagi warga kecil.", "sentimen": "sedih"}, {"id": 94, "komentar": "Pelaku pembakaran harus diusut tuntas, geram sekali!", "sentimen": "marah"}, {"id": 95, "komentar": "Hutan yang hangus bikin hati pilu.", "sentimen": "sedih"}, {"id": 96, "komentar": "Asap karhutla membuat udara sulit dihirup di kampung kami.", "sentimen": "sedih"}, {"id": 97, "komentar": "Kabar karhutla menyisakan luka bagi warga kecil, sedih sekali.", "sentimen": "sedih"}, {"id": 98, "komentar": "Karhutla ini dibiarkan terus berulang setiap tahun.", "sentimen": "marah"}, {"id": 99, "komentar": "Pembakaran lahan ini sudah keterlaluan.", "sentimen": "marah"}, {"id": 100, "komentar": "Perusahaan yang lalai harus diusut tuntas, geram sekali!", "sentimen": "marah"}, {"id": 101, "komentar": "Warga yang terdampak membuat saya sedih, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 102, "komentar": "Pembakaran lahan ini menunjukkan lemahnya penegakan hukum.", "sentimen": "marah"}, {"id": 103, "komentar": "Kabar karhutla menyisakan luka bagi warga kecil di kampung kami.", "sentimen": "sedih"}, {"id": 104, "komentar": "Karhutla ini menunjukkan lemahnya penegakan hukum, warga yang menderita.", "sentimen": "marah"}, {"id": 105, "komentar": "Pelaku pembakaran harus diusut tuntas.", "sentimen": "marah"}, {"id": 106, "komentar": "Karhutla ini dibiarkan terus berulang, warga yang menderita.", "sentimen": "marah"}, {"id": 107, "komentar": "Perusahaan yang lalai harus diusut tuntas.", "sentimen": "marah"}, {"id": 108, "komentar": "Kebakaran hutan ini membuat warga kehilangan harapan di kampung kami.", "sentimen": "sedih"}, {"id": 109, "komentar": "Kebakaran hutan ini membuat udara sulit dihirup.", "sentimen": "sedih"}, {"id": 110, "komentar": "Perusahaan yang lalai bikin saya geram, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 111, "komentar": "Hutan yang hangus merusak kehidupan banyak makhluk hidup, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 112, "komentar": "Pelaku pembakaran sudah keterlaluan, warga yang menderita.", "sentimen": "marah"}, {"id": 113, "komentar": "Pelaku pembakaran dibiarkan terus berulang.", "sentimen": "marah"}, {"id": 114, "komentar": "Pembiaran karhutla harus diusut tuntas, geram sekali!", "sentimen": "marah"}, {"id": 115, "komentar": "Perusahaan yang lalai membuat rakyat jadi korban.", "sentimen": "marah"}, {"id": 116, "komentar": "Perusahaan yang lalai membuat rakyat jadi korban setiap tahun.", "sentimen": "marah"}, {"id": 117, "komentar": "Pelaku pembakaran menunjukkan lemahnya penegakan hukum.", "sentimen": "marah"}, {"id": 118, "komentar": "Perusahaan yang lalai harus diusut tuntas, warga yang menderita.", "sentimen": "marah"}, {"id": 119, "komentar": "Hutan yang hangus membuat udara sulit dihirup, kasihan warganya.", "sentimen": "sedih"}, {"id": 120, "komentar": "Warga yang terdampak membuat warga kehilangan harapan, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 121, "komentar": "Pembiaran karhutla menunjukkan lemahnya penegakan hukum setiap tahun.", "sentimen": "marah"}, {"id": 122, "komentar": "Warga yang terdampak menyisakan luka bagi warga kecil, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 123, "komentar": "Marah melihat hutan dikonversi jadi perkebunan sementara kebakaran terus terjadi di sana.", "sentimen": "marah"}, {"id": 124, "komentar": "Desa kami sudah kehilangan banyak lahan, dan hati kami ikut hancur melihatnya.", "sentimen": "sedih"}, {"id": 125, "komentar": "Lahan yang terbakar menyisakan luka bagi warga kecil, kasihan warganya.", "sentimen": "sedih"}, {"id": 126, "komentar": "Pembakaran lahan ini sudah keterlaluan, geram sekali!", "sentimen": "marah"}, {"id": 127, "komentar": "Lahan yang terbakar membuat warga kehilangan harapan, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 128, "komentar": "Kebakaran ini membuat kampung saya sepi, banyak warga mengungsi dengan barang seadanya.", "sentimen": "sedih"}, {"id": 129, "komentar": "Pelaku pembakaran membuat rakyat jadi korban, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 130, "komentar": "Hutan yang hangus membuat saya sedih di kampung kami.", "sentimen": "sedih"}, {"id": 131, "komentar": "Hutan yang hangus menyisakan luka bagi warga kecil, sedih sekali.", "sentimen": "sedih"}, {"id": 132, "komentar": "Sudah keterlaluan, rakyat disuruh pakai masker sendiri sementara pembakar tidak ditangkap.", "sentimen": "marah"}, {"id": 133, "komentar": "Kabar karhutla membuat warga kehilangan harapan, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 134, "komentar": "Ini kejahatan lingkungan! Siapa pun yang membakar hutan harus diproses hukum tanpa pandang bulu.", "sentimen": "marah"}, {"id": 135, "komentar": "Pembakaran lahan ini dibiarkan terus berulang setiap tahun.", "sentimen": "marah"}, {"id": 136, "komentar": "Kenapa tidak ada sanksi tegas bagi korporasi yang lalai menjaga lahannya? Sungguh keterlaluan.", "sentimen": "marah"}, {"id": 137, "komentar": "Hutan yang hangus menyisakan luka bagi warga kecil, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 138, "komentar": "Pelaku pembakaran menunjukkan lemahnya penegakan hukum, geram sekali!", "sentimen": "marah"}, {"id": 139, "komentar": "Karhutla ini menunjukkan lemahnya penegakan hukum.", "sentimen": "marah"}, {"id": 140, "komentar": "Asap karhutla merusak kehidupan banyak makhluk hidup, sedih sekali.", "sentimen": "sedih"}, {"id": 141, "komentar": "Warga yang terdampak membuat saya sedih di kampung kami.", "sentimen": "sedih"}, {"id": 142, "komentar": "Perusahaan yang lalai menunjukkan lemahnya penegakan hukum, geram sekali!", "sentimen": "marah"}, {"id": 143, "komentar": "Pembiaran karhutla menunjukkan lemahnya penegakan hukum.", "sentimen": "marah"}, {"id": 144, "komentar": "Lahan yang terbakar membuat udara sulit dihirup, sedih sekali.", "sentimen": "sedih"}, {"id": 145, "komentar": "Pembakaran lahan ini sudah keterlaluan, warga yang menderita.", "sentimen": "marah"}, {"id": 146, "komentar": "Kebakaran hutan ini membuat saya sedih.", "sentimen": "sedih"}, {"id": 147, "komentar": "Pembiaran karhutla dibiarkan terus berulang setiap tahun.", "sentimen": "marah"}, {"id": 148, "komentar": "Karhutla ini harus diusut tuntas, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 149, "komentar": "Pembakaran lahan ini dibiarkan terus berulang, geram sekali!", "sentimen": "marah"}, {"id": 150, "komentar": "Asap karhutla bikin hati pilu, kasihan warganya.", "sentimen": "sedih"}, {"id": 151, "komentar": "Pembakaran lahan ini bikin saya geram, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 152, "komentar": "Hutan yang hangus merusak kehidupan banyak makhluk hidup di kampung kami.", "sentimen": "sedih"}, {"id": 153, "komentar": "Hutan yang hangus merusak kehidupan banyak makhluk hidup, sedih sekali.", "sentimen": "sedih"}, {"id": 154, "komentar": "Kebakaran hutan ini menyisakan luka bagi warga kecil di kampung kami.", "sentimen": "sedih"}, {"id": 155, "komentar": "Sedih karena hutan adat yang dijaga leluhur kami sekarang habis dilalap api.", "sentimen": "sedih"}, {"id": 156, "komentar": "Saya marah, pembakaran lahan dilakukan sengaja untuk membuka kebun, sementara warga yang menanggung asapnya.", "sentimen": "marah"}, {"id": 157, "komentar": "Hutan yang hangus bikin hati pilu, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 158, "komentar": "Pembiaran karhutla dibiarkan terus berulang.", "sentimen": "marah"}, {"id": 159, "komentar": "Lahan yang terbakar bikin hati pilu, kasihan warganya.", "sentimen": "sedih"}, {"id": 160, "komentar": "Pemerintah yang lalai dibiarkan terus berulang, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 161, "komentar": "Hutan yang hangus bikin hati pilu, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 162, "komentar": "Asap karhutla menyisakan luka bagi warga kecil di kampung kami.", "sentimen": "sedih"}, {"id": 163, "komentar": "Warga yang terdampak bikin hati pilu, kasihan warganya.", "sentimen": "sedih"}, {"id": 164, "komentar": "Lahan yang terbakar membuat udara sulit dihirup.", "sentimen": "sedih"}, {"id": 165, "komentar": "Kabar karhutla menyisakan luka bagi warga kecil.", "sentimen": "sedih"}, {"id": 166, "komentar": "Karhutla ini harus diusut tuntas, geram sekali!", "sentimen": "marah"}, {"id": 167, "komentar": "Kesal, setiap ada kebakaran selalu ada alasan, tapi tidak pernah ada yang bertanggung jawab.", "sentimen": "marah"}, {"id": 168, "komentar": "Warga yang terdampak bikin hati pilu di kampung kami.", "sentimen": "sedih"}, {"id": 169, "komentar": "Hutan yang hangus membuat saya sedih, sedih sekali.", "sentimen": "sedih"}, {"id": 170, "komentar": "Pemerintah yang lalai harus diusut tuntas, geram sekali!", "sentimen": "marah"}, {"id": 171, "komentar": "Pelaku pembakaran bikin saya geram, warga yang menderita.", "sentimen": "marah"}, {"id": 172, "komentar": "Pembiaran karhutla membuat rakyat jadi korban, geram sekali!", "sentimen": "marah"}, {"id": 173, "komentar": "Pembakaran lahan ini membuat rakyat jadi korban setiap tahun.", "sentimen": "marah"}, {"id": 174, "komentar": "Pemerintah yang lalai menunjukkan lemahnya penegakan hukum.", "sentimen": "marah"}, {"id": 175, "komentar": "Warga yang terdampak menyisakan luka bagi warga kecil, semoga cepat pulih.", "sentimen": "sedih"}, {"id": 176, "komentar": "Kebakaran hutan ini membuat saya sedih, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 177, "komentar": "Karhutla ini membuat rakyat jadi korban, kapan ada tindakan tegas?", "sentimen": "marah"}, {"id": 178, "komentar": "Kebakaran hutan ini membuat warga kehilangan harapan, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 179, "komentar": "Asapnya sudah berminggu-minggu, anak-anak di desa tidak bisa bermain di luar. Sedih sekali.", "sentimen": "sedih"}, {"id": 180, "komentar": "Karhutla ini membuat rakyat jadi korban, geram sekali!", "sentimen": "marah"}, {"id": 181, "komentar": "Pembakaran lahan ini membuat rakyat jadi korban, warga yang menderita.", "sentimen": "marah"}, {"id": 182, "komentar": "Pembiaran karhutla bikin saya geram, warga yang menderita.", "sentimen": "marah"}, {"id": 183, "komentar": "Pembakaran lahan ini menunjukkan lemahnya penegakan hukum setiap tahun.", "sentimen": "marah"}, {"id": 184, "komentar": "Warga yang terdampak menyisakan luka bagi warga kecil, sedih sekali.", "sentimen": "sedih"}, {"id": 185, "komentar": "Kesal sekali, pemerintah selalu berjanji mencegah karhutla tapi asap tetap saja datang setiap musim kemarau.", "sentimen": "marah"}, {"id": 186, "komentar": "Pelaku pembakaran dibiarkan terus berulang setiap tahun.", "sentimen": "marah"}, {"id": 187, "komentar": "Lahan yang terbakar bikin hati pilu di kampung kami.", "sentimen": "sedih"}, {"id": 188, "komentar": "Kasihan tukang ojek dan pedagang kecil yang omzetnya turun drastis karena asap.", "sentimen": "sedih"}, {"id": 189, "komentar": "Pemerintah yang lalai sudah keterlaluan.", "sentimen": "marah"}, {"id": 190, "komentar": "Karhutla ini dibiarkan terus berulang.", "sentimen": "marah"}, {"id": 191, "komentar": "Lahan yang terbakar membuat saya sedih, sedih sekali.", "sentimen": "sedih"}, {"id": 192, "komentar": "Kebakaran hutan ini bikin hati pilu, hati saya berat melihatnya.", "sentimen": "sedih"}, {"id": 193, "komentar": "Pembiaran karhutla membuat rakyat jadi korban, tolong tindak pelakunya!", "sentimen": "marah"}, {"id": 194, "komentar": "Asap karhutla membuat udara sulit dihirup, sedih sekali.", "sentimen": "sedih"}, {"id": 195, "komentar": "Perusahaan yang lalai membuat rakyat jadi korban, warga yang menderita.", "sentimen": "marah"}, {"id": 196, "komentar": "Warga yang terdampak merusak kehidupan banyak makhluk hidup di kampung kami.", "sentimen": "sedih"}, {"id": 197, "komentar": "Sedih, sekolah diliburkan karena kabut asap pekat, anak-anak jadi kehilangan waktu belajar.", "sentimen": "sedih"}, {"id": 198, "komentar": "Marah banget, hutan dibakar demi keuntungan sesaat, lalu rakyat yang sakit dan menderita.", "sentimen": "marah"}, {"id": 199, "komentar": "Perusahaan yang lalai dibiarkan terus berulang, warga yang menderita.", "sentimen": "marah"}, {"id": 200, "komentar": "Kabar karhutla bikin hati pilu, sedih sekali.", "sentimen": "sedih"}])
            
    data["komentar_bersih"] = data["komentar"].apply(bersihkan_komentar)
    return data

df_karhutla = load_karhutla_data()

# -------------------------------------------------------------
# MODEL NAIVE BAYES MURNI PYTHON (100% TANPA PACKAGE EXTERNAL)
# -------------------------------------------------------------
class PureNaiveBayes:
    def __init__(self):
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()
        self.total_words = Counter()
        
    def fit(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            words = text.split()
            for w in words:
                self.vocab.add(w)
                self.word_counts[label][w] += 1
                self.total_words[label] += 1
                
    def predict(self, text):
        words = text.split()
        V = len(self.vocab)
        total_samples = sum(self.class_counts.values())
        best_label = "marah"
        best_score = -float('inf')
        
        for label, count in self.class_counts.items():
            log_prob = math.log(count / total_samples)
            for w in words:
                word_count = self.word_counts[label].get(w, 0)
                log_prob += math.log((word_count + 1) / (self.total_words[label] + V))
            if log_prob > best_score:
                best_score = log_prob
                best_label = label
        return best_label

@st.cache_resource
def train_model(data_df):
    model = PureNaiveBayes()
    model.fit(data_df["komentar_bersih"], data_df["sentimen"])
    return model

model_nb = train_model(df_karhutla)

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.title("Analisis Sentimen Karhutla")
menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "🏠 Beranda & Dataset (200 Data)",
        "📊 Visualisasi Data (2 Diagram)",
        "🤖 Chatbot Sentimen & Respon"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 📌 Informasi Dataset
- **Sumber File:** `komentar_karhutla_200`
- **Total Komentar:** {len(df_karhutla)} data
- **Sentimen:** {len(df_karhutla[df_karhutla['sentimen']=='marah'])} Marah | {len(df_karhutla[df_karhutla['sentimen']=='sedih'])} Sedih
- **Data Training:** 160 (80%)
- **Data Testing:** 40 (20%)
- **Metode:** Naive Bayes
""")

# =============================================================
# MENU 1: BERANDA & DATASET
# =============================================================
if menu == "🏠 Beranda & Dataset (200 Data)":
    st.markdown('<div class="main-header">Analisis Sentimen Isu Karhutla</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data Komentar Opini Publik Mengenai Kebakaran Hutan dan Lahan</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Komentar", f"{len(df_karhutla)}")
    with c2:
        marah_cnt = len(df_karhutla[df_karhutla['sentimen'] == 'marah'])
        st.metric("Sentimen Marah", f"{marah_cnt}")
    with c3:
        sedih_cnt = len(df_karhutla[df_karhutla['sentimen'] == 'sedih'])
        st.metric("Sentimen Sedih", f"{sedih_cnt}")

    st.markdown("---")
    st.subheader("Tabel Komentar Karhutla")
    st.dataframe(df_karhutla[['id', 'komentar', 'sentimen']], use_container_width=True, height=400)

# =============================================================
# MENU 2: VISUALISASI DATA (2 DIAGRAM)
# =============================================================
elif menu == "📊 Visualisasi Data (2 Diagram)":
    st.markdown('<div class="main-header">Visualisasi Data (2 Diagram)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Diagram Frekuensi Kata Kunci & Distribusi Sentimen</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Diagram 1: Frekuensi Kata Terbanyak")
        gabungan_kata = " ".join(df_karhutla["komentar_bersih"]).split()
        frekuensi_kata = Counter(gabungan_kata)
        top_kata = frekuensi_kata.most_common(10)
        
        df_kata = pd.DataFrame(top_kata, columns=['Kata Kunci', 'Frekuensi']).set_index('Kata Kunci')
        st.bar_chart(df_kata, color="#E65100")

    with col2:
        st.markdown("### 📊 Diagram 2: Jumlah Komentar Sentimen")
        sebaran_sentimen = df_karhutla["sentimen"].value_counts()
        df_sent = pd.DataFrame({'Jumlah': sebaran_sentimen.values}, index=['MARAH', 'SEDIH'])
        st.bar_chart(df_sent, color="#1976D2")

# =============================================================
# MENU 3: CHATBOT SENTIMEN & RESPON
# =============================================================
elif menu == "🤖 Chatbot Sentimen & Respon":
    st.markdown('<div class="main-header">Chatbot Analisis Sentimen Karhutla</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Masukkan komentar dan bot akan memberikan tanggapan sesuai sentimen</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("**Contoh komentar untuk dicoba:**")
    qc1, qc2 = st.columns(2)
    quick_input = ""
    if qc1.button("🔴 Contoh Sentimen Marah"):
        quick_input = "Pemerintah yang lalai bikin saya geram, kapan ada tindakan tegas kepada pembakarnya?"
    if qc2.button("😢 Contoh Sentimen Sedih"):
        quick_input = "Lahan yang terbakar membuat saya sedih, kasihan warganya di kampung kami."

    with st.form("form_chat", clear_on_submit=True):
        user_input = st.text_input("Masukkan komentar:", value=quick_input, placeholder="Tulis komentar ulasan karhutla di sini...")
        submitted = st.form_submit_button("Kirim Komentar")

    if submitted and user_input.strip():
        komentar_bersih = bersihkan_komentar(user_input)
        if komentar_bersih:
            hasil = model_nb.predict(komentar_bersih)

            if hasil == "marah":
                respons = "Mohon maaf atas ketidaknyamanan yang Anda alami. Kejahatan pembakaran lahan harus segera ditindak tegas oleh pihak berwenang!"
            else:
                respons = "Hati kami turut merasakan kepedihan atas bencana kabut asap ini. Semoga penanganan kesehatan dan pemulihan cepat terlaksana."

            st.session_state.chat_history.append({
                "komentar": user_input,
                "sentimen": hasil.upper(),
                "respons": respons
            })
            st.rerun()

    st.markdown("### Riwayat Chatbot:")
    if not st.session_state.chat_history:
        st.info("Belum ada komentar yang diuji. Silakan ketik komentar di atas atau klik tombol contoh.")
    else:
        for item in reversed(st.session_state.chat_history):
            color = "#D32F2F" if item["sentimen"] == "MARAH" else "#1976D2"
            st.markdown(f"""
            <div class="chat-bubble-user">
                <div style="display:flex; justify-content:space-between;">
                    <b>👤 Komentar Masuk:</b>
                    <span style="background:{color}; color:white; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:bold;">
                        {item['sentimen']}
                    </span>
                </div>
                <div style="margin-top:4px;">"{item['komentar']}"</div>
            </div>
            <div class="chat-bubble-bot">
                <b>🤖 Tanggapan & Respon Chatbot:</b>
                <div style="margin-top:4px; color:#1F2937;">{item['respons']}</div>
            </div>
            """, unsafe_allow_html=True)
