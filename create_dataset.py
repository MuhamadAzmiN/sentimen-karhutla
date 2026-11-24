import pandas as pd
import numpy as np

# Sample reviews aligned with Colab Untitled7 & Sumatif Irma
# Products: Hoodie Polos, Dress Floral, Kaos Oversize, Celana Jeans, Kemeja Flanel, Jaket Denim, Blouse Wanita, Tas Selempang
base_reviews = [
    # Rows 0-4 exactly as seen in video
    ("Hoodie Polos", "Warnanya tidak sesuai dengan foto, saya agak kecewa dengan pengirimannya.", "negatif"),
    ("Dress Floral", "Suka sekali sama kualitasnya, sesuai foto dan mantap sekali bahannya.", "positif"),
    ("Kaos Oversize", "Warnanya tidak sesuai dengan foto, saya agak kecewa dan ukurannya terlalu kecil.", "negatif"),
    ("Dress Floral", "Modelnya keren, bahannya juga bagus dan nyaman sekali dipakai.", "positif"),
    ("Celana Jeans", "Respon admin cukup lama ketika saya bertanya mengenai stok ukuran celana.", "negatif"),
]

# Additional reviews designed to achieve word frequencies:
# bagus ~38, nyaman ~23, puas ~22, kurang ~19, ramah ~11, rusak ~7, cantik ~2
reviews_pos = [
    ("Hoodie Polos", "Bahan hoodie tebal dan bagus, jahitan rapi sekali sangat puas belanja di sini.", "positif"),
    ("Dress Floral", "Dress sangat cantik, warna cerah dan bahan adem nyaman dipakai sehari-hari.", "positif"),
    ("Kaos Oversize", "Bagus banget kaosnya, potongan oversize pas dan bahan katun nyaman sekali.", "positif"),
    ("Celana Jeans", "Kualitas jeans bagus dan jahitannya kuat, penjual ramah dan pengiriman kilat.", "positif"),
    ("Kemeja Flanel", "Kemeja flanel bagus kualitas premium, sangat nyaman dipakai kerja maupun santai.", "positif"),
    ("Jaket Denim", "Jaket jeans tebal dan bagus, model kekinian sangat puas dengan pelayanannya.", "positif"),
    ("Blouse Wanita", "Baju blouse cantik dan elegan, kain lembut sangat puas belanja di toko ini.", "positif"),
    ("Hoodie Polos", "Barang sampai dengan aman, packing rapi, kualitas produk bagus mantap.", "positif"),
    ("Celana Jeans", "Ukuran celana pas di badan, bahan melar dan nyaman dipakai jalan-jalan.", "positif"),
    ("Kaos Oversize", "Sangat puas dengan barangnya, bahan lembut adem dan kualitas sablon bagus.", "positif"),
    ("Dress Floral", "Model dress bagus sesuai ekspektasi, admin toko sangat ramah dan responsif.", "positif"),
    ("Kemeja Flanel", "Bagus banget kemejanya, bahan halus nyaman dan tidak mudah luntur.", "positif"),
    ("Jaket Denim", "Pelayanan toko ramah dan cepat, barang kualitas bagus harga terjangkau.", "positif"),
    ("Hoodie Polos", "Hoodie hangat dan nyaman, warna bagus tidak cepat pudar setelah dicuci.", "positif"),
    ("Celana Jeans", "Sangat puas, kualitas bahan jeans bagus dan tidak kaku saat dipakai duduk.", "positif"),
    ("Kaos Oversize", "Kaos bagus bahan katun adem, seller ramah pengemasan aman dan rapi.", "positif"),
    ("Dress Floral", "Puas banget sama bahannya, jatuh lembut dan nyaman di kulit saat beraktivitas.", "positif"),
    ("Kemeja Flanel", "Motif flanel bagus, ukuran pas dan nyaman dipakai seharian di kantor.", "positif"),
    ("Jaket Denim", "Keren jaketnya, kualitas jahitan bagus dan packaging rapi sangat puas.", "positif"),
    ("Blouse Wanita", "Bahan blouse halus dan jatuh bagus, admin ramah pengiriman cepat sampai.", "positif"),
    ("Hoodie Polos", "Bagus sekali bahannya, tebal tapi adem dan sangat nyaman untuk bepergian.", "positif"),
    ("Celana Jeans", "Kualitas bagus banget, warna pekat tidak luntur dan sangat nyaman di paha.", "positif"),
    ("Kaos Oversize", "Puas belanja di sini, kaosnya bagus dan respon admin sangat ramah melayani.", "positif"),
    ("Dress Floral", "Bagus dressnya, jahitan rapih tidak ada cacat, sangat puas dengan pembelian ini.", "positif"),
    ("Kemeja Flanel", "Bahan bagus berkualitas tinggi, dingin nyaman dan warna sesuai gambar katalog.", "positif"),
    ("Jaket Denim", "Bahan jaket bagus dan kuat, sangat puas dengan produk dan layanannya.", "positif"),
    ("Hoodie Polos", "Pelayanan mantap, respon admin ramah dan pengiriman kilat, barang bagus.", "positif"),
    ("Celana Jeans", "Barang bagus sesuai pesanan, celana nyaman dipakai seharian tidak gerah.", "positif"),
    ("Kaos Oversize", "Kualitas kaos bagus dan awet, bahan menyerap keringat sangat nyaman.", "positif"),
    ("Dress Floral", "Bahan kain bagus halus, seller ramah dan proses pengiriman cepat puas sekali.", "positif"),
    ("Kemeja Flanel", "Kemeja bagus dan modis, bahan tebal nyaman dipakai saat cuaca dingin.", "positif"),
    ("Jaket Denim", "Sangat puas, produk bagus jahitan presisi dan admin sangat ramah.", "positif"),
    ("Blouse Wanita", "Bagus sekali motifnya, ukuran sesuai deskripsi dan bahan adem nyaman.", "positif"),
    ("Hoodie Polos", "Kualitas jempolan, bahan bagus tidak nerawang, puas order kedua kalinya.", "positif"),
    ("Celana Jeans", "Jeans berkualitas, bahan melar nyaman dan cutting rapi sangat puas.", "positif"),
    ("Kaos Oversize", "Sangat puas, kaos distro kualitas bagus harga bersahabat dan nyaman.", "positif"),
    ("Dress Floral", "Admin ramah dan membantu pilih ukuran, bahan baju bagus dan nyaman.", "positif"),
    ("Kemeja Flanel", "Puas beli di sini, kemeja keren bahannya bagus dan packing aman.", "positif"),
    ("Jaket Denim", "Jaket denim bagus, warna sesuai foto katalog, sangat puas dengan pembelian.", "positif"),
    ("Hoodie Polos", "Respon penjual ramah, barang bagus tiba tepat waktu sangat memuaskan.", "positif"),
    ("Celana Jeans", "Puas belanja celana di sini, ukuran pas dan bahan sangat nyaman dipakai.", "positif"),
    ("Kaos Oversize", "Bahan kaos katun tebal tidak tipis, bagus dan adem dipakai siang hari.", "positif"),
    ("Dress Floral", "Bagus dan recommended, bahan lembut nyaman di badan tidak bikin gatal.", "positif"),
    ("Kemeja Flanel", "Suka sekali kemejanya, kualitas jahitan bagus dan bahan sangat nyaman.", "positif"),
    ("Jaket Denim", "Sangat puas dengan barangnya, model bagus tidak ketinggalan zaman.", "positif"),
    ("Blouse Wanita", "Bahan nyaman dan jatuh, warna bagus sesuai foto, penjual ramah bintang 5.", "positif"),
    ("Hoodie Polos", "Hoodie bagus mantap, bahan halus hangat dan nyaman untuk motoran.", "positif"),
    ("Celana Jeans", "Respon penjual sangat ramah, produk bagus jahitan rapi puas sekali.", "positif"),
    ("Kaos Oversize", "Kualitas kaos sangat memuaskan, bahan tebal bagus dan nyaman.", "positif"),
    ("Dress Floral", "Dress bagus, jahitan rapih dan packing rapi, sangat puas berbelanja.", "positif"),
    ("Kemeja Flanel", "Produk berkualitas bagus, bahan katun flanel nyaman dan halus dipakai.", "positif"),
    ("Jaket Denim", "Puas berbelanja, jaket bagus ukuran sesuai deskripsi produk.", "positif"),
    ("Hoodie Polos", "Sangat puas, kualitas bahan bagus dan pelayanan customer ramah.", "positif"),
    ("Celana Jeans", "Celana bagus pas di pinggang, bahan jeans stretch nyaman dan fleksibel.", "positif"),
    ("Kaos Oversize", "Baju kaos bagus dan trendy, sangat puas dengan bahan yang adem ini.", "positif"),
    ("Dress Floral", "Bagus sekali bajunya, bahan nyaman di kulit dan pelayanan toko ramah.", "positif"),
    ("Kemeja Flanel", "Kemeja flanel bagus, jahitan rapih dan pengiriman super cepat puas.", "positif"),
    ("Jaket Denim", "Jaket tebal dan bagus, nyaman dipakai outdoor sangat puas.", "positif"),
    ("Blouse Wanita", "Blouse bagus bahannya nyaman, potongan pas dan penjual ramah tamah.", "positif"),
    ("Hoodie Polos", "Barang original bagus, bahan tebal lembut dan nyaman sekali dipakai.", "positif"),
    ("Celana Jeans", "Puas dengan celananya, bahan bagus tebal namun tetap lentur dan nyaman.", "positif"),
    ("Kaos Oversize", "Kaos bagus, sablon awet tidak pecah, kain nyaman dipakai sehari-hari.", "positif"),
    ("Dress Floral", "Kualitas baju bagus dan tidak luntur saat dicuci, sangat puas.", "positif"),
    ("Kemeja Flanel", "Model keren bahan bagus, penjual ramah melayani konsultasi ukuran.", "positif"),
    ("Jaket Denim", "Jaket jeans bagus sekali, kancing kokoh dan bahan nyaman dipakai.", "positif"),
    ("Hoodie Polos", "Puas sekali, hoodie bagus warna pekat dan bahan sangat nyaman.", "positif"),
    ("Celana Jeans", "Produk bagus, ukuran sesuai chart dan nyaman dipakai saat kerja.", "positif"),
    ("Kaos Oversize", "Bagus dan berkualitas, pengemasan rapi dan seller sangat ramah.", "positif"),
    ("Dress Floral", "Sangat puas dengan jahitan dan bahannya yang bagus serta adem.", "positif"),
    ("Kemeja Flanel", "Kemeja bagus dan elegan, kain lembut nyaman dipakai seharian penuh.", "positif"),
    ("Jaket Denim", "Puas sekali dengan barang yang datang, kualitas bahan bagus kuat.", "positif"),
    ("Blouse Wanita", "Bahan bagus halus dan nyaman, respon admin ramah dan pengiriman aman.", "positif"),
    ("Hoodie Polos", "Barang bagus pengiriman cepat, bahan tebal dan sangat nyaman dipakai.", "positif"),
    ("Celana Jeans", "Kualitas celana bagus tahan lama, bahan tebal dan nyaman dipakai aktif.", "positif"),
    ("Kaos Oversize", "Baju kaos bagus dan murah, tidak nerawang dan nyaman untuk santai.", "positif"),
    ("Dress Floral", "Dress bagus cantik sekali, bahan flowy nyaman di badan sangat puas.", "positif"),
    ("Kemeja Flanel", "Sangat puas, kemeja bagus warna tajam dan bahan katun nyaman.", "positif"),
    ("Jaket Denim", "Jaket bagus jahitan rapi, admin ramah dan responsif saat dihubungi.", "positif"),
    ("Hoodie Polos", "Kualitas hoodie sangat bagus, bahan tebal hangat dan nyaman dipakai.", "positif"),
    ("Celana Jeans", "Puas belanja jeans di toko ini, bahan bagus dan potongan rapi.", "positif"),
    ("Kaos Oversize", "Kaos bagus, bahan tidak mudah kusut dan sangat nyaman seharian.", "positif"),
    ("Dress Floral", "Puas banget, produk bagus jahitan rapi dan seller sangat ramah.", "positif"),
    ("Kemeja Flanel", "Kemeja bagus bahan nyaman tidak panas, packing rapi dan pengiriman oke.", "positif"),
    ("Jaket Denim", "Bahan jaket bagus tebal dan nyaman, sangat puas dengan transaksi ini.", "positif"),
    ("Blouse Wanita", "Baju bagus kualitas butik, bahan jatuh nyaman dipakai sangat puas.", "positif"),
    ("Hoodie Polos", "Hoodie bagus bahan fleece tebal lembut, sangat nyaman dan hangat.", "positif"),
    ("Celana Jeans", "Sangat puas, jeans bagus warna tidak cepat pudar dan nyaman.", "positif"),
    ("Kaos Oversize", "Kaos bagus motif simpel, bahan adem nyaman dan seller sangat ramah.", "positif"),
]

reviews_neg = [
    ("Hoodie Polos", "Bahan hoodie tipis kurang bagus dan tidak sesuai dengan deskripsi toko.", "negatif"),
    ("Dress Floral", "Kualitas jahitan kurang rapi ada benang lepas dan sobek rusak di bagian lengan.", "negatif"),
    ("Kaos Oversize", "Ukuran baju kurang pas, terlalu kecil dan bahan kaos terasa panas gerah.", "negatif"),
    ("Celana Jeans", "Resleting celana rusak dan macet, bahan jeans kasar dan kurang nyaman dipakai.", "negatif"),
    ("Kemeja Flanel", "Warna kemeja luntur parah saat dicuci pertama kali, sangat mengecewakan kurang bagus.", "negatif"),
    ("Jaket Denim", "Kancing jaket copot dan rusak satu saat baru dibuka paketnya, kecewa sekali.", "negatif"),
    ("Blouse Wanita", "Ukuran blouse kurang sesuai, bahan tipis menerawang dan jahitan kurang rapi.", "negatif"),
    ("Hoodie Polos", "Pengiriman lama sekali hampir dua minggu, respon admin lambat dan kurang ramah.", "negatif"),
    ("Celana Jeans", "Kain celana tipis dan kasar, potongan kurang simetris tidak nyaman dipakai.", "negatif"),
    ("Kaos Oversize", "Sablon kaos pecah-pecah dan rusak setelah sekali cuci, kualitas kurang memuaskan.", "negatif"),
    ("Dress Floral", "Warna baju tidak sesuai dengan foto di etalase, terlihat kusam dan kurang bagus.", "negatif"),
    ("Kemeja Flanel", "Bahan kemeja kasar dan gatal di kulit, ukuran kurang pas mengecewakan pembeli.", "negatif"),
    ("Jaket Denim", "Jaket ada noda minyak kotor dan kainnya rusak di bagian kantong depan.", "negatif"),
    ("Blouse Wanita", "Respon penjual tidak ramah saat dikomplain barang cacat, sangat mengecewakan.", "negatif"),
    ("Hoodie Polos", "Tali hoodie tidak ada dan kantong bolong rusak, kendali mutu toko kurang bagus.", "negatif"),
    ("Celana Jeans", "Ukuran celana terlalu sempit tidak sesuai panduan size chart, kurang teliti.", "negatif"),
    ("Kaos Oversize", "Bahan tipis panas gerah, jahitan kerah miring kurang rapi dan cepat melar.", "negatif"),
    ("Dress Floral", "Dress sobek rusak di bagian samping, kain tipis sekali tidak seperti foto.", "negatif"),
    ("Kemeja Flanel", "Pengemasan paket hanya plastik tipis rusak robek di jalan, barang kurang aman.", "negatif"),
    ("Jaket Denim", "Bau apek menyengat dan bahan kaku keras, kurang nyaman dipakai seharian.", "negatif"),
    ("Hoodie Polos", "Bahan hoodie berbulu rontok setelah dicuci, kualitas bahan sangat kurang.", "negatif"),
    ("Celana Jeans", "Jahitan saku celana jebol lepas, kualitas benang jelek dan kurang kokoh.", "negatif"),
    ("Kaos Oversize", "Kerah kaos melar setelah dicuci, bahan kasar kurang nyaman dan gerah.", "negatif"),
    ("Dress Floral", "Motif tidak sesuai pesanan, dikirim motif lain tanpa konfirmasi, kurang profesional.", "negatif"),
    ("Kemeja Flanel", "Kancing kemeja kurang satu dan lubang kancing belum dipotong dengan rapi.", "negatif"),
    ("Jaket Denim", "Jaket terlalu pendek tidak sesuai spesifikasi ukuran, kurang puas sama sekali.", "negatif"),
    ("Blouse Wanita", "Kain blouse panas tidak menyerap keringat, potongan miring kurang bagus.", "negatif"),
    ("Hoodie Polos", "Warna hoodie belang dan kotor seperti barang returan lama, kurang teliti.", "negatif"),
    ("Celana Jeans", "Bagian selangkangan celana sobek rusak saat dicoba pertama kali, bahan rapuh.", "negatif"),
    ("Kaos Oversize", "Bahan sangat tipis terawang, ukuran lengan kurang panjang beda dari foto.", "negatif"),
    ("Dress Floral", "Jahitan resleting rusak tersangkut kain, tidak bisa dipakai sama sekali.", "negatif"),
    ("Kemeja Flanel", "Kemeja ada bolong kecil di punggung, pengecekan barang sebelum kirim kurang.", "negatif"),
    ("Jaket Denim", "Warna pudar tidak pekat seperti di gambar, kualitas jaket denim kurang oke.", "negatif"),
    ("Hoodie Polos", "Customer service kurang ramah dan pengiriman sangat molor tidak sesuai estimasi.", "negatif"),
    ("Celana Jeans", "Pola celana aneh dan kurang proporsional, dipakai duduk terasa sesak.", "negatif"),
    ("Kaos Oversize", "Baju kaos cacat ada noda tinta hitam tidak hilang dicuci, sangat kecewa.", "negatif"),
    ("Dress Floral", "Panjang dress kurang sesuai deskripsi, terlalu pendek dan bahan agak panas.", "negatif"),
    ("Kemeja Flanel", "Bahan menyusut drastis setelah dicuci, sekarang tidak muat lagi kurang awet.", "negatif"),
    ("Jaket Denim", "Respon penjual lambat berhari-hari saat komplain barang cacat rusak.", "negatif"),
    ("Blouse Wanita", "Jahitan pinggir lepas brodol semua, pengerjaan baju kurang rapi mengecewakan.", "negatif"),
    ("Hoodie Polos", "Kain hoodie tipis dan kasar bergesekan di leher, kurang nyaman dipakai.", "negatif"),
    ("Celana Jeans", "Kancing pengait lepas rusak, lubang kancing kekecilan kurang teliti dibuat.", "negatif"),
    ("Kaos Oversize", "Warna kaos cepat luntur dan pudar, kualitas katun kurang bermutu.", "negatif"),
    ("Dress Floral", "Baju ada sobekan kecil di jahitan bawah, pengawasan kualitas sangat kurang.", "negatif"),
    ("Kemeja Flanel", "Ukuran dada terlalu sempit padahal sudah pesan ukuran XL, size chart kurang akurat.", "negatif"),
    ("Jaket Denim", "Denim kaku sekali dan berbau zat kimia menyengat, kurang nyaman di badan.", "negatif"),
    ("Blouse Wanita", "Bahan nerawang sekali harus pakai dalaman lagi, kualitas kurang memuaskan.", "negatif"),
    ("Hoodie Polos", "Jahitan tangan hoodie miring dan berkerut kurang rapi, sangat mengecewakan.", "negatif"),
    ("Celana Jeans", "Barang tidak sesuai deskripsi, penjual kurang ramah saat dimintai retur.", "negatif"),
    ("Kaos Oversize", "Kain terasa gatal di kulit, jahitan tepi rusak lepas benang di mana-mana.", "negatif"),
    ("Dress Floral", "Respon admin acuh dan lambat sekali, baju datang dalam kondisi sobek rusak.", "negatif"),
    ("Kemeja Flanel", "Pengiriman telat lama, kemasan basah dan baju kena noda kotoran kurang rapi.", "negatif"),
    ("Jaket Denim", "Bahan jaket tipis seperti kemeja bukan jaket denim asli, sangat kurang bagus.", "negatif"),
    ("Blouse Wanita", "Potongan leher asimetris dan jahitan benang berantakan kurang rapi.", "negatif"),
    ("Hoodie Polos", "Sablon tulisan di hoodie terkelupas rusak saat pertama kali dibuka paketnya.", "negatif"),
    ("Celana Jeans", "Panjang kaki kanan dan kiri beda panjangnya, potongan produk kurang teliti.", "negatif"),
    ("Kaos Oversize", "Ukuran sangat kecil tidak oversize sama sekali, penamaan produk kurang sesuai.", "negatif"),
]

# We combine base_reviews + subset of pos + subset of neg to reach exactly 150 rows
# Typically e-commerce sentiment is ~60% pos / ~40% neg (e.g. 90 pos, 60 neg)
all_data = []

# Add base reviews first
all_data.extend(base_reviews)

# Total needed: 150
# Currently base_reviews has 2 pos, 3 neg.
# We need 88 more pos and 57 more neg to get 90 pos, 60 neg (total 150)
selected_pos = reviews_pos[:88]
selected_neg = reviews_neg[:57]

for item in selected_pos:
    all_data.append(item)
for item in selected_neg:
    all_data.append(item)

# Build DataFrame
df = pd.DataFrame(all_data, columns=["produk", "komentar", "sentimen"])
df.insert(0, "id", range(1, len(df) + 1))

# Save to CSV
df.to_csv("dataset_ulasan.csv", index=False, encoding="utf-8")
print(f"Dataset successfully created with {len(df)} rows.")
print("Sentiment distribution:")
print(df["sentimen"].value_counts())
