## Business Understanding
Jaya Jaya Institut mengalami tantangan serius dengan tingginya tingkat mahasiswa yang Dropout sebelum menyelesaikan studi. Hal ini berdampak pada reputasi institusi, efisiensi penggunaan sumber daya, serta kesejahteraan mahasiswa.
Pihak kampus ingin memahami faktor-faktor utama yang menyebabkan mahasiswa Dropout agar dapat merancang kebijakan dan intervensi yang tepat.

Dropout siswa yang tinggi bisa menjadi indikator adanya masalah dalam perkembangan instutusi kedepannya, seperti kurangnya motivasi untuk belajar dengan sungguh-sungguh, menyepelekan tugas dan kewajiban siswa, kesempatan memberikan sumber daya yang berkualitas pun jadi berkuran, atau lebih bahayanya lagi reputasi dari institut akan tercemar.

Manajemen Institut ingin memahami faktor utama yang menyebabkan tingginya dropout agar dapat mengambil tindakan strategis untuk mengurangi dropout dan mempertahankan talenta terbaik dan dapat memberikan manfaat kepada masyarakat.

## Permasalahan Bisnis
Dropout rate adalah persentase mahasiswa yang keluar dari sistem pendidikan sebelum lulus. Tingginya angka ini menunjukkan potensi masalah dalam dukungan akademik, finansial, maupun personal yang dihadapi mahasiswa.

### **Cakupan Proyek**
Proyek ini menganalisis data historis mahasiswa untuk menemukan pola-pola dan karakteristik umum dari mahasiswa yang cenderung mengalami Dropout, serta menyediakan alat bantu pemantauan berbasis dashboard.

**Batasan Proyek:**
- Hanya menggunakan data internal dari kampus.
- Rekomendasi hanya bersifat strategis, tidak mencakup implementasi teknis.
- Visualisasi dan pemantauan dilakukan melalui Metabase.

### **Tujuan Proyek**
1. Mengidentifikasi fitur-fitur utama yang berkontribusi terhadap status Dropout.
2. Mengelompokkan karakteristik umum mahasiswa yang keluar.
3. Menyediakan dashboard interaktif agar pihak kampus dapat memonitor faktor penyebab Dropout secara real time.

### **Output Akhir**
- **Dashboard Interaktif (Metabase)** → Memvisualisasikan faktor-faktor penting dropout.
- **Model Prediksi Dropout** →  Model yang dapat Memprediksi kemungkinan seorang mahasiswa akan Dropout.
- **Rekomendasi Action** → Solusi berbasis data untuk mengurangi dropout seperti Intervensi berbasis data.

### **Teknologi yang Digunakan**
1. **Metabase** → Untuk dashboard visualisasi data.
2. **PostgreSQL** → Untuk menganalisis data menggunakan SQL menyimpan data dan query data.
3. **Python** →Untuk preprocessing data dan modelling prediktif.

### **Data yang Digunakan**
Dataset yang digunakan mencakup berbagai faktor demografis dan indikator terkait karyawan:

| Fitur                 | Deskripsi                                      |
|-----------------------|----------------------------------------------|
| Admission_grade           | Nilai saat masuk                             |
| Curricular_units_*            | Jumlah mata kuliah diambil, disetujui, gagal, dst.                        |
| Scholarship_holder                  | Apakah menerima beasiswa                               |
| Debtor               | Apakah memiliki utang                                |
| Marital_status        | Status pernikahan                           |
| Tuition_fees_up_to_date           | Apakah membayar UKT tepat waktu                   |
| Displaced, Educational_special_needs              | Status sosial atau kebutuhan khusus                             |
| Mother's/Father's occupation             | Pekerjaan orang tua                   |
| Mother's/Father's qualification        | Tingkat pendidikan orang tua                          |
| Age    | Umur Siswa           |
| Gender      | Jenis Kelamin Siswa                    |
| Unemployment_rate            | Tingkat pengangguran saat tahun masuk            |
| Inflation_rate     | Tingkat inflasi saat tahun masuk             |
| GDP    | Produk Domestik Bruto saat tahun masuk                       |
| Status       | Status akhir mahasiswa (Dropout, Enrolled, Graduate)                   |

### **Setup dan Konfigurasi**
#### **1. Menjalankan Metabase dengan Docker:**
```
docker pull metabase/metabase:v0.46.4
docker run -p 3000:3000 --name metabase metabase/metabase
```
#### **2. Menyiapkan Database di Google Colab:**
```
from sqlalchemy import create_engine
URL = "DATABASE_URL"
engine = create_engine(URL)
df.to_sql('education', engine)
```
#### **3. Menjalankan Analisis di Google Colab:**
Gunakan Google Colab untuk menjalankan file **.ipynb** dan **.py**, serta mengunggah model prediksi untuk analisis lebih lanjut.

#### **4. Deploy Model di Streamlit:**
1. Pastikan Anda telah membuat berkas requirement.txt dan mengupload proyek ke dalam repository github.
2. Buat akun pada streamlit sesuai panduan berikut: Sign up for Streamlit Community Cloud. Anda disarankan untuk melakukan sign up menggunakan akun github.
3. Pada tampilan awal, klik “New app”.
4. Pilih repository yang memuat kode streamlit app (app.py) beserta dependencies-nya (helper function, object model dan data preprocessing). Kemudian masukkan juga nama berkas streamlit app.
5. Terakhir klik “Deploy!” dan tunggu proses deployment selesai.

Link Deploy : https://educationds-amienramdhani.streamlit.app/ 
### **Dashboard Monitoring**
![Dashboard](https://i.ibb.co.com/5XfdXzwj/Screenshot-300.png "Dashboard")

**Akses Dashboard Metabase:**
[Link Metabase](https://drive.google.com/file/d/1XhoSO7on8UOrifsvj2dUoUvub2fU7TqI/view?usp=sharing)
- **Email:** root@mail.com
- **Password:** root123
- 
### **Kesimpulan dan Rekomendasi**
Berdasarkan analisis, ditemukan bahwa:
1. Mahasiswa dengan **nilai rendah**, **utang**, atau **tidak membayar UKT tepat waktu** lebih berisiko Dropout.
2. **Faktor sosial** seperti **tidak menerima beasiswa** atau **berasal dari keluarga dengan pendidikan rendah** juga berkontribusi terhadap dropout.
3. Mahasiswa yang **tidak mengikuti evaluasi di semester awal** memiliki dropout rate lebih tinggi.
4. Mahasiswa yang tidak lulus SKS nya banyak mengalami Dropout.

#### **Rekomendasi Actions:**
1. **Pendampingan Akademik** untuk mahasiswa dengan nilai masuk rendah atau gagal banyak mata kuliah.
2. **Skema Pembayaran Fleksibel** atau **bantuan keuangan** untuk mahasiswa dengan kendala ekonomi.
3. **Monitoring Awal** terhadap mahasiswa dengan **status sosial khusus**.
4. **Adakan tambahan pembelajaran persiapan ujian** agar siswa dapat mengikuti ujian dan mendapatkan nilai yang memuaskan
5. **Sosialisasi Beasiswa** dan transparansi proses pengajuannya.

Dengan memahami faktor-faktor ini, Institut dapat mengembangkan kebijakan yang lebih efektif untuk mempertahankan talenta terbaik mereka dan menjamin kualitas pendidikan institut jadi lebih maju dan baik.