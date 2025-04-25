# Laporan Proyek Machine Learning - Kornelius Setiawan

## Domain Proyek

Pendidikan adalah fondasi utama dalam pengembangan sumber daya manusia. Proyek ini berfokus pada prediksi Performance Index murid menggunakan model regresi berdasarkan beberapa faktor/variabel seperti jam belajar, nilai sebelumnya, kegiatan ekstrakurikuler, jam tidur, dan jumlah soal latihan yang dikerjakan. Solusi ini dapat membantu institusi pendidikan mengidentifikasi murid yang membutuhkan intervensi khusus dan merancang strategi pembelajaran yang lebih efektif

[Factors Affecting Student Academic Performance](https://ijonse.net/index.php/ijonse/article/view/276)
[Factors affecting students’ academic performance and teachers](https://www.tandfonline.com/doi/full/10.1080/23311983.2024.2412944#abstract)

## Business Understanding
### Problem Statements
-  Bagaimana cara memprediksi Performance Index siswa dengan performa prediksi yang tinggi menggunakan fitur-fitur yang tersedia?
-  Metode apa yang paling efektif untuk meminimalkan error prediksi, seperti MSE, dalam model regresi?

### Goals
- Mengembangkan beberapa model regresi dengan error prediksi yang rendah
- Menguji dan membandingkan setidaknya dua algoritma regresi berdasarkan performa mereka dalam Mean Squared Error (MSE).

### Solution statements
- Menggunakan algoritma **Random Forest Regressor**, **Linear Regression**, **K-Nearest Neighbor** untuk memprediksi Performance Index
- Evaluasi menggunakan MSE, kemudian pilih model terbaik berdasarkan nilai metrik tersebut.

## Data Understanding
Dataset ini dirancang untuk menganalisis faktor-faktor yang mempengaruhi performance akademik siswa. Dataset yang digunakan berisi 6 kolom dan 10000 baris dengan berbagai fitur seperti Hours Studied, Previous Score, Extracurricular Activities, Sleep Hours, dan Sample Question Papers Practiced. Fitur-fitur tersebut adalah fitur yang akan digunakan dalam menemukan pola pada data, sedangkan Performance Index merupakan fitur target. 
Berikut adalah sumber atau tautan dari [dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data)

Kondisi dataset: 
- Tidak ada ditemukan missing value pada dataset.
- Ditemukan duplikasi data dalam dataset, yaitu sebanyak 127 data. Namun, telah dihapus menggunakan df.drop_duplicates().
- Tidak ditemukan outlier pada fitur-fitur tersebut.



### Variabel-variabel pada dataset ini adalah sebagai berikut:
- **Hours Studied** : Jumlah total jam yang dihabiskan oleh setiap siswa untuk belajar.
- **Previous Scores** : Nilai yang diperoleh siswa pada ujian-ujian sebelumnya.
- **Extracurricular Activities** : Apakah siswa berpartisipasi dalam kegiatan ekstrakurikuler (Yes or No).
- **Sleep Hours** : Rata-rata jumlah jam tidur yang dimiliki siswa per hari.
- **Sample Question Papers Practiced** : Jumlah soal latihan yang telah dikerjakan oleh siswa.
- **Performance Index** : Ukuran kinerja keseluruhan dari setiap siswa. Indeks kinerja ini mewakili kinerja akademik siswa dan telah dibulatkan ke angka terdekat. Indeks ini memiliki rentang nilai dari 10 hingga 100, dengan nilai yang lebih tinggi menunjukkan kinerja yang lebih baik.


### **Exploratory Data Analysis**:
**Correlation Matrix**
![Correlation Matrix](https://res.cloudinary.com/daoqz3rdr/image/upload/v1745551581/correlation_matrix_qk1gak.png)
Fitur **Previous Score** memiliki skor korelasi positif yang besar (di atas 0.9) dengan fitur target **Performance Index**.

## Data Preparation

### Drop Duplicate value
Pada bagian ini, dilakukan identifikasi dan menghapus nilai duplikat dalam dataset untuk memastikan bahwa data yang digunakan bersih dan bebas dari pengulangan yang tidak perlu.
```
print(f"Number of duplicated: {df.duplicated().sum()}")
df = df.drop_duplicates()
```

### Encoding Fitur Kategori
Pada bagian ini, dilakukan encoding pada kolom `Extracurricular Activities` untuk mengubah data kategorikal menjadi data numerik. Ini penting karena banyak algoritma machine learning memerlukan input numerik untuk melakukan perhitungan dan prediksi.
```
from sklearn.preprocessing import StandardScaler, LabelEncoder
encoder = LabelEncoder()

df["Extracurricular Activities"] = encoder.fit_transform(df["Extracurricular Activities"])
```

### Train Test Split
Pada bagian ini, dataset dibagi menjadi dua bagian: 80% data pelatihan (training data) dan 20% data pengujian (test data). Pembagian ini penting untuk mengevaluasi kinerja model secara objektif dengan menguji model pada data yang tidak pernah dilihat sebelumnya.
```
from sklearn.model_selection import train_test_split

X = df.drop(["Performance Index"], axis=1)
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Standarisasi
Pada bagian ini, standarisasi dilakukan pada beberapa fitur numerik menggunakan `StandardScaler`. Proses ini sangat penting dalam model machine learning, terutama ketika fitur memiliki skala yang berbeda. StandardScaler mengubah data sehingga memiliki rata-rata 0 dan deviasi standar 1
```
features = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]

scaler = StandardScaler()
X_train[features] = scaler.fit_transform(X_train[features])
```

## Modeling

### Linear Regression
Linear Regression adalah model statistik yang mencoba menemukan hubungan linear antara fitur dan target. Model ini bekerja dengan menghitung koefisien dan intersep dari setiap fitur untuk meminimalkan selisih antara nilai prediksi dan nilai sebenarnya.

 Parameter :
    - Menggunakan semua parameter default dari sklearn.linear_model.LinearRegression.
```
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
```
Kelebihan: Interpretasi koefisien yang jelas, tidak terlalu rawan overfitting pada dataset kecil.  
Kekurangan: Tidak efektif untuk hubungan non-linear.  

### Random Forest Regressor
Random Forest adalah ensemble model berbasis Decision Trees. Model ini membangun beberapa pohon keputusan (tree) di mana masing-masing dilatih pada subset acak dari data dan fitur. Prediksi akhir didapatkan dari rata-rata hasil semua pohon.
Hal ini mengurangi risiko overfitting yang sering terjadi pada decision tree tunggal dan meningkatkan stabilitas prediksi.

Parameter : 
    - n_estimators=50 : Jumlah pohon dalam hutan. Semakin banyak, biasanya prediksi lebih stabil.
    - max_depth=30 : Batas kedalaman tiap pohon. Mencegah overfitting.
    - random_state=55 : Seed random untuk memastikan hasil eksperimen bisa direproduksi (konsisten setiap run).
    - n_jobs=-1 : Menggunakan seluruh core CPU yang tersedia untuk paralelisasi.
```
from sklearn.ensemble import RandomForestRegressor
 
RF = RandomForestRegressor(n_estimators=50, max_depth=30, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
```
Kelebihan: Menangkap non‑linearitas dengan baik.
Kekurangan: Perbedaan besar antara train dan test MSE menunjukkan overfitting.

### K-Nearest Neighbors 
Cara kerja KNN yaitu mencari titik data paling dekat (neighbors) dengan data yang ingin diprediksi berdasarkan jarak Euclidean. Kemudian, target prediksi dihitung sebagai rata-rata nilai target dari tetangga-tetangga tersebut.
Parameter :
    - n_neighbors=20 : Menentukan jumlah tetangga terdekat (k) yang digunakan saat melakukan prediksi, dalam hal ini yaitu 20. Nilai prediksi akan jadi rata-rata dari target tetangga-tetangga ini. Semakin besar k, prediksi jadi lebih halus tapi kurang sensitif terhadap variasi lokal.
```
from sklearn.neighbors import KNeighborsRegressor
 
knn = KNeighborsRegressor(n_neighbors=20)         
knn.fit(X_train, y_train)
```

Kelebihan: Mudah diimplementasi.
Kekurangan: Sensitif pada skala fitur dan outlier, performa relatif buruk pada kasus ini.

### **Pemilihan Model**
Model `Linear Regression` dipilih sebagai model final karena:
- Test MSE lebih rendah daripada Random Forest dan KNN.  
- Tidak overfit sehingga lebih reliable untuk data baru.  


## Evaluation
**Metrik Evaluasi yang Digunakan :**
Model dievaluasi menggunakan Mean Squared Error (MSE), yang mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. MSE lebih sensitif terhadap error besar, sehingga cocok digunakan untuk mendeteksi seberapa baik model dalam membuat prediksi yang akurat.

**Pembahasan Hasil**
| Model             | Train MSE  | Test MSE   |
|-------------------|------------|------------|
| Linear Regression | 4.144575   | 4.305901   |
| Random Forest     | 0.970478   | 5.652286   |
| KNN               | 7.52006    | 8.344308   |
- `Linear Regression` memberikan Test MSE terendah (4.3059), artinya error rata‑rata kuadratnya paling kecil di antara ketiga.
- `Random Forest Regression` tampil sangat baik di data train (MSE 0.97) tetapi mengalami overfitting sehingga Test MSE naik menjadi 5.65.
- `KNN` memiliki performa paling lemah dengan Test MSE 8.34.

Dari hasil evaluasi tersebut, model Linear Regression menunjukkan performa terbaik dengan nilai Test MSE sebesar 4.30, mengungguli Random Forest (5.65) dan K-Nearest Neighbors (8.34). Random Forest memang menghasilkan MSE yang sangat rendah pada data latih (0.97), namun performanya menurun drastis pada data uji, menunjukkan indikasi overfitting. Sementara itu, KNN memberikan hasil prediksi dengan MSE tertinggi, yang menandakan bahwa pendekatan berbasis tetangga kurang efektif pada karakteristik data ini.

Jika dikaitkan dengan bagian Business Understanding, hasil ini berhasil menjawab kedua problem statement yang telah ditetapkan. Pertama, proyek ini telah berhasil menemukan pendekatan yang dapat memprediksi Performance Index siswa dengan performa tinggi dan kesalahan prediksi (MSE) yang rendah, yaitu melalui model Linear Regression. Kedua, dengan menguji dan membandingkan tiga algoritma yang berbeda, ditemukan bahwa metode Linear Regression adalah yang paling efektif dalam menurunkan nilai MSE secara konsisten pada data pelatihan maupun data uji.

Seluruh goals dari proyek ini juga tercapai dengan baik. Proyek telah berhasil membangun beberapa model regresi dan melakukan evaluasi MSE untuk memilih model terbaik. Selain itu, semua solution statement yang direncanakan sebelumnya terbukti memberikan dampak. Penggunaan ketiga model : Linear Regression, Random Forest, dan KNN membantu memberikan gambaran yang komprehensif terkait efektivitas masing-masing pendekatan terhadap data. Evaluasi menggunakan MSE sebagai dasar pengambilan keputusan juga terbukti memberikan hasil yang objektif dan terukur.

Kesimpulannya, model Linear Regression tidak hanya menjadi model dengan kinerja terbaik, tetapi juga menjawab tantangan utama dalam proyek ini karena dapat memprediksi performa siswa dengan akurat dan andal, sehingga dapat digunakan untuk mendukung keputusan strategis dalam dunia pendidikan.
