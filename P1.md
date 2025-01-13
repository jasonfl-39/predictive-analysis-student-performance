# Laporan Proyek Machine Learning - Jason Filbert Leo

## Domain Proyek

Dalam dunia pendidikan, performa siswa merupakan hal yang penting untuk dimaksimalkan sehingga siswa dapat  belajar dengan optimal dan mendapatkan hasil pembelajaran yang memuaskan. Masalah penentuan performa siswa merupakan suatu masalah kompleks dengan banyak terdapatnya faktor yang dapat menyebabkan perbedaan di antara siswa satu dan lainnya, bahkan saat setiap siswa mendapatkan prasarana pembelajaran dan kualitas materi yang sama. Untuk menentukan performa siswa, dilakukan analisis prediksi dengan data kuantitatif dengan menentukan faktor yang paling berpengaruh terhadap performa siswa.
Dengan analisis prediksi ini, diharapkan bahwa stakeholders yaitu institusi pendidikan, guru, orang tua dan siswa dapat mengetahui faktor apa  saja yang dapat meningkatkan performa akademik siswa. Untuk mencapai tujuan tersebut, hasil dari prediksi ini harus seakurat mungkin dan dapat dijelaskan oleh faktor-faktor yang dianalisis, sehingga dapat dipahami alasannya dan menjadi bermanfaat untuk mengoptimalisasikan performa siswa kedepannya. (Alamri & Alharbi, 2021)  

Untuk mendapatkan hasil prediksi yang seakurat mungkin, akan dibandingkan performa dari tiga algoritma, yaitu Random Forest, K-Nearest, dan Boosting pada keakuratan prediksi performa siswa. Ketiga algoritma tersebut akan ditraining dan diuji pada dataset yang sama sehingga dapat dibandingkan secara langsung tingkat keakuratan setiap algoritma.

 Referensi : [R. Alamri and B. Alharbi, "Explainable Student Performance Prediction Models: A Systematic Review," in IEEE Access, vol. 9, pp. 33132-33143, 2021, doi: 10.1109/ACCESS.2021.3061368](https://ieeexplore.ieee.org/abstract/document/9360749)
 
## Business Understanding
Untuk mendalami masalah yang akan dibahas pada proyek ini, maka dibahas terlebih dahulu pernyataan dan tujuan dari masalah yang dihadapi.

### Problem Statements

Pernyataan masalah latar belakang pada proyek ini adalah:
- Bagaimana cara performa siswa dapat ditentukan dengan seakurat mungkin?
- Bagaimana fitur-fitur yang dapat memengaruhi performa siswa dan efeknya dapat ditentukan?
  

### Goals

Tujuan dari pernyataan masalah pada proyek ini adalah:
- Menentukan performa siswa berdasarkan pada data dengan seakurat mungkin
  Pada proyek ini, berdasarkan data performa siswa diukur dengan suatu indeks performa. Indeks ini merepresentasikan prestasi akademik setiap siswa, di mana saat nilai indeks semakin tinggi menunjukkan performa yang lebih tinggi pula. Indeks ini, beserta dengan setiap fitur-fitur lainnya pada data kemudian akan ditraining dan divalidasi pada suatu model untuk memprediksi nilai indeks performa. Model akan menggunakan tiga jenis algoritma, yaitu Random Forest, K-Nearest, dan Boosting, dan membandingkan hasil prediksi setiap algoritma dengan data validasi sehingga dapat diperoleh hasil prediksi performa siswa yang paling akurat.
- Menentukan fitur-fitur yang dapat memengaruhi performa siswa beserta efeknya
  Untuk menganalisis pengaruh setiap metrik pada data dengan performa siswa, akan dilakukan analisis multivariate untuk melihat bagaimana hubungan plot setiap fitur, baik yang bersifat kategori atau numerik, dengan indeks performa siswa. Selanjutnya, diperiksa matriks korelasi antara fitur-fitur numerik dengan indeks performa untuk menentukan fitur numerik apa saja yang memiliki pengaruh besar pada performa siswa, dan juga jika pengaruh tersebut berbanding lurus atau terbalik dengan performa siswa.

## Data Understanding
Pada proyek ini, digunakan dataset sintetik regresi logistik berisi 10.000 sampel dari Kaggle: [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression?resource=download)

### Variabel-variabel pada Dataset
Dalam dataset ini, terdapat variabel-variabel berikut:
- `Hours Studied`: Jumlah total jam belajar setiap siswa.
- `Previous Scores`: Nilai siswa pada tes sebelumnya.
- `Extracurricular Activities`: Bernilai `Yes` jika siswa mengikuti kegiatan ekstrakurikuler dan `No` jika tidak.
- `Sleep Hours`: Rata-rata jam tidur siswa setiap harinya.
- `Sample Question Papers Practiced`: Jumlah lembar soal sampel yang dilakukan siswa.
- `Performance Index`: Pengukuran keseluruhan performa siswa. Indeks ini merepresentasikan performa akademik yang telah dibulatkan ke integer terdekat. Nilai indeks berada di antara 10 sampai 100, dengan nilai lebih tinggi menunjukkan performa lebih baik. Variabel ini merupakan variabel tujuan yang dianalisis dan diprediksi.

### Visualisasi Data dan Analisis Eksplorasi Data
Untuk mengetahui bagaimana hubungan antara variabel-variabel dengan variabel tujuan `Performance Index`, maka dilakukan analisis multivariate pada fitur kategori dan fitur numerik.
Pada fitur kategori, dibentuk bar chart untuk melihat bagaimana pengaruh fitur terhadap `Performance Index`.
Diperoleh hasil:

![categorical_feature](https://github.com/user-attachments/assets/a031332e-9562-4a3d-baeb-c367a42f1905)

Karena tidak ada perbedaan besar antara rata-rata `Performance Index` antara siswa yang mengikuti dan tidak mengikuti kegiatan ekstrakurikuler, maka dapat disimpulkan bahwa pengaruh rendah terhadap `Performance Index`.
Pada fitur numerik, dibentuk plot untuk menentukan hubungan antar fitur numerik.
Diperoleh hasil:

![numerical_feature](https://github.com/user-attachments/assets/3ba99bca-44e1-43a4-9fc7-7df9e31b91ec)


Terlihat bahwa fitur `Previous Scores` memiliki korelasi positif dengan `Performance Index`.
Selanjutnya dibentuk matriks korelasi antar fitur numerik.
Diperoleh hasil:

![correlation_matrix](https://github.com/user-attachments/assets/10ca5c55-deaf-4b13-ad66-555416aff597)

Karena nilai korelasi antara fitur `Previous Scores` dengan `Performance Index` mencapai >0.92, hal ini menunjukkan adanya korelasi tinggi di antara kedua fitur ini.

## Data Preparation
### Encoding Fitur Kategori
Encoding pada fitur berjenis kategori, `Extracurricular Activities` dilakukan dengan one-hot encoding untuk mendapatkan fitur baru untuk merepresentasikan variabel kategori. Hal ini dilakukan supaya fitur ini dapat digunakan oleh model.
Snippet kode:
```
from sklearn.preprocessing import  OneHotEncoder
dataframe = pd.concat([dataframe, pd.get_dummies(dataframe['Extracurricular Activities'], prefix='Extracurricular Activities')],axis=1)
dataframe.drop(['Extracurricular Activities'], axis=1, inplace=True)
dataframe.head()
```
Diperoleh fitur baru `Extracurricular_Activities_No` dan `Extracurricular_Activities_Yes`.

![encoding](https://github.com/user-attachments/assets/786d76d9-b226-40bb-8ee1-4dd10a419d5d)

### Train-Test-Split
Untuk memastikan data yang digunakan untuk training model tidak mengontaminasi data pada saat testing model, maka dilakukan pembagian dataset menjadi train set dan test set dengan ratio 80:20 mengingat terdapat 10000 sampel data.
Snippet kode:
```
from sklearn.model_selection import train_test_split
 
X = dataframe.drop(["Performance Index"],axis =1)
y = dataframe["Performance Index"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```
Diperoleh pembagian data sebagai berikut:

![split](https://github.com/user-attachments/assets/621bf7f0-7d6f-4e2f-a3db-272536026ecb)

### Standardisasi Fitur Numerik
Untuk mempermudah pengolahan fitur-fitur numerik oleh model, dilakukan standardisasi fitur.
Snippet kode:
```
from sklearn.preprocessing import StandardScaler
num_features = num_features[:-1]
scaler = StandardScaler()
scaler.fit(X_train[num_features])
X_train[num_features] = scaler.transform(X_train.loc[:, num_features])
```
Diperoleh hasil:

![standardization](https://github.com/user-attachments/assets/6f7bda9e-ff1f-42d9-8045-4dfe3a3049e9)

## Modeling
Digunakan tiga jenis algoritma pada model machine learning, yaitu Random Forest, K-Nearest, dan Boosting Algorithm. Ketiga algoritma ini akan dibandingkan performanya sehingga dapat ditentukan model terbaik diantara ketiganya.
Pada algoritma K-Nearest (KNN), diambil parameter `n_neighbor = 10`.
Snippet kode:
```
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
```
Algoritma ini bekerja dengan cara membandingkan fitur diantara beberapa data yang saling berdekatan. Hasil prediksi ditentukan dari perhitungan jarak di antara titik-titik data yang berdekatan, di mana pada model ini digunakan jarak euclidean. Jumlah data saling berdekatan yang dibandingkan diberikan oleh parameter `n_neighbors`.

Pada algoritma Random Forest (RF), diambil parameter `n_estimators=50`, `max_depth=16`, `random_state=55`, `n_jobs=-1`
Snippet kode:
```
from sklearn.ensemble import RandomForestRegressor
 
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)  
```
Algoritma ini bekerja dengan cara mencari nilai mean dari hasil prediksi sejumlah decision tree yang masing-masing bekerja sendiri secara paralel, di mana setiap decision tree menggunakan sampel acak dari bagian-bagian berbeda dari dataset. Jumlah decision tree yang digunakan pada model diberikan oleh parameter `n_estimators`, kedalaman setiap decision tree diberikan oleh `max_depth`, dan pekerjaan yang dilakukan secara paralel diberikan oleh `n_jobs`.

Pada algoritma Boosting, diambil parameter `learning_rate=0.1`, `random_state=55`
Snippet kode:
```
from sklearn.ensemble import GradientBoostingRegressor

boosting = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```
Algoritma ini bekerja dengan cara melakukan training kembali pada model regresi pertama sehingga model dapat terus memperbaiki kemampuan prediksinya pada setiap kali training sehingga diperoleh model yang lebih akurat. Bobot dari setiap iterasi proses regresi diberikan oleh parameter`learning_rate`, jumlah iterasi yang dilakukan diberikan oleh parameter `n_estimators`.

## Evaluation
Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE) pada masalah regresi logistik ini untuk membandingkan error di antara hasil prediksi setiap algoritma dengan nilai data validasi. Metrik ini diambil karena model memproses data regresi logistik multivariat sehingga MSE menjadi metrik yang dapat menunjukkan tren dari kesuluruhan data secara intutitif. Saat nilai MSE semakin kecil, hal ini mengindikasikan bahwa akurasi model lebih tinggi sehingga dapat memprediksi nilai performa siswa dengan lebih akurat berdasarkan pada fitur-fitur data seperti dijabarkan pada bagian Data Understanding.
Dari hasil evaluasi model diperoleh bar chart dari nilai MSE berikut:

![prediction](https://github.com/user-attachments/assets/35f53e3f-2e85-4bf6-acda-a7289c53be80)

Terlihat bahwa algoritma RF memberikan error MSE terkecil di antara ketiga algoritma.
Untuk hasil prediksi dari model, terlihat pula bahwa algoritma RF memberikan hasil prediksi performa siswa terdekat dengan nilai asli pada data validasi:

![evaluation](https://github.com/user-attachments/assets/6108393f-2fea-4249-aa75-b9d9ddd51b9f)

Maka dapat disimpulkan bahwa algoritma RF merupakan algoritma terbaik untuk memprediksi performa siswa dengan seakurat mungkin pada dataset yang diberikan.
Selanjutnya, berdasarkan pada penjabaran dari Data Understanding, dapat dijelaskan bahwa faktor yang paling berpengaruh pada performa siswa adalah nilai siswa pada masa ujian sebelumnya. Sementara faktor seperti kegiatan ekstrakurikuler, jam belajar, dan jam tidur tidak memiliki pengaruh signifikan pada performa siswa. Hal ini sebenarnya tidak menutup kemungkinan bahwa terdapat faktor luar yang tidak diliputi dalam dataset sebagai penyebab mengapa terdapat korelasi tinggi antara performa siswa dengan performa sebelumnya. Jadi, faktor-faktor yang menentukan performa siswa belum sepenuhnya dapat dijelaskan dengan dataset yang digunakan, tetapi diperoleh bahwa performa siswa pada umumnya konsisten pada setiap periode pembelajaran dan tidak terpengaruh secara signifikan oleh faktor seperti kegiatan ekstrakurikuler, jam belajar, dan jam tidur.
