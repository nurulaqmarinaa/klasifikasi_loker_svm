import re
import pandas as pd
import streamlit as st
import pickle
import numpy as np
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

# Inisialisasi stemmer untuk proses stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Muat model prediksi
models = pickle.load(open("model_svm.sav", "rb"))

# Muat model TFIDF
vectorizer = pickle.load(open("tfidf_model.sav", "rb"))

# Aplikasi Streamlit
st.title("Aplikasi Prediksi Lowongan kerja")
st.write("Masukkan deskripsi lowongan kerja untuk diprediksi")
# Input pengguna
Text = st.text_area("Masukkan deskripsi:", height=100)

#preprocessing data
def preprocess(Text):
    # Cleaning
    Text = re.sub(r'http\S+', '', Text)
    Text = re.sub('@[\w]*', '', Text)
    Text = re.sub(r'#[A-Za-z0-9_]+', '', Text)
    Text = re.sub(r'[\S]+\.(net|com|org|gov|id|co|COM|ID)[\S]*\s?', '', Text)
    Text = re.sub(r'\b\w{1,2}\b', '', Text)
    Text = re.sub(r'^b[\s]+', '', Text)
    Text = re.sub(r'[,-/]', ' ', Text)
    Text = re.sub('\n ', '', Text)
    Text = re.sub(r"\b[a-zA-Z]\b", "", str(Text))
    Text = re.sub(r"[^\w\s]", " ", str(Text))
    Text = re.sub('\s+', ' ', Text)

    # Case Folding
    Text = Text.lower()

    # Tokenization
    token = word_tokenize(Text)

    # Slangwords
    kamus_slangword = pd.read_csv("slangword.txt")
    kata_normalisasi_dict = {}

    for index, row in kamus_slangword.iterrows():
        if row[0] not in kata_normalisasi_dict:
            kata_normalisasi_dict[row[0]] = row[1]

    token = [kata_normalisasi_dict[term]
              if term in kata_normalisasi_dict else term for term in token]

    # Stopwords Removal
    stopword = nltk.corpus.stopwords.words('indonesian')
    token = [word for word in token if word not in stopword]

    # Stemming
    token = [stemmer.stem(word) for word in token]
    return ' '.join(token)

    # Fungsi untuk membuat prediksi


def predict(Text):
    # Preprocessing
    preprocessed_Text = preprocess(Text)

    # Transformasi kalimat input menjadi list
    input_data = [preprocessed_Text]

    # Praproses teks menggunakan vectorizer yang telah dimuat
    input_data_vectorized = vectorizer.transform(input_data)

    # Konversi sparse matrix menjadi array
    input_data_dense = input_data_vectorized.toarray()

    # Melakukan prediksi menggunakan model yang dipilih
    predictions = models.predict(input_data_dense)

    # konversi label prediksi menjadi label kategori
    label_predict = ["Akuntansi", "Teknologi Informasi", "Manufaktur", "Pelayanan", "Penjualan",
                     "Sumber Daya Manusia", "Teknik"]
    label_result = label_predict[predictions[0]]
    return label_result


# Melakukan prediksi saat tombol diklik
if st.button("Prediksi"):
    if Text:
        kategori = predict(Text)
        st.write(f"Lowongan kerja yang dimasukkan diprediksi kedalam kategori: {kategori}")
    else:
        st.warning("Silakan masukkan deskripsi pekerjaan.")
st.markdown("---")
