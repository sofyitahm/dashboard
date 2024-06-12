import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import string
import re #regex library
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

nltk.download('stopwords')

#set page
st.set_page_config(
  page_title = "dashboard!", 
  page_icon = "?", # emoji ikon untuk ikon tab, bisa dengan "random"
  layout = "wide", #centered atau #wide
)

#HEADER
st.header('Dashboard', divider='grey')

#===============GATAU BUAT APA, BIAR PENUH AJA=================
Aspek = ['Performa', 'Pemesanan', 'Pembayaran', 'Fitur', 'Tampilan', 'Umum']
row1 = st.columns(6)
i=0
for col in row1:
    tile = col.container(height=60)
    tile.write(Aspek[i])
    i=i+1
#==================================================


#======================EXPANDER BIAR GAK MENUH2IN==================
#===========================
#MENGAMBIL DATA
#===========================

dfbaru=pd.DataFrame
url = 'https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/730-SUDAH-HAPUS-000000000.csv'
dfbaru = pd.read_csv(url)

#---SPLIT----#
X = dfbaru["ulasan"] #kolom ulasan
y = np.asarray(dfbaru[dfbaru.columns[1:13]]) #mengambil kolom multilabel dimulai dari kolom ke-2 hingga x dan dijadikan array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#===========================
   #MENGAMBIL DATA LATIH HASIL PREPROCESSING
#===========================

dflatih=pd.DataFrame
url = 'https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/dflatih_prepro(200).csv'
dflatih = pd.read_csv(url)

#===========================
   #TF-IDF DATA LATIH
#===========================

vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(dflatih['kalimat_cf'])

#===========================
   #TRAIN DATA LATIH
#===========================
# Create SVC classifier object with the best parameters
clf = SVC(C=100, kernel='rbf', gamma=0.01)

# Create BinaryRelevance classifier object with the SVC classifier
classifier = BinaryRelevance(classifier=clf)

# Train the model
classifier.fit(tfidf_vectors, y_train)

############################################################################
#-----------------------

#-----------------------
   # CLEANING 
#-----------------------
nltk.download('punkt')

# hapus tab, enter, back slice dll
def remove_tweet_special(text):
    # remove tab, baris baru, back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)//semua string di encode k ascii, yg tidak ada akan di replace ke ?, setelah itu dikembaliin lagi ke string
    text = text.encode('ascii', 'replace').decode('ascii')
     # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
def remove_number(text):
    return  re.sub(r"\d+", "", text)
def remove_punctuation(text):
    # Fungsi ini menghapus tanda baca dan menggantinya dengan satu spasi
    #string.punc=semua variabel di modul string yg berisi tanda baca
    #diganti dengan spasi sesuai dengan pandang tanda baca
    ganti = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(ganti)
#hapus whitespace tpi
def remove_whitespace_LT(text):
    return text.strip() #hanya awal&akhir
# hapus karater tunggal
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)
# hapus multi space, ubah ke 1 spasi
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text) #ganti 1 spasi
def cleaning (text):
    text=remove_tweet_special(text)
    text=remove_number(text)
    text=remove_punctuation(text)
    text=remove_whitespace_LT(text)
    text=remove_singl_char(text)
    text=remove_whitespace_multiple(text)
    return text

#-----------------------
#NORMALISASI
#----------------------

# Fungsi untuk normalisasi kata
def word_normalize(text):
    url = "https://api.prosa.ai/v2/text/normalizer"
    api_key = ""
    
    # Request header
    headers = {
        "Content-Type": "application/json",
        "x-API-Key": api_key
    }

    # Request body
    payload = {
        "text": text
    }

    # request post
    response = requests.post(url, headers=headers, json=payload)

    # cek jika berhasil
    if response.status_code == 200:
        # Mengembalikan teks hasil normalisasi
        return response.json().get('normalized_text', '')
    else:
        # memberikan hasil status code dan alasan text tidak berhasil
        return response.status_code, response.text
#-----------------------
# CASE FOLDING #
#-----------------------
def lower (text):
    text = text.lower()
    return text

#-------------------
# TOKENISASI
# ------------------
def word_tokenize_wrapper(text):
    return word_tokenize(text)

#-------------------
#STOPWORD REMOVAL
#---------------------
# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia

list_stopwords = stopwords.words('indonesian')

# convert list to dictionary
list_stopwords = set(list_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


#---------------------------
#STEMMING
#--------------------------
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stemmed wrapper function
def stemmed_wrapper(term):
    return stemmer.stem(term)

# Function to preprocess and stem terms
def prestem(input_list):
    # Stem each term in the list
    stemmed_terms = [stemmed_wrapper(term) for term in input_list]
    return stemmed_terms

#-----------------------------------
#JOIN KALIMAT
#--------------------------------
def join(text):
  text=' '.join(text)
  return text

#===========================
   # UJI
#===========================
   #---------------------------PREPROCESING------------------

def prepro (text):
    text = cleaning (text)
    text = word_normalize(text)
    text = lower(text)
    text = word_tokenize(text)
    text = stopwords_removal(text)
    text = prestem (text)
    text = join (text)
    return text

   #---------------------------TF-IDF------------------
def tfidf(text):
    tfidf_vectors2 = vectorizer.transform([text])
    return tfidf_vectors2

   #---------------------------PREDIKS------------------
def prediks(text_tfidf):
    predictions = classifier.predict(text_tfidf)
    hasil=predictions.toarray()
    return hasil

# -----------------------------------------------
my_expander = st.expander("**CEK SENTIMEN**", expanded=True)
aspek = [['Performa Positif', 'Performa Negatif', 'Pemesanan Positif', 'Pemesanan Negatif', 'Pembayaran Positif', 
            'Pembayaran Negatif', 'Fitur Positif', 'Fitur Negatif', 'Tampilan Positif', 'Tampilan Negatif', 'Umum Positif', 'Umum Negatif']]
   
with my_expander:
    inputx = st.text_input('Masukkan Teks')

    if st.button('Cek Hasil'):
        # Loop untuk memeriksa setiap elemen array
        inputx=prepro(inputx)
        inputvec=tfidf(inputx)
        hasil=prediks(inputvec)
        st.write("Hasil label: ", hasil)
        for i in range(len(hasil[0])):
            if hasil[0][i] == 1:
                st.write(aspek[0][i])
#===============================================================

dfaspek=pd.DataFrame

g1, g2 = st.columns([2,1])
with g1:
    option = st.selectbox(
    'Pilih Aspek',
    ('Performa', 'Pemesanan', 'Pembayaran', 'Fitur', 'Tampilan', 'Umum'))

with g1:
    if(option=='Performa'):
        url='https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/aspek-sudah-prepro/PERFORMA-SUDAH%20PREPRO-(200).csv'
    elif((option=='Pemesanan')):
        url='https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/aspek-sudah-prepro/PEMESANAN-SUDAH%20PREPRO-(200).csv'
    elif((option=='Pembayaran')):
        url='https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/aspek-sudah-prepro/PEMBAYARAN-SUDAH%20PREPRO-(200).csv'
    elif((option=='Fitur')):
        url='https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/aspek-sudah-prepro/FITUR-SUDAH%20PREPRO-(200).csv'
    elif((option=='Tampilan')):
        url='https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/aspek-sudah-prepro/TAMPILAN-SUDAH%20PREPRO-(200).csv'
    elif((option=='Umum')):
        url='https://raw.githubusercontent.com/sofyitahm/SKRIPSI/main/FIX/aspek-sudah-prepro/UMUM-SUDAH%20PREPRO-(200).csv'
    dataspek= pd.read_csv(url)
    pie = px.pie(dataspek['ulasan'], names=dataspek['label'])
    st.plotly_chart(pie) 
    df_positif = dataspek [dataspek['label'] == "Positif"]
    teks_positif = ' '.join(df_positif['ulasan'])

    df_negatif = dataspek [dataspek['label'] == "Negatif"]
    teks_negatif = ' '.join(df_negatif['ulasan'])

    # Membuat WordCloud
    wordcloud1 = WordCloud(width=700, height=450, background_color='white').generate(teks_positif)
    wordcloud2 = WordCloud(width=700, height=450, background_color='white').generate(teks_negatif)
    aspek=option
    
with g2:
    radio= st.radio(
        "Pilih Sentimen",
        ["Positif", "Negatif"],
        )
    if (radio=="Positif"):
        st.image(wordcloud1.to_array())
    elif (radio=="Negatif"):
        st.image(wordcloud2.to_array())
