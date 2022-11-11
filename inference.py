import pickle
import pprint
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import re, os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tqdm import tqdm       
import seaborn as sns
import matplotlib.pyplot as plt  # visualizations                                          

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.utils import shuffle   #data manipulation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
    
import tensorflow as tf     
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dropout, LSTM, Bidirectional, GRU, Dense, Embedding, BatchNormalization

print(pickle.format_version)

np.random.seed(1234)
tf.random.set_seed(1234)

print('Tensorflow version : ', tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
    print("Consuming GPU for Training.") 
    
else:
    print("WARNING: Could not find GPU! Using CPU only.")


############################################################################################

reviews_path = "files/data.csv"
treatment_path = "files/treatments.csv"
doctor_score_path = "files/doctor scores.csv"
                                                 
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
stopwords_list = stopwords.words('english')

oov_token = '<OOV>'
pad_token = '<PAD>'

############################################################################################

def load_data():
    df_reviews = pd.read_csv(reviews_path) 
    df_reviews.review = df_reviews.apply(lambda x : eval(x.review).decode('utf-8'), axis=1)

    df_reviews.S = df_reviews.S.astype(int)  #astype for convert datatype as a existing data column 
    df_reviews.P = df_reviews.P.astype(int)
    df_reviews.H = df_reviews.H.astype(int)
    df_reviews.K = df_reviews.K.astype(int)
    df_reviews.doc_id = df_reviews.doc_id.astype(int)
    df_reviews.review = df_reviews.review.astype(str)

    X = df_reviews.review.values
    Y = df_reviews[['S','P','H','K']].values

    doc_id = df_reviews.doc_id.values
    doc_name = df_reviews.name.values
    scoliosis_type = df_reviews.scoliosis_type.values

    id2doc = dict(zip(doc_id, doc_name))
    id2scoliosis = dict(zip(doc_id, scoliosis_type))

    X, Y, doc_id = shuffle(X, Y, doc_id, random_state=1234)
    return X, Y, id2doc, id2scoliosis, doc_id

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    return [k for k in lem if k]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        updated_reviews = [preprocess_one(review) for review in reviews]
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

def vis_length_variation(X_SEQ):
    X_len = [len(i) for i in X_SEQ]
    X_len = pd.Series(X_len)
    X_len.hist()
    plt.xlabel('Token Length')
    plt.ylabel('Samples')
    plt.savefig('visualization/sequence length.png')
    plt.show()
    
    print(X_len.describe())

def retrieve_data():
    X, Y, id2doc, id2scoliosis, doc_id = load_data()
    X_SEQ = preprocessed_data(X)
 
    YS = Y[:,0] - 1
    YP = Y[:,1] - 1
    YH = Y[:,2] - 1
    YK = Y[:,3] - 1
    return X_SEQ, YS, YP, YH, YK, id2doc, id2scoliosis, doc_id

class DoctorRecommendation(object):
    def __init__(self):
        X, YS, YP, YH, YK, id2doc, id2scoliosis, doc_id = retrieve_data()

        self.X = X
        self.doc_id = doc_id
        self.id2doc = id2doc
        self.id2scoliosis = id2scoliosis
        self.YS = tf.keras.utils.to_categorical(YS, num_classes=5)
        self.YP = tf.keras.utils.to_categorical(YP, num_classes=5)
        self.YH = tf.keras.utils.to_categorical(YH, num_classes=5)
        self.YK = tf.keras.utils.to_categorical(YK, num_classes=5)

        self.max_length = 50
        self.tokenizer_path = 'weights/TOKENIZER.pkl'
        self.model_weights = 'weights/doctor_recommendation.h5'

    def save_load_tokenizer(self):
        if not os.path.exists(self.tokenizer_path):
            tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<oov>')
            tokenizer.fit_on_texts(self.X)
            
            with open(self.tokenizer_path, 'wb') as fp:
                pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            with open(self.tokenizer_path, 'rb') as fp:
                tokenizer = pickle.load(fp)
                
        return tokenizer
    
    def handle_data(self):
        tokenizer = self.save_load_tokenizer()
        
        X_seq = tokenizer.texts_to_sequences(self.X) # tokenize train data
        self.X_pad = pad_sequences(
                                X_seq, 
                                maxlen=self.max_length, 
                                padding='pre', 
                                truncating='pre'
                                )# Pad Train data

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1
    
    def loaded_model(self): # Load and compile pretrained model
        self.model = load_model(self.model_weights)
        self.model.compile(
                    loss='categorical_crossentropy', 
                    optimizer = Adam(learning_rate=0.001), 
                    metrics=['accuracy']
                        )

    def predict_doctor_score(self, ID):
        idxs = (self.doc_id == ID)
        if sum(idxs) > 0:
            X_doc_pad = self.X_pad[idxs]
            pred = self.model.predict(X_doc_pad)
            PS, PP, PH, PK = pred
            
            PS = np.argmax(PS, axis=1) + 1  #return the argument that target function return the max value
            PS = PS.mean().astype(int) 

            PP = np.argmax(PP, axis=1) + 1
            PP = PP.mean().astype(int) 

            PH = np.argmax(PH, axis=1) + 1
            PH = PH.mean().astype(int) 

            PK = np.argmax(PK, axis=1) + 1
            PK = PK.mean().astype(int) 
            
            P = (PS + PP + PH + PK) / 4
            return P
        else:
            return 0

    def generate_all_doctor_scores(self):
        if not os.path.exists(doctor_score_path):
            self.doc_scores = {}
            doc_id_unique = list(self.id2doc.keys())
            self.doc_scores['Doctor ID'] = []
            self.doc_scores['Doctor Name'] = []
            self.doc_scores['Doctor Score'] = []
            self.doc_scores['Scoliosis Type'] = []
            for idx, ID in enumerate(doc_id_unique):
                if idx % 100 == 0:
                    print("processing {}/{}".format(idx+1, len(doc_id_unique)))
                self.doc_scores['Doctor ID'].append(ID)
                self.doc_scores['Doctor Score'].append(self.predict_doctor_score(ID)) #get data from above function
                self.doc_scores['Scoliosis Type'].append(self.id2scoliosis[ID])
                self.doc_scores['Doctor Name'].append(self.id2doc[ID])

            self.doc_scores = pd.DataFrame(self.doc_scores)
            self.doc_scores.to_csv(
                                doctor_score_path, 
                                index=False
                                )
        else:
            self.doc_scores = pd.read_csv(doctor_score_path)

    def retrieve_doctors(self, ScoliosisType):
        doctor_data = self.doc_scores[self.doc_scores['Scoliosis Type'] == ScoliosisType]
        doctor_data = doctor_data.sort_values(by=['Doctor Score'], ascending=False)
        doctor_data = doctor_data[['Doctor Name', 'Doctor Score']].head(5) # Chnage Here to retrieve more doctors
        doctor_data.reset_index(drop=True, inplace=True)
        Doctors = doctor_data['Doctor Name'].values.tolist()
        return Doctors

    def retrieve_treatment_plan(self, SpineAngle):
        SpineAngle = float(SpineAngle)
        df_treatment = pd.read_csv(treatment_path)
        df_treatment['Lower Bound'] = df_treatment['Lower Bound'].apply(lambda x: float(x))
        df_treatment['Upper Bound'] = df_treatment['Upper Bound'].apply(lambda x: float(x))
        treatment_idx = np.logical_and(df_treatment['Lower Bound'] <= SpineAngle, df_treatment['Upper Bound'] > SpineAngle)
        treatment_plan = df_treatment[treatment_idx]

        assert len(treatment_plan) == 1, "There should be only one plan but found {}".format(len(treatment_plan))

        ImageUrl = treatment_plan['Image Url'].values[0]
        Treatment = treatment_plan['Treatment'].values[0]
        Description = treatment_plan['Description'].values[0]
        return ImageUrl, Treatment, Description

    def make_response(self, request):
        ScoliosisType = request['ScoliosisType']
        SpineAngle = request['SpineAngle']

        Doctors = self.retrieve_doctors(ScoliosisType)
        ImageUrl, Treatment, Description = self.retrieve_treatment_plan(SpineAngle)

        return {
                'Doctors': Doctors,
                'ImageUrl': f"{ImageUrl}",
                'Treatment': f"{Treatment}",
                'Description': f"{Description}"
                }

    def run(self):
        self.handle_data()
        self.loaded_model()
        self.generate_all_doctor_scores()