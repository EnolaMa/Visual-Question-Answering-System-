import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model, tokenizer, and label_encoder
project_path = 'C:/Users/Enola/OneDrive/Desktop/767/project/dataset'
images_path = f'{project_path}/images'
model_path = f'{project_path}/vqa_model.keras'
tokenizer_path = f'{project_path}/question_tokenizer.pkl'
label_encoder_path = f'{project_path}/answer_label_encoder.pkl'
data_path = f'{project_path}/data.csv'

vqa_model = load_model(model_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(label_encoder_path, 'rb') as enc:
    label_encoder = pickle.load(enc)

# Function to extract image features using VGG16 model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return np.squeeze(features)

# Get the VGG16 base model
def get_base_model():
    base_model = VGG16(weights='imagenet', include_top=True)  # Set include_top to True
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    return model

base_model = get_base_model()

# Read data from CSV file and select a question and answer by line number
def get_question_answer_by_line(data_path, line_number):
    data_df = pd.read_csv(data_path)
    specific_data = data_df.iloc[line_number]
    return specific_data['question'], specific_data['answer'], specific_data['image_id']

# Function to visualize the prediction
def display_prediction(image_id, question, true_answer, model, tokenizer, max_length, label_encoder):
    image_path = f'{images_path}/{image_id}.png'
    features = extract_features(image_path, base_model)

    seq = tokenizer.texts_to_sequences([question])
    padded_seq = pad_sequences(seq, maxlen=max_length)

    pred = model.predict([np.array([features]), padded_seq])
    pred_class = label_encoder.inverse_transform([np.argmax(pred)])

    img = image.load_img(image_path, target_size=(224, 224))

    print(image_id)
    print(f"Question: {question}")
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    print(f"True Answer: {true_answer}")
    print(f"Predicted Answer: {pred_class[0]}")

# Ensure max_seq_length is consistent with the one used during model training
max_seq_length = 27

# Function to read and display prediction for a given line number in the CSV file
def predict_by_line_number(x):
    line_number = x
    question, true_answer, image_id = get_question_answer_by_line(data_path, line_number)
    display_prediction(image_id, question, true_answer, vqa_model, tokenizer, max_seq_length, label_encoder)

# Predict for line number
predict_by_line_number(9)
predict_by_line_number(11)
predict_by_line_number(3456)
predict_by_line_number(567)
predict_by_line_number(12297)