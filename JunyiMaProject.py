import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelEncoder
import pickle

# Set file paths
project_path = 'C:/Users/Enola/OneDrive/Desktop/767/project/dataset'
images_path = f'{project_path}/images'
data_path = f'{project_path}/data.csv'
train_images_list_path = f'{project_path}/train_images_list.txt'
eval_images_list_path = f'{project_path}/test_images_list.txt'
answer_space_path = f'{project_path}/answer_space.txt'


# Load data
data_df = pd.read_csv(data_path)
train_df = data_df[data_df['image_id'].isin(pd.read_csv(train_images_list_path, header=None)[0])]
eval_df = data_df[data_df['image_id'].isin(pd.read_csv(eval_images_list_path, header=None)[0])]
answer_space_df = pd.read_csv(answer_space_path, header=None, names=['answer'])

# Clean and process possible NaN values in the answer space
answer_space_df.dropna(inplace=True)

# Preprocess question text
question_tokenizer = Tokenizer()
question_tokenizer.fit_on_texts(data_df['question'])

# Use train_questions_seq and eval_questions_seq to find max length for consistency
train_questions_seq = question_tokenizer.texts_to_sequences(train_df['question'])
eval_questions_seq = question_tokenizer.texts_to_sequences(eval_df['question'])
max_seq_length = max(max(len(seq) for seq in train_questions_seq), max(len(seq) for seq in eval_questions_seq))
# print(max_seq_length)
train_questions_padded = pad_sequences(train_questions_seq, maxlen=max_seq_length, padding='post')
eval_questions_padded = pad_sequences(eval_questions_seq, maxlen=max_seq_length, padding='post')

# Fit and process answer labels
all_answers = np.concatenate((
    answer_space_df['answer'].unique(),
    train_df['answer'].unique(),
    eval_df['answer'].unique()
))
label_encoder = LabelEncoder()
label_encoder.fit(all_answers)
train_answers = label_encoder.transform(train_df['answer'])
eval_answers = label_encoder.transform(eval_df['answer'])
num_classes = len(label_encoder.classes_)
train_answers_categorical = to_categorical(train_answers, num_classes=num_classes)
eval_answers_categorical = to_categorical(eval_answers, num_classes=num_classes)

# Load the pre-trained VGG16 model to extract image features
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return np.squeeze(features)

# Check if feature files exist, load them if they do, otherwise extract and save
train_features_path = os.path.join(project_path, 'train_features.npy')
eval_features_path = os.path.join(project_path, 'eval_features.npy')

if os.path.exists(train_features_path) and os.path.exists(eval_features_path):
    train_features = np.load(train_features_path)
    eval_features = np.load(eval_features_path)
else:
    # If feature files do not exist, perform feature extraction logic and save
    train_features = np.array([extract_features(os.path.join(images_path, f"{row['image_id']}.png"), model) for index, row in train_df.iterrows()])
    eval_features = np.array([extract_features(os.path.join(images_path, f"{row['image_id']}.png"), model) for index, row in eval_df.iterrows()])
    np.save(train_features_path, train_features)
    np.save(eval_features_path, eval_features)

# Build the VQA model
vocab_size = len(question_tokenizer.word_index) + 1
text_input = Input(shape=(max_seq_length,))
text_embedding = Embedding(input_dim=vocab_size, output_dim=256)(text_input)
text_dropout = Dropout(0.4)(text_embedding)
encoded_text = LSTM(128)(text_dropout)
image_input = Input(shape=(4096,))
dense_image = Dense(256, activation='relu', kernel_regularizer=l2(0.05))(image_input)
merged = Concatenate()([encoded_text, dense_image])
dense_merged = Dense(256, activation='relu')(merged)
output_dropout = Dropout(0.5)(dense_merged)
final_output = Dense(num_classes, activation='softmax')(output_dropout)

vqa_model = Model(inputs=[image_input, text_input], outputs=final_output)

# Setting the learning rate
learning_rate = 0.0003

# Instantiate the optimizer and set the learning rate
adam_optimizer = Adam(learning_rate=learning_rate)

# Compile the model using a custom optimizer
vqa_model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the updated optimizer
vqa_model.fit([train_features, train_questions_padded], train_answers_categorical, epochs=100, batch_size=64)


# Evaluate the model
eval_loss, eval_acc = vqa_model.evaluate([eval_features, eval_questions_padded], eval_answers_categorical)
print(f'Evaluation loss: {eval_loss}')
print(f'Evaluation accuracy: {eval_acc}')

# Save model in SavedModel format
vqa_model.save(os.path.join(project_path, 'vqa_model.keras'))