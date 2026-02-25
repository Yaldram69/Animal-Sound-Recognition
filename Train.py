import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define paths
base_path = 'C:\Pycharm Projects\Animal Sound Recognition\Dataset\Animals'

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def load_data(base_path):
    features = []
    labels = []
    for label in os.listdir(base_path):
        animal_path = os.path.join(base_path, label)
        if os.path.isdir(animal_path):
            for file in os.listdir(animal_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(animal_path, file)
                    features.append(extract_features(file_path))
                    labels.append(label)
    return np.array(features), np.array(labels)

# Load data
X, y = load_data(base_path)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32)

# Save model
model.save('animal_sound_recognition_model.h5')

# Save label encoder
import joblib
joblib.dump(le, 'label_encoder.pkl')
