import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Đọc file CSV
df = pd.read_csv('keypoints_dataset.csv')

# Tách đặc trưng và nhãn
X = df.drop('label', axis=1).values.astype('float32')
y = df['label'].values

# Mã hóa nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encode nhãn
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Tạo model đơn giản
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình
model.save(r'D:\DoAn_XuLiAnhSo\Sign_Language\.venv\ASL_model.h5')

# Lưu LabelEncoder
with open(r'D:\DoAn_XuLiAnhSo\Sign_Language\.venv\label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Đã train và lưu model + label encoder")
