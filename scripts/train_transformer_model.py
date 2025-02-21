import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
# === Параметры ===
DATA_DIR = './gesture_type_data/dynamic'
MODEL_PATH = '../app/models/transformer_dynamic_model.keras'
SEQUENCE_LENGTH = 30
# === Загрузка данных ===
X, y = [], []
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}
for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    files = sorted(os.listdir(label_dir))
    for i in range(len(files) - SEQUENCE_LENGTH):
        sequence = []
        for j in range(SEQUENCE_LENGTH):
            file_path = os.path.join(label_dir, files[i + j])
            keypoints = np.load(file_path)
            sequence.append(keypoints.flatten())
        X.append(sequence)
        y.append(label_map[label])
X = np.array(X)
y = to_categorical(y)

# === Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Transformer Encoder Layer ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Self-attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

