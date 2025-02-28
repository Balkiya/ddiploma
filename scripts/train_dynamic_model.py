import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# === Параметры ===
DATA_DIR = './gesture_type_data/dynamic'
MODEL_PATH = './lstm_dynamic_model.keras'
SEQUENCE_LENGTH = 30
