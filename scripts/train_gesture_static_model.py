import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Папка с данными для статических жестов
DATA_DIR = './gesture_type_data/static'
MODEL_PATH = '../app/models/gesture_static_model.pkl'

X, y = [], []
