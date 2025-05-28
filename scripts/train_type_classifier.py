import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
DATA_DIR = './gesture_type_data'
MODEL_PATH = '../app/models/gesture_type_model.pkl'
