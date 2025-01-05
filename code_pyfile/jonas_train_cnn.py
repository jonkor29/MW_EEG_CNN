import numpy as np
from EEGModels import EEGNet

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa



from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


#inspect GPU
print("TensorFlow version: ", tf.__version__)
print("GPU info: ", tf.config.list_physical_devices('GPU'))

loaded = np.load('loaded_dataset.npz')
X, y, s = loaded['X'], loaded['y'], loaded['s']

#define constants
NUM_SAMPLES = X.shape[0]
NUM_CHANNELS = X.shape[1]
NUM_TIMESTEPS = X.shape[2]
NUM_MW = np.count_nonzero(y == 1)
NUM_OT = np.count_nonzero(y == 0)
NUM_CLASSES = 2
CLASSES = np.unique(y)
SAMPLE_RATE = 128 #TODO: find the actual sampling rate by reading Jins paper or looking through the matlab code

print("Number of samples: ", NUM_SAMPLES)
print("Number of channels: ", NUM_CHANNELS)
print("Number of timesteps: ", NUM_TIMESTEPS)
print("Number of MW: ", NUM_MW)
print("Number of OT: ", NUM_OT)

y_onehot = to_categorical(y, num_classes=NUM_CLASSES)
print("y_onehot shape: ", y_onehot.shape)
#split data into train, val and test
X_train, X_temp, y_train_onehot, y_temp_onehot = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val_onehot, y_test_onehot = train_test_split(X_temp, y_temp_onehot, test_size=0.5, random_state=42, stratify=np.argmax(y_temp_onehot, axis=1))

NUM_TRAIN_SAMPLES = X_train.shape[0]
NUM_VAL_SAMPLES = X_val.shape[0]
NUM_TEST_SAMPLES = X_test.shape[0]
print("Number of train samples: ", NUM_TRAIN_SAMPLES)
print("Number of val samples: ", NUM_VAL_SAMPLES)
print("Number of test samples: ", NUM_TEST_SAMPLES)

class_weights = compute_class_weight(class_weight='balanced', classes=CLASSES, y=y)
class_weights_dict = dict(zip(CLASSES, class_weights))
print("Class weights: ", class_weights_dict)

model = EEGNet(nb_classes=NUM_CLASSES, Chans=NUM_CHANNELS, Samples=NUM_TIMESTEPS, dropoutRate=0.5, kernLength=SAMPLE_RATE//2, F1=8, D=2, F2=16, dropoutType='Dropout')
model.compile(optimizer='adam', loss='categorical_crossentropy', 
        metrics=['accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tfa.metrics.F1Score(num_classes=NUM_CLASSES, average='weighted', name='f1_score')
        ]
        )
history = model.fit(X_train, y_train_onehot, epochs=500, batch_size=16, validation_data=(X_val, y_val_onehot), class_weight=class_weights_dict)

model.save('/cluster/home/jonasjko/prosjektoppgave/MW_EEG_CNN/results/models/EEGNet8.2')

print("Loaded model loss and accuracy:")
loaded_model = load_model('/cluster/home/jonasjko/prosjektoppgave/MW_EEG_CNN/results/models/EEGNet8.2')
loss, precision, recall, F1_score = loaded_model.evaluate(X_test, y_test_onehot)
print(f"Loss: {loss}, precision: {precision}, Recall: {recall}, F1 score: {F1_score}")

