# Importing libraries
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle

# Set random seed
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# Dataset class name function
def get_className(files_list):
    classNames = []
    for file in files_list:
        label = file.split("/")[-1].split("_")[0]
        if label not in classNames:
            classNames.append(label)
    classNames.sort()
    return classNames

# class list ot dict
def get_classDict(classNames):
    classDict = {}
    for index, className in enumerate(classNames):
        classDict[className] = index
    return classDict


# Data load function
def load_data(filePath, classDict, totalClass):
    df = pd.read_csv(filePath).iloc[: ,1:].to_numpy()
    label = classDict[filePath.split("/")[-1].split("_")[0]]
    # label = tf.keras.utils.to_categorical(label, totalClass, dtype='int8')
    return df, label


# Data generator function
def get_dataFrame(filesList, classDict, classNames):
    dataFrame = []
    labels = []
    for filePath in tqdm(filesList):
        data, label = load_data(filePath, classDict, len(classNames))
        dataFrame.append(data)
        labels.append(label)
    dataFrame = np.array(dataFrame)
    labels = np.array(labels)
    return dataFrame, labels


# Labels one hot encoding function
def labels_oneHot_Encoding(labels, totalClasses):
    return tf.keras.utils.to_categorical(labels, totalClasses, dtype='int8')


# Data path
train_dir = "datasets/TrainingValidation"
test_dir = "datasets/Testing"


#  Get all files list
train_files = glob.glob(f"{train_dir}/*.csv")
test_files = glob.glob(f"{test_dir}/*.csv")


# get classNames and classDict
classNames = get_className(train_files)
classDict = get_classDict(classNames)


# get dataFrame and labels
trainData, trainLabels =  get_dataFrame(train_files, classDict, classNames)
testData, testLabels =  get_dataFrame(test_files, classDict, classNames)



# labels one hot
trainLabels = labels_oneHot_Encoding(trainLabels, len(classNames))
testLabels = labels_oneHot_Encoding(testLabels, len(classNames))


#Data shuffle
trainData, trainLabels = shuffle(trainData, trainLabels, random_state=SEED)



#Creating checkpoint
checkpoint_path = f"CheckPoint/cp.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True
)



# Create model
def create_model():
    input_shape = trainData.shape[-2:]
    input_layer = tf.keras.layers.Input(shape=input_shape, name='input_layer')
    layer_1 = tf.keras.layers.Conv1D(64, 3, activation='relu',padding='same')(input_layer)
    layer_2 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    query_value_attention_seq = tf.keras.layers.Attention()([layer_1, layer_2])
    layer_3 = tf.keras.layers.Conv1D(32, 10, activation='relu',padding='same')(layer_1)
    layer_4 = tf.keras.layers.Conv1D(128, 10, activation='relu',padding='same')(layer_3)
    layer_5 = tf.keras.layers.Conv1D(256, 10, activation='relu',padding='same')(layer_2)
    layer_6 = tf.keras.layers.Conv1D(128, 3, activation='relu',padding='same')(layer_5)
    query_value_attention_seq_2 = tf.keras.layers.Attention()([layer_4, layer_6])
    concatinate_layer = tf.keras.layers.Concatenate()([query_value_attention_seq, query_value_attention_seq_2])
    x = tf.keras.layers.Flatten()(concatinate_layer)
    output_layer = tf.keras.layers.Dense(len(classNames), activation='softmax', name='output_layer')(x)
    model = tf.keras.Model(input_layer, output_layer)
    return model

model = create_model()
# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()




#Fit model
model.fit(
    trainData,
    trainLabels,
    validation_data = (testData, testLabels),
    batch_size=32,
    epochs= 200,
    callbacks = [
        checkpoint_path
    ]
)


#Colne model
best_model = tf.keras.models.clone_model(model)

#Compile model
best_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Load weights
best_model.load_weights(checkpoint_path)


# Evaluate
best_model.evaluate(trainData, trainLabels), best_model.evaluate(testData, testLabels)





