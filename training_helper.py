
#import libraries using try/except

try:
    import pandas as pd
    import numpy as np
    import pathlib
    import os
    import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf 
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    import weightwatcher as ww
    import warnings
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    np.set_printoptions(precision=4)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
except:
    !pip install keras
    !pip install --upgrade tensorflow
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    
class util:

    def data_prep(my_state):
        #select features
        X = df[['EMG(mv)', 'ContractionNo', 'Trial Type', 'ContractionDuration']]
        y = df['Pain Label']
        #create data split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=my_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=my_state)
        print("Number of samples in the training set: ", len(X_train))
        print("Number of samples in the test set: ", len(X_test))
        print("Number of samples in the validation set: ", len(X_val))
        #scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val


    def create_CNN_model(my_state, num_feat):
        tf.random.set_seed(my_state)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(100, num_feat, activation='relu', name="convLayer", input_shape=(num_feat, 1)),
            tf.keras.layers.Dense(128, activation='relu', name="relu1Layer"),
            tf.keras.layers.Dense(256, activation='relu', name="relu2Layer"),
            tf.keras.layers.Dense(256, activation='relu', name="relu3Layer"),
            #tf.keras.layers.Dense(1, activation='sigmoid', name="sigmoidLayer")
            tf.keras.layers.Dense(1, activation='relu', name="sigmoidLayer")
        ])
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.BinaryCrossentropy(name='binary cross entropy'),
                tf.keras.metrics.BinaryIoU(name='binary IoU')
            ]

        )
        return model

    def create_LSTM_model(my_state, num_feat):
        tf.random.set_seed(my_state)
        model = tf.keras.Sequential([
            #tf.keras.layers.LSTM(128, input_shape=(num_feat, 1)),
            tf.keras.layers.LSTM(256, input_shape=(num_feat, 1)),
            #tf.keras.layers.Dense(128, activation='relu', name="relu1Layer"),
            #tf.keras.layers.Dense(256, activation='relu', name="relu2Layer"),
            #tf.keras.layers.Dense(256, activation='relu', name="relu3Layer"),
            tf.keras.layers.Dense(1, activation='sigmoid', name="sigmoidLayer")
        ])

    def create_regularized_LSTM_model(my_state, factor, rate, num_feat):
        tf.random.set_seed(my_state)
        model = tf.keras.Sequential([
            #Dropout(rate),
            tf.keras.layers.LSTM(256, kernel_regularizer=l2(factor), input_shape=(num_feat, 1)),
            Dropout(rate),
            tf.keras.layers.LSTM(128, input_shape=(256, 1)),
            Dropout(rate),
            tf.keras.layers.Dense(128, kernel_regularizer=l2(factor), activation='relu', name="relu1Layer"),
            Dropout(rate),
            tf.keras.layers.Dense(256, kernel_regularizer=l2(factor), activation='relu', name="relu2Layer"),
            Dropout(rate),
            tf.keras.layers.Dense(256, kernel_regularizer=l2(factor), activation='relu', name="relu3Layer"),
            Dropout(rate),
            tf.keras.layers.Dense(1, activation='sigmoid', name="sigmoidLayer")
            #tf.keras.layers.Dense(1, activation='relu', name="reluLayer")
        ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryCrossentropy(name='binary cross entropy'),
            tf.keras.metrics.BinaryIoU(name='binary IoU')
        ]
    )
    return model

    def train_model(X_train_scaled, y_train, num_epochs):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        print("Training the neural network.. ")
        history = model.fit(X_train_scaled, y_train, batch_size=32, epochs=num_epochs, callbacks=[tensorboard_callback])
        print("Finished training!")
        model.summary()
        return history

    def visualize_performance(num_epochs, history):
        rcParams['figure.figsize'] = (18, 8)
        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history['loss'], label='Loss'
        )
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history['accuracy'], label='Accuracy'
        )
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history['precision'], label='Precision'
        )
        plt.plot(
            np.arange(1, num_epochs+1), 
            history.history['recall'], label='Recall'
        )
        plt.title('Evaluation metrics', size=20)
        plt.xlabel('Epoch', size=14)
        plt.legend();

    def evaluate_performance(model, X_test_scaled, y_test):
        predictions = model.predict(X_test_scaled)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.4f}')
        print(f'Precision: {precision_score(y_test, prediction_classes):.4f}')
        print(f'Recall: {recall_score(y_test, prediction_classes):.4f}')