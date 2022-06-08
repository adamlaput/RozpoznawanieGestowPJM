import numpy as np
from numpy.random import seed
import pandas as pd
from keras.layers import Dense, LSTM, MaxPool2D, Conv2D, Flatten
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

activation_list = ["tanh", "relu"]
epochs_list = [30, 50, 70]
dataset_list = ["550-points", "550-vectors","550-images" , 
                "2150-points", "2150-vectors", "2150-images", 
                "PJM/Edited/PJM-points", "PJM/Edited/PJM-vectors", "PJM/PJM-images"]

for dataset_index, choosed_dataset in enumerate(dataset_list):
    dataset = pd.read_csv("Do Kaggle/"+choosed_dataset+".csv", sep=',', header=0)
    print("Wczytano "+choosed_dataset)

    X = []
    y = []

    if (dataset_index % 3 == 2):
        for i in range(dataset.shape[0]):
            hand1 = np.array(dataset.iloc[i, 6:1606]).reshape(40,40)
            hand2 = np.array(dataset.iloc[i, 1609:3209]).reshape(40,40)
            hand3 = np.array(dataset.iloc[i, 3212:4812]).reshape(40,40)
            X.append([hand1, hand2, hand3])
            y.append(dataset.iloc[i, 2])
    else:
        dataset_hand_size = int((dataset.shape[1] - 3 ) / 3)
        for i in range(dataset.shape[0]):
            hand1 = dataset.iloc[i, 3:3+dataset_hand_size]
            hand2 = dataset.iloc[i, 3+dataset_hand_size:3+dataset_hand_size*2]
            hand3 = dataset.iloc[i, 3+dataset_hand_size*2:3+dataset_hand_size*3]
            X.append([hand1, hand2, hand3])
            y.append(dataset.iloc[i, 2])

    X, y = np.array(X), np.array(y)
    X = np.asarray(X).astype('float32')

    cross_validation = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=222222)
    split_data = cross_validation.split(X, y)

    def create_network_lstm(input_shape_hand,act_fun):
        model = Sequential()
        model.add(LSTM(128, input_shape=(3, input_shape_hand), return_sequences=True, activation=act_fun))
        model.add(LSTM(64, activation=act_fun))
        model.add(Dense(40, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['categorical_accuracy'])
        return model

    def create_network_cnn(act_fun):
        model = Sequential()
        model.add(Conv2D(128, (3,3) , strides = 1 , padding = 'same' , activation = act_fun , input_shape = (3,40,40)))
        model.add(MaxPool2D((2,2), strides=2, padding='same'))
        model.add(Conv2D(64, (2,2), strides = 1 , padding = 'same',  activation = act_fun))
        model.add(MaxPool2D((2,2), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(40 , activation = 'softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['categorical_accuracy'])
        return model

    for choosed_act in activation_list:
        for choosed_epoch in epochs_list:
            scores = []
            train_acc = []
            train_loss = []
            val_acc = []
            val_loss = []

            for train, test in cross_validation.split(X, y):

                model = 0
                seed(241)
                if (dataset_index % 3 == 2):
                    model = create_network_cnn(choosed_act)
                else:
                    model = create_network_lstm(dataset_hand_size, choosed_act)
                
                y_train = y[train].reshape(-1, 1)
                encoder = OneHotEncoder(sparse=False)
                y_train = encoder.fit_transform(y_train)

                y_test = y[test].reshape(-1, 1)
                encoder = OneHotEncoder(sparse=False)
                y_test = encoder.fit_transform(y_test)

                X_train, X_val, y_train, y_val = train_test_split(X[train], y_train, test_size=.10, random_state=241350)

                history = model.fit(X_train, y_train, epochs=choosed_epoch, validation_data=(
                    X_val, y_val), batch_size=16, verbose=2)
                test_y_predicted = model.predict(X[test])
                test_y_predicted = (test_y_predicted > 0.5)

                scores.append([accuracy_score(y_test, test_y_predicted),
                            recall_score(y_test, test_y_predicted, average='weighted'),
                            precision_score(y_test, test_y_predicted, average='weighted'),
                            balanced_accuracy_score(y_test.argmax(axis=1), test_y_predicted.argmax(axis=1)),
                            f1_score(y_test, test_y_predicted, average='weighted')])
                
                train_acc.append(history.history['categorical_accuracy'])
                train_loss.append(history.history['loss'])
                val_acc.append(history.history['val_categorical_accuracy'])
                val_loss.append(history.history['val_loss'])

            scores = np.array(scores)
            ost_scores = []
            for i in range(5):
                ost_scores.append([np.mean(scores, axis=0)[i], np.std(scores, axis=0)[i]])

            np.savez('Scores/'+choosed_dataset+'-'+choosed_act+'-'+str(choosed_epoch), scores, ost_scores, train_acc, train_loss, val_acc, val_loss)