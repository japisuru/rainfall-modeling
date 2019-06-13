from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
import os

class ModifiedLSTM:
    def train(self, features_set, labels, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(features_set, labels, epochs=100, batch_size=32)
        print("Finished training")

    def save(self, path):
        model_json = self.model.to_json()
        with open(os.path.join(path, "model.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(path, "model.h5"))
        print("Saved model to disk")

    def load(self, path):
        json_file = open(os.path.join(path, 'model.json'), 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights(os.path.join(path,"model.h5"))
        print("Loaded model from disk")

    def predict(self, test_features):
        return self.model.predict(test_features)