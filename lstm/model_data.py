import numpy as np
from sklearn.preprocessing import MinMaxScaler

class UnivariateModelData:

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_training(self, rf_training_processed, input_time_steps_size, taining_set_size):
        rf_training_scaled = self.scaler.fit_transform(rf_training_processed)

        features_set = []
        labels = []
        for i in range(input_time_steps_size, taining_set_size):
            features_set.append(rf_training_scaled[i - input_time_steps_size:i, 0])
            labels.append(rf_training_scaled[i, 0])

        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
        return features_set, labels, features_set.shape[1]

    def load_inference(self, test_inputs, input_time_steps_size, inference_set_size):
        test_inputs = test_inputs.reshape(-1, 1)
        test_inputs = self.scaler.transform(test_inputs)

        test_features = []
        for i in range(input_time_steps_size, input_time_steps_size + inference_set_size):
            test_features.append(test_inputs[i - input_time_steps_size:i, 0])

        test_features = np.array(test_features)
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
        return test_features

    def load_prediction(self, predictions):
        return self.scaler.inverse_transform(predictions)