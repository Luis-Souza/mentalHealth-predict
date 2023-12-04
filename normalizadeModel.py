from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential

class NormalizedModel:
    def __init__(self, norm_X_train, X_train, y_train, norm_X_test, y_test, X_test):
        self.norm_X_train = norm_X_train
        self.norm_X_test = norm_X_test
        self.y_test = y_test
        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train
        self.model = Sequential([
            Dense(9),
            Dense(18,activation='relu'), #18 neurons, ReLU activation function
            Dense(18,activation='relu'), #18 neurons, ReLU activation function
            Dense(9,activation='relu'), #9 neurons, ReLU activation function
            Dense(1,activation='sigmoid') #1 output neuron, sigmoid activation function for binary classification
        ])
        self.compile()

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    def history(self):
        history = self.model.fit(self.norm_X_train, self.y_train,validation_data=[self.norm_X_test, self.y_test],
          epochs = 50)
        return history

    def evaluate(self):
        self.model.evaluate(self.norm_X_test, self.y_test)

    def prediction(self):
        predictions = self.model.predict(self.norm_X_test)
        pred_labels = [int(round(x[0])) for x in predictions]
        return pred_labels
