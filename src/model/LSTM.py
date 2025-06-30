import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from PrePostPredictor import prePostPrediction
import tensorflow as tf
import time
#from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

class LSTM_model(prePostPrediction):
  def __init__(self, folder_name):
    super().__init__(folder_name)

  def LSTMmodelTrainerPredictor(self, inputDf, test_size, win_length):
        #print("Test Train split started")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(inputDf,
                                                    inputDf['target'],
                                                    test_size=test_size,
                                                    random_state=42,
                                                    shuffle=False)
        self.X_train_shape = self.X_train.shape
        self.X_test_shape = self.X_test.shape
        self.y_train_shape = self.y_train.shape
        self.y_test_shape = self.y_test.shape
        scalar = StandardScaler()
        self.X_train=scalar.fit_transform(self.X_train)
        #self.X_test=scalar.fit_transform(self.X_test)
        self.X_test = scalar.transform(self.X_test)
        #print("Time series generator started")
        batch_size=150
        num_features=16
        train_generator = TimeseriesGenerator(self.X_train, self.y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
        test_generator = TimeseriesGenerator(self.X_test, self.y_test, length=win_length, sampling_rate=1, batch_size=batch_size)
        #print("Model definition started")
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(128, input_shape=(win_length, num_features), return_sequences=True))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        self.model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.LSTM(64, return_sequences=False))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(1))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor ='val_loss',
                                                  patience=2,
                                                  mode='min')
        self.model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])
        start_time = time.time()
        #print("Model Train started")
        history = self.model.fit(train_generator, epochs=50,
                              validation_data=test_generator,
                              shuffle=False,
                              callbacks=[early_stopping])

        end_time = time.time()
        self.execution_time = (end_time - start_time) / 60
        #print(f"Model train ended {self.execution_time}")
        self.actuals = self.y_test[win_length:]
        self.predictions=self.model.predict(test_generator)
        self.predictions = self.predictions.flatten()  # Convert to 1D array
        self.metricsCalculator()
        self.model.save(r"F:\OneEnergy\OneEnergyPredictor\models\LSTM_model_trained.h5")

source_dir = r'F:\OneEnergy\OneEnergyPredictor'
a = LSTM_model(source_dir)
'''for i in a.file_list:
            os.chdir(os.path.join(source_dir, i))
            print(i, 'input.csv', 0.3, sep="\t", end="---------")
            inputDf = pd.read_csv('input.csv', header=0, index_col=0, parse_dates=True)
            for time_lag in range(2,7):
                a.LSTMmodelTrainerPredictor(inputDf=inputDf, test_size=0.3,win_length=time_lag)
                a.outputLogger(file_name='Test_ml_output.csv', generator=i, split=0.3, algorithm="LSTM", time_lag=time_lag, input_df_shape=inputDf.shape, execution_time=a.execution_time)
            print(f"The file input {time_lag}  model results logged with execution time {a.execution_time}")
'''

#os.chdir(os.path.join(source_dir, 'southgate'))
os.chdir(os.path.join(source_dir, 'data'))
#print('southgate', 'input.csv', 0.3, sep="\t", end="---------")
inputDf = pd.read_csv('input.csv', header=0, index_col=0, parse_dates=True)
a.LSTMmodelTrainerPredictor(inputDf=inputDf, test_size=0.3,win_length=3)
#a.actuals.to_csv('actuals.csv', index=False)
pd.DataFrame(a.actuals).to_csv('actuals.csv', index=False)
pd.DataFrame(a.predictions).to_csv('predictios.csv', index=False)
a.residualsPlotter(destination_folder=r'F:\OneEnergy\OneEnergyPredictor\src\predict', algorithm='LSTM',name='Long-Short Term Memory of Amarnath Wind farm')
print(f"The file input {6}  model results logged with execution time {a.execution_time}")