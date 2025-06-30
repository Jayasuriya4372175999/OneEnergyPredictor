import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class prePostPrediction():
    def __init__(self, file_path):
        self.input_dir = file_path
        self.folders_list = [f for f in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, f))]
        self.winter_months = [10,11,12,1,2,3]
        self.file_list = {}
        for i in self.folders_list:
            os.chdir(os.path.join(self.input_dir, i))
            file_list = [f for f in os.listdir(os.path.join(self.input_dir, i)) if '_time_lag' in f or f=='input.csv']
            self.file_list[i] = file_list
    

    def dataSplitStandardizer(self, input_df, test_size):
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(input_df.drop(columns =['target']),
                                                            input_df['target'],
                                                            test_size=test_size,
                                                            random_state=42

        )
        self.X_train_shape = self.X_train.shape
        self.X_test_shape = self.X_test.shape
        self.y_train_shape = self.y_train.shape
        self.y_test_shape = self.y_test.shape
        scalar = StandardScaler()
        self.X_train=scalar.fit_transform(self.X_train)
        self.X_test=scalar.fit_transform(self.X_test)
    
    def metricsCalculator(self):
        self.MAE = mean_absolute_error(self.actuals, self.predictions)
        self.MSE = mean_squared_error(self.actuals, self.predictions)
        self.RMSE = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        self.R2 = r2_score(self.actuals, self.predictions)

    '''def multipleLinearRegressionTrainerPredictor(self):
        self.regression = LinearRegression()
        self.regression.fit(self.X_train, self.y_train)
        self.actuals = self.y_test
        self.predictions=self.regression.predict(self.X_test)
        self.metricsCalculator()'''
        

    def outputLogger(self, file_name, generator, split, algorithm, time_lag, input_df_shape, execution_time =None):
        new_row = pd.Series({'generator' : generator, 'split': split, 'algorithm': algorithm, 'time_lag': time_lag, 'MAE': self.MAE, 'MSE': self.MSE, 'RMSE': self.RMSE, 'R2': self.R2, 'input_df': input_df_shape ,'X_train':self.X_train_shape, 'X_test':self.X_test_shape, 'y_train': self.y_train_shape, 'y_test':self.y_train_shape, 'Execution_Time': execution_time})
        os.chdir(self.input_dir)
        metrics_df = pd.read_csv(file_name, header=0, index_col=0)
        metrics_df = metrics_df._append(new_row, ignore_index=True)
        metrics_df.to_csv(file_name)
        print("Result Logged")

    def residualsPlotter(self, destination_folder, algorithm, name):
        output_folder = os.path.join(destination_folder, name)
        if not (os.path.exists(output_folder) and os.path.isdir(output_folder)):
            os.makedirs(output_folder)
            print('Newfolder created')
        print(output_folder, end="\n________________\n")
        os.chdir(output_folder)
        
        font1 = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 16}


        self.residuals =self.actuals-self.predictions
        plt.figure(figsize=(8,8))
        sns.scatterplot(x=self.predictions, y=self.residuals, color='#90EE90', edgecolor='black')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Predictions", fontdict=font1)
        plt.ylabel("Residuals", fontdict=font1)
        title = f"{name} \n predictions vs residuals scatter plot"
        plt.title(title, fontdict=font1)
        plt.tight_layout()
        plt.show()
        plt.savefig('predictions vs residuals scatter plot')
        print(f"{title} saved")
        plt.figure(figsize=(8,8))
        g = sns.displot(self.residuals, kind='kde')
        g.set_titles(row_template='', col_template='')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Residuals", fontdict=font1)
        plt.ylabel("Frequency", fontdict=font1)
        plt.title("Residuals displot".format(name), fontdict=font1)
        title = f"{name} \n residuals displot"
        plt.suptitle(title, fontdict=font1)
        plt.savefig('residuals displot')
        plt.tight_layout()
        plt.show()
        plt.close()
        print(f"{title} saved", end ="\n__________\n")
        plt.figure(figsize=(8,8))
        sns.scatterplot(x=self.actuals, y=self.predictions, color='#FFD700', edgecolor='black')  #
        sns.regplot(x=self.actuals, y=self.predictions, scatter=False, color='black', ci=None, line_kws={"linewidth": 2}) # Adjust the line width as needed
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Actuals", fontdict=font1)
        plt.ylabel("Predictions", fontdict=font1)
        title = "{} \n actual value vs predictions plot".format(name)
        plt.title(title, fontdict=font1)
        plt.tight_layout()
        plt.savefig('actual value vs predictions plot')
        plt.show()
        print(f"{title} saved")
        print("\n\n\n")
                
        '''
source_dir = r'F:\OneEnergyPredictor\machine_learning_models'
a = prePostPrediction(source_dir)
for i in a.file_list:
    files = a.file_list[i]
    for j in files:
        for k in [0.3, 0.2, 0.35, 0.50]:
            os.chdir(os.path.join(source_dir, i))
            print(i, j, k, sep="\t", end="---------")
            if j == 'input.csv':
                time_lag = 0
            else:
                time_lag = j.split('.')[0].split('_')[-1]
            inputDf = pd.read_csv(j, header=0, index_col=0, parse_dates=True)
            a.dataSplitStandardizer(inputDf, k)
            a.multipleLinearRegressionTrainerPredictor()
            a.outputLogger(generator=i, split=k, algorithm="Multiple Linear Regression", time_lag=time_lag, input_df_shape=inputDf.shape)
        print(f"The file {j} model results logged")
            
            
            #print(a.X_train_shape, a.y_train_shape, a.X_test_shape, a.y_test_shape)
            #print(inputDf.shape)
            '''