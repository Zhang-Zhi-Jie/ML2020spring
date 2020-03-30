import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data

class DataLoadAndPreprocess():

    def __init__(self, path_train, path_test, batch_size = 5652):
        self.path_train = path_train
        self.path_test = path_test
        self.train, self.test = self.load_data()
        self.processData()
        self.train_x, self.train_y = self.splitTrainData()
        self.test_x = self.splitTestData()
        dataset = Data.TensorDataset(self.train_x, self.train_y)
        self.train_dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)

    def load_data(self):
        train = pd.read_csv(self.path_train)
        test = pd.read_csv(self.path_test, header=None)
        return train, test
    
    def processData(self):
        self.train = self.train.iloc[:,2:].replace("NR", 0).to_numpy()
        self.test = self.test.iloc[:,2:].replace("NR", 0).to_numpy()
    
    def splitTrainData(self):
        month_data = {}
        for month in range(12):
            sample = np.empty([18, 480])
            for day in range(20):
                sample[:, day * 24 : (day + 1) * 24] = self.train[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
            month_data[month] = sample
        
        x = np.empty([12*471, 18*9], dtype = float)
        y = np.empty([12*471, 1], dtype = float)
        for month in range(12):
            for day in range(20):
                for hour in range(24):
                    if day == 19 and hour > 14:
                        continue
                    x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
                    y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
        
        mean_x = np.mean(x, axis = 0) #18 * 9 
        std_x = np.std(x, axis = 0) #18 * 9 
        for i in range(len(x)): #12 * 471
            for j in range(len(x[0])): #18 * 9 
                if std_x[j] != 0:
                    x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
        return self.to_tensor(x), self.to_tensor(y)
    
    def splitTestData(self):
        x = np.empty([240, 18*9], dtype = float)
        for i in range(240):
            x[i, :] = self.test[18 * i : 18 * (i + 1)].reshape(1, -1)

        mean_x = np.mean(x, axis = 0) #18 * 9 
        std_x = np.std(x, axis = 0) #18 * 9 
        for i in range(len(x)): #12 * 471
            for j in range(len(x[0])): #18 * 9 
                if std_x[j] != 0:
                    x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
        return self.to_tensor(x)

    def to_tensor(self, numpy_data):
        return torch.from_numpy(numpy_data)
