from data.data_preprocess import DataLoadAndPreprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.linear import LinearNet
from utils.saveResult import import_csv
from torch.nn import init

train_path = "./data/train2.csv"
test_path = "./data/test.csv"
data = DataLoadAndPreprocess(train_path, test_path)

device = "cuda" if torch.cuda.is_available else "cpu"
train_loader = data.train_dataloader
criterion = nn.MSELoss()

def train(net, train_loader, criterion, lr, epochs, device):
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    for epoch in range(epochs):
        for X, y in train_loader:
            X = X.clone().detach().float()
            y = y.clone().detach().float()
            y_hat = net(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch % 100 == 0:
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, loss.item()))

if __name__ == "__main__":
    net = LinearNet()
    train(net, train_loader, criterion, 0.02, 2000, device)
    torch.save(net.state_dict(), "./checkpoint/model.pt")
    net.load_state_dict(torch.load("./checkpoint/model.pt"))
    result = net(data.test_x.clone().detach().float()).detach().numpy()
    result_PATH = "./checkpoint/submit.csv"
    import_csv(result, result_PATH)

    
    