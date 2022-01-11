from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Sequential(nn.Linear(784,256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2))
        self.hidden2 = nn.Sequential(nn.Linear(256,128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2)),
        self.fc1 = nn.Sequential(nn.Linear(128,10))
        
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.fc1(x)