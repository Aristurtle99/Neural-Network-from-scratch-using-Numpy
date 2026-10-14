import numpy as np
import torch
import torch.nn as nn




torch.manual_seed(42)
np.random.seed(0)
X = np.random.randn(2, 400)                              
Y = (X[0]**2 + X[1]**2 < 1).astype(int).reshape(1, 400)  

X = torch.tensor(X.T, dtype=torch.float32)
Y = torch.tensor(Y.T, dtype=torch.float32)


class TwoLayerNet(nn.Module):
    def __init__(self, n_x, n_h, n_y):
        super().__init__()                 
        self.layer1 = nn.Linear(n_x, n_h)
        self.layer2 = nn.Linear(n_h,n_y)

    def forward(self, X):
        Z1 = self.layer1(X)
        A1 = torch.tanh(Z1)
        Z2 = self.layer2(A1)
        A2 = torch.sigmoid(Z2)

        return A2


model = TwoLayerNet(2, 4, 1)
print(model)
print(model.layer1.weight.shape, model.layer1.bias.shape)
A2 = model(X)
print(A2.shape)



loss_fn = nn.BCELoss()                                    
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for i in range(10000):
    optimizer.zero_grad()       
    A2 = model(X)               
    loss = loss_fn(A2, Y)        
    loss.backward()             
    optimizer.step()            

    if i % 1000 == 0:
        print("Iteration %d: L = %.4f" % (i, loss.item()))
preds = (A2 > 0.5).float()
print("accuracy:", (preds == Y).float().mean().item())
