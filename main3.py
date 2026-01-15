import torch

###
# Rede neural muito simples, q são 2 perceptons sem ativaçao, com constante compartilhada, alterei para
# constante nao compartilhada

def teste(y_known):
    lr = 1e-2
    x = torch.Tensor([2,4]) #input data
    m = torch.randn(2, requires_grad=True) #parameter 1
    b = torch.randn(1, requires_grad=True) #parameter 2
    t=0
    while t<50:
        y = m*x+b #linear model
        print('Y', y)
        loss = ((y_known - y)**2).mean()#loss function
        print('loss', loss)
        loss.backward() #calculate gradients
        print('grad', m.grad)
        with torch.no_grad():
            m -= lr * m.grad
            b -= lr * b.grad
            m.grad.zero_()
            b.grad.zero_()
        t+=1

teste(torch.Tensor([10,8]))


##
# camada totalmente conectada (fully connected / linear layer), saida depende da mistura das entradas.
# Quando as saidas dependem das 2 entradas, os neuronios devem ter conexao

def teste2(y_known):
    lr = 1e-2
    x = torch.Tensor([2,4]) #input data
    m = torch.randn(2, requires_grad=True) #parameter 1
    b = torch.randn(1, requires_grad=True) #parameter 2
    t=0
    while t<50:
        y = m*x+b #linear model
        print('Y', y)
        loss = ((y_known - y)**2).mean()#loss function
        print('loss', loss)
        loss.backward() #calculate gradients
        print('grad', m.grad)
        with torch.no_grad():
            m -= lr * m.grad
            b -= lr * b.grad
            m.grad.zero_()
            b.grad.zero_()
        t+=1



X = torch.tensor([
    [2., 4.],   # -> y = [6, -2]
    [1., 3.],   # -> y = [4, -2]
    [-1., 2.],  # -> y = [1, -3]
    [0., 5.],   # -> y = [5, -5]
    [3., -2.],  # -> y = [1, 5]
], dtype=torch.float32)

Y = torch.tensor([
    [ 6., -2.],
    [ 4., -2.],
    [ 1., -3.],
    [ 5., -5.],
    [ 1.,  5.],
], dtype=torch.float32)

def teste_dataset(X, Y, lr=1e-2, steps=500):
    W = torch.randn(2, 2, requires_grad=True)
    b = torch.randn(2, requires_grad=True)

    for t in range(steps):
        y_pred = X @ W.T + b          # (N,2)  <- atenção: W.T
        loss = ((y_pred - Y)**2).mean()

        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
            W.grad.zero_()
            b.grad.zero_()

        if t % 100 == 0:
            print(t, "loss=", float(loss))

    return W.detach(), b.detach()

W_fit, b_fit = teste_dataset(X, Y)
print("W:", W_fit)
print("b:", b_fit)


##
## exemplo de uma rede feita usando o foward, que substitui o sequential. 
import torch.nn as nn


class TwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 4) ## 2 ramos em paralelo de entrada a e b cada um com 4 neuronios
        self.b = nn.Linear(2, 4)
        self.out = nn.Linear(8, 1) 

    def forward(self, x):
        h1 = self.a(x)
        h2 = self.b(x)
        h = torch.cat([h1, h2], dim=1)  # junta features: (batch, 8)
        return self.out(h)