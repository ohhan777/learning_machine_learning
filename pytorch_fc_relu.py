import torch

device = torch.device('cuda')

input_size = 1000
hidden_size = 100
output_size = 10
batch_size = 64

x = torch.randn(batch_size, input_size, device=device)  #input
t = torch.randn(batch_size, output_size, device=device) #label

w1 = torch.randn(input_size, hidden_size, device=device)
w2 = torch.randn(hidden_size, output_size, device=device)

learning_rate = 1e-6

for i in range(500):
    a1 = x.mm(w1)           #a = x*w
    z1 = a1.clamp(min=0)   #relu(a)
    y = z1.mm(w2)

    loss = (y - t).pow(2).sum()
    print(i, loss.item())

    #back-propagation
    dy = 2.0 * (y - t)   #dL/dy
    dw2 = z1.t().mm(dy)  #dL/dw2
    dz1 = dy.mm(w2.t())  #dL/dz1
    da1 = dz1.clone()    #dL/da1
    da1[a1 < 0] = 0      #dL/da1
    dw1 = x.t().mm(da1)  #dL/dw1

    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2