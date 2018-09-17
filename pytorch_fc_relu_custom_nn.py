import torch

device = torch.device("cuda")

input_size = 1000
hidden_size = 100
output_size = 10
batch_size = 64

class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z1 = self.linear1(x).clamp(min=0)
        y = self.linear2(z1)
        return y

x = torch.randn(batch_size, input_size, device=device)
t = torch.randn(batch_size, output_size, device=device)

model = TwoLayerNet(input_size, hidden_size, output_size).to(device)

learning_rate = 1e-4

loss_func = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(500):
    y = model(x)
    loss = loss_func(y, t)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

