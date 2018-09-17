import torch

device = torch.device("cuda")

input_size = 1000
hidden_size = 100
output_size = 10
batch_size = 64

x = torch.randn(batch_size, input_size, device=device)  #input
t = torch.randn(batch_size, output_size, device=device) #label

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, output_size),
).to(device)

loss_func = torch.nn.MSELoss(size_average=False)


learning_rate = 1e-4

for i in range(500):
    y = model(x)
    loss = loss_func(y, t)
    print(i, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad
