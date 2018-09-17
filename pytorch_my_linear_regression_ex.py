import torch

input_size = 100

x = torch.rand(input_size) * 10 - 5
y = 3 * x + 2 + torch.randn(input_size) * 0.1

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


loss_func = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

for i in range(500):
    pred = a * x + b
    loss = loss_func(pred, y)
    print(i, loss.item())
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad

        a.grad.zero_()
        b.grad.zero_()

print('a=', a.item(), ' b=', b.item())






