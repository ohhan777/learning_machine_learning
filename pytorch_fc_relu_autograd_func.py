import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dx = grad_output.clone()
        dx[x < 0] = 0
        return dx


device = torch.device("cuda")

input_size = 1000
hidden_size = 100
output_size = 10
batch_size = 64

x = torch.randn(batch_size, input_size, device=device)  #input
t = torch.randn(batch_size, output_size, device=device) #label

w1 = torch.randn(input_size, hidden_size, device=device, requires_grad=True)
w2 = torch.randn(hidden_size, output_size, device=device, requires_grad=True)

learning_rate = 1e-6

for i in range(500):
    y = MyReLU.apply(x.mm(w1)).mm(w2)
    loss = (y - t).pow(2).sum()
    print(i, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
