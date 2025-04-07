import torch
import matplotlib.pyplot as plot
from torch.autograd import Variable
import torch.optim
import numpy as np

# numpy data -- two dimensional array
x_data = np.linspace(-2, 2, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.2, x_data.shape)
y_data = np.square(x_data) + noise

# convert to tensor
x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)
inputs = Variable(x_data)
labels = Variable(y_data)

class NonLinearRegression(torch.nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.hidden = torch.nn.Linear(1, 30)
        self.relu = torch.nn.ReLU()
        self.predict = torch.nn.Linear(30, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.predict(x)
        return x


# show model parameters
model = NonLinearRegression()
for name, param in model.named_parameters():
    print("name:{} , parameter: {}".format(name, param))

# model shape
for name, param in model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}")


# start training
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
loss_func = torch.nn.MSELoss()
for epoch in range(5000):
    outputs = model(inputs)
    loss = loss_func(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("epoch:{}, loss:{}".format(epoch, loss.data.numpy()))

# save model
torch.save(model.state_dict(), "params/nonlinear.pt")

# evaluate model
# load weights
model.load_state_dict(torch.load("params/nonlinear.pt"))
y_pred = model(inputs)
plot.scatter(x_data, y_data)
plot.plot(x_data, y_pred.data.numpy(), 'r-', lw=3)
plot.show()