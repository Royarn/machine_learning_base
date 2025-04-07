import torch
import matplotlib.pyplot as plot
from torch.autograd import Variable
import torch.optim
import numpy as np

# numpy data -- one dimensional array
x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = 0.5 + 0.3 * x_data + noise

plot.scatter(x_data, y_data)
plot.show()


# convert to tensor -- multi dimensional array
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

# convert to tensor
x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)
inputs = Variable(x_data)
labels = Variable(y_data)


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# view model parameters
for name, param in model.named_parameters():
    print("model parameters:================")
    print("name:{} , parameter: {}".format(name, param))

for epoch in range(1000):
    outputs = model(inputs)
    # loss
    loss = criterion(outputs, labels)
    # gradient zero
    optimizer.zero_grad()
    # backward propagation
    loss.backward()
    # update parameters
    optimizer.step()
    if epoch % 200 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

# save model
torch.save(model.state_dict(), "params/linear.pt")

# load model
model = LinearRegression()
model.load_state_dict(torch.load("params/linear.pt"))
model.eval()
# view model parameters
print("model parameters:================")
for name, param in model.named_parameters():
    print("name:{} , parameter: {}".format(name, param))

# evaluate model
y_pred = model(inputs)
plot.scatter(x_data, y_data)
plot.plot(x_data, y_pred.data.numpy(), 'r-', lw=3)
plot.show()