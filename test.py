from multiprocessing import cpu_count

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from sen import SEN

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# x_0 = torch.randn(100, 50)
# x_1 = torch.randn(100, 8)
# x = torch.cat((x_0, x_1), 1).type(torch.FloatTensor)


# y_0 = torch.zeros(50)
# y_1 = torch.ones(50)
# y = torch.cat((y_0, y_1), ).type(torch.FloatTensor).view(-1, 1)
#
# x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):

    def __init__(self, feature, hidden, output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(feature, hidden)
        self.output = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.output(x))
        return x


train_data = SEN(
    data_dir='./datasets',
    create_data=False,
    max_sequence_length=60
)


net = Net(58, 16, 8).to(device)
writer = SummaryWriter('logs/test')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(
    dataset=train_data,
    batch_size=512,
    shuffle=True,
    num_workers=cpu_count(),
    pin_memory=torch.cuda.is_available()
)


for epoch in range(5000):

    for iteration, batch in enumerate(train_loader):

        batch_size = len(batch['answer'])

        x_0 = torch.randn(batch_size, 50)

        x_1 = torch.randn(batch_size, 8)

        x = torch.cat((x_0, x_1), 1).type(torch.FloatTensor).to(device)
        # y = torch.zeros(batch_size).type(torch.LongTensor).to(device)
        y = torch.from_numpy(np.array(batch['answer']).astype(dtype=float)).type(torch.LongTensor).to(device)

        x, y = Variable(x), Variable(y)

        predict = net(x)

        loss = loss_func(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar(tag='loss', scalar_value=loss.data.item(), global_step=epoch)

    print(loss)
