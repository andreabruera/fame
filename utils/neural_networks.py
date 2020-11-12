import torch

#from torch import nn, optim

class torch_linear(torch.nn.Module):
    def __init__(self, output_dimensions, input_dimensions):
        self.output_dimensions = output_dimensions
        self.input_dimensions = input_dimensions
        super(torch_linear, self).__init__()
        self.linear1 = torch.nn.Linear(self.input_dimensions, self.output_dimensions)
        #self.linear1 = nn.Linear(self.input_dimensions, 800)
        #self.linear2 = nn.Linear(800, 500)
        #self.linear3 = nn.Linear(500, self.output_dimensions)

    def forward(self, x):
        x = self.linear1(x)
        #x = self.linear2(x)
        #x = self.linear3(x)
        return(x)

class torch_CNN(torch.nn.Module):

    def __init__(self, in_size, output_dimensions):
        super(torch_CNN, self).__init__()
        self.output_dimensions = output_dimensions
        self.in_size = in_size
        self.conv1 = torch.nn.Conv3d(1, 1, 3)
        self.max1 = torch.nn.MaxPool3d(10)


        # Determine output size of the convolutional part
        out = self.max1(self.conv1(torch.rand(self.in_size)))
        out_shape = out.view(-1).shape

        self.final_linear = torch.nn.Linear(out_shape[0], self.output_dimensions)

    def forward(self, x):

        x = x.view(self.in_size)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.max1(x)

        # Flatten the output ==> shape [channels * ? * ? * ?]
        x = x.view(1, -1)

        # Apply linear layer
        x = self.final_linear(x)

        return x

def train_torch_net(model, examples, targets):

    net = model[0]
    loss_function = model[1]
    cuda = model[2]

    net.apply(torch_weights_init)
    net.zero_grad()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer.zero_grad()
    #examples_reduced = [[k for k in example if k != 0.0] for example in examples]
    for epoch in range(10):
        for example_index, example in enumerate(examples):
        #for example_index, example in enumerate(examples_reduced):
            if type(example) != torch.Tensor:
                example = torch.cuda.FloatTensor(example, device=cuda).view(1, net.input_dimensions)
                target = torch.cuda.FloatTensor(targets[example_index], device=cuda).view(1, net.output_dimensions)
            else:
                target = targets[example_index]
            forward_pass = net(example)
            loss = loss_function(forward_pass, target, torch.cuda.FloatTensor([1], device=cuda))
            net.zero_grad()
            loss.backward()
            optimizer.step()
        #print(loss.item())

def torch_weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
