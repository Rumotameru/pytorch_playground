import torch.nn as nn
import Data
import torch


class MyLittleModel(nn.Module):

    def __init__(self, parameters, n_features, n_classes, act="linear"):
        self.act = act
        super(MyLittleModel, self).__init__()

        inp = parameters.copy()  # num of in-layers
        inp.insert(0, n_features)
        out = parameters.copy()  # num of out-layers
        out.append(n_classes)
        # deciding-future-architecture-by-given-parametrs
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(inp, out)])

    def forward(self, x):
        # linear-activator-prepare
        def linear(inp):
            return inp

        # deciding-activation-function
        if self.act == "relu":
            activation = torch.relu
        elif self.act == "tanh":
            activation = torch.tanh
        elif self.act == "sigmoid":
            activation = torch.sigmoid
        else:
            activation = linear

        for layer in self.layers:
            x = activation(layer(x))
        return x


def train(model, loader, t_loss, opt, criterion, device):

    for data, target in loader:
        # move-tensors-to-GPU
        data = data.to(device).float()
        target = target.to(device)
        target = target.to(dtype=torch.long)
        # clear-the-gradients-of-all-optimized-variables
        opt.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        opt.step()
        t_loss += loss.item() * data.size(0)

    return model, t_loss, opt


def valid(model, loader, v_loss, criterion, device):

    for data, target in loader:
        data = data.to(device).float()
        target = target.to(device)
        target = target.to(dtype=torch.long)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        loss = criterion(output, target)
        # update-average-validation-loss
        v_loss += loss.item() * data.size(0)

    return model, v_loss


def test(model, loader, device):

    correct = 0
    total = 0
    results = Data.pd.DataFrame()
    for data, target in loader:
        data = data.to(device).float()
        target = target.to(device)
        target = target.to(dtype=torch.long)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().float()
        # collect-resulting-data
        results = Data.results_collector(results, data.cpu().numpy(), predicted.cpu().numpy())
    results = results.drop_duplicates()
    accuracy(correct, total)

    return results


def accuracy(correct, total):
    print('\t Accuracy of the model: {} %'.format(100 * correct/total))
