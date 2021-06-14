import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import Data
import Model
import Visualization


def training(params):

    features = params["additional_features"]
    batch_size = params["batch_size"]
    activator = params["activator"]
    learning_rate = params["learning_rate"]
    layers = params["layers"]
    num_epochs = params["epochs"]

    data, targets = Data.data_load()
    train_data, valid_data, test_data = Data.dataset_prepare(data, targets, features)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model.MyLittleModel(layers, len(features)+2, targets.nunique(), activator)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    train_losses = []
    valid_losses = []

    print(model)
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}:")

        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0

        # training-the-model
        model.train()
        model, train_loss, optimizer = Model.train(model, train_loader, train_loss, optimizer, criterion, device)

        # validate-the-model
        model.eval()
        model, valid_loss = Model.valid(model, valid_loader, valid_loss, criterion, device)

        # calculate-average-losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print-training/validation-statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(
            train_loss, valid_loss))

    # show-training/validation-relationship
    Visualization.print_losses(train_losses, valid_losses)

    # test-the-model
    with torch.no_grad():
        results = Model.test(model, test_loader, device)
    Visualization.show_result(data, targets, results)


if __name__ == "__main__":

    parameters ={"additional_features": ["x1^2"],  # "x1", "x2" are already in
                 # "x1^2", "x2^2", "x1x2", "sin(x1)","sin(x2)" are available
                 'layers': [2, 8, 5, 4],  # number of neurons for each layer
                 "activator": "tanh",  # "relu", "tanh", "sigmoid", "linear" are available. "linear" is default.
                 "epochs": 100,
                 "learning_rate": 0.03,
                 "batch_size": 15}
    training(parameters)
