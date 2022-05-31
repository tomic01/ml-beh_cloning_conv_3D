#!/usr/bin/python3

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from model import Breakout3DConvModel
# BreakoutCnnModel2, BreakoutCnnModelDDQ, BreakoutCnnModel2, BreakoutDenseModel
import utils as gu
from fourChannelLoadDataset import CustomDataSet, CustomDataSet2

data_dir = './data'
data_save = './saves'


def main():
    # Images are loaded in a standard way (for pytorch), each of them have associated label
    dataset = CustomDataSet2(data_dir, transform=ToTensor())

    # Splitting data into two sets: 1. train 2. validation
    random_seed = 42  # fixed for reproducibility
    torch.manual_seed(random_seed)
    full_size = len(dataset)
    validation_size = int(full_size * 0.1)  # ~~10%
    train_size = full_size - validation_size
    train_ds, val_ds = random_split(dataset, [train_size, validation_size])
    assert len(train_ds) + len(val_ds) == len(dataset)

    # Define Train Dataset and Validation Dataset - Batches
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
    gu.show_batch(train_dl)

    # Load to GPU if possible
    device = gu.get_default_device()
    train_dl = gu.DeviceDataLoader(train_dl, device)
    val_dl = gu.DeviceDataLoader(val_dl, device)

    # THE MODEL #
    model = gu.to_device(Breakout3DConvModel(), device)
    print(model)

    # TRAINING #
    num_epochs = 10
    learning_rate = 0.001
    opt_func = torch.optim.Adam

    # Calculate Initial losses
    history = []
    initial_training_loss = []
    for batch in train_dl:
        loss = model.training_step(batch)
        initial_training_loss.append(loss)
        break
    result = gu.evaluate(model, val_dl)
    result['train_loss'] = torch.stack(initial_training_loss).mean().item()
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        0, result['train_loss'], result['val_loss'], result['val_acc']))
    history.append(result)

    # Training
    for i in range(num_epochs):
        i += 1
        print(f"Epoc{i}/{num_epochs}")
        his = gu.fit(1, learning_rate, model, train_dl, val_dl, opt_func)
        torch.save(model.state_dict(), data_save + f"/{i}_model-cnn.pth")
        history += his

    # Display accuracies and losses after the training
    gu.plot_accuracies(history)
    gu.plot_losses(history)


if __name__ == '__main__':
    main()
