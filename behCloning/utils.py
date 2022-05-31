#!/usr/bin/python3

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np

def simple_show(torchtensor):
    plt.imshow(torchtensor.permute(1, 2, 0))
    plt.show()

# Display one image in dataset, given its index
def show_example(dataset, index):
    img, label = dataset[index]
    #print("Label:", dataset.classes[label], "(" + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

### Given the dataset, display a batch of images
def show_batch(dataset):
    for images, labels in dataset:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break # only one batch

def accuracy(outputs, lables):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == lables).item() / len(preds))


def get_default_device(ddevice='cuda'):
    if ddevice!='cuda':
        return torch.device('cpu')
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    #print("Training...")
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        i = 0
        for batch in train_loader:
            i+=1
            loss = model.training_step(batch)
            if i==1 or i%30==0:
                print(f"Batch({i}), loss({loss})")
                train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch+1, result)
        history.append(result)
    return history

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.show()

# Basic Info about provided/given data
def info(data_dir):
    actions = np.load(data_dir + '/actions.npy')
    episode_starts = np.load(data_dir + '/episode_starts.npy')
    obs = np.load(data_dir + '/obs.npy')
    rewards = np.load(data_dir + '/rewards.npy')

    print("---SHAPES---")
    print("\tActions ", actions.shape)
    print("\tEpisode_start ", episode_starts.shape)
    print("\tObservations ", obs.shape)
    print("\tRewards ", rewards.shape)

    print("---EXAMPLES---")

    print("\tACTIONS DATA")
    print("\tLast 10 elements :", actions[-10:-1].flatten())
    print("\tUnique set of action values: ", np.unique(actions[:]))

    print("\tEPISODE STARTS DATA")
    print("\tLast 10 elements :", episode_starts[-10:-1])

    print("\tOBSERVATION DATA")
    print("\tFirst and Last 10 elements :")
    print(obs[:10])
    print(obs[-10:-1])

    print("\tREWARDS DATA")
    print("\tLast 10 elements:", rewards[-10:-1].flatten())
    print("\tUnique set of reward values: ", np.unique(rewards[:]))