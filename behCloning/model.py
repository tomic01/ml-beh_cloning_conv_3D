import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as gu


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        # device = gu.get_default_device()
        # loss = F.cross_entropy(out, labels, weight=class_weights)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = gu.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        ret = {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        return ret

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# The idea for this model comes from DeepMind Paper
# Human-level control through deep reinforcement learning
class BreakoutCnnModelDDQ(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> 32, 20, 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64, 9, 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64, 7, 7
            nn.ReLU(),
            nn.Flatten(),  # => 3136
            # nn.ReLU(),
            nn.Linear(64 * 7 * 7, 512),  # -> 512
            nn.ReLU(),
            # nn.Linear(512, 128),  # -> 128
            # nn.ReLU(),
            nn.Linear(512, 4),  # -> 4
        )

    def forward(self, xb):
        return self.network(xb)


class BreakoutCnnModel2(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            # -> 32, 20, 20 (we have 8 kernels, each applied to an input channel)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64, 9, 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64, 7, 7
            nn.ReLU(),  # TODO: nn.Dropout(0.2)
            nn.Flatten(),  # => 3136
            # nn.ReLU(),
            nn.Linear(64 * 7 * 7, 512),  # -> 512
            nn.ReLU(),
            nn.Linear(512, 4),  # -> 4
        )

    def forward(self, xb):
        return self.network(xb)


# Simple Dense/Linear model
class BreakoutDenseModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(28224, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )

    def forward(self, xb):
        xb = xb.reshape(-1, 28224)
        out = self.network(xb)
        return out


# Since the input is 3D (different time steps in each image), try 3D convolution
class Breakout3DConvModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(2, 8, 8), stride=(1, 4, 4)),  # 32, 3, 20, 20 (or 21?)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(2, 4, 4), stride=(1, 2, 2)),  # 64, 2, 9, 9
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1)),  # 64, 1, 7, 7
            nn.ReLU(),
            nn.Flatten(),  # 64 * 7 * 7= 3136
            nn.Linear(64 * 7 * 7, 512),  # -> 512
            nn.ReLU(),
            nn.Linear(512, 4),  # -> 4
        )

    def forward(self, xb):
        xb = torch.unsqueeze(xb, 1)
        return self.network(xb)
