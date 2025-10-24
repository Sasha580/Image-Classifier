import os
import csv
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Initialize the MiniPlaces dataset with the root directory for the images,
        the split (train/val/test), an optional data transformation,
        and an optional label dictionary.

        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []

        self.label_dict = label_dict if label_dict is not None else {}

        with open(os.path.join(self.root_dir, self.split + '.txt')) as r:
            lines = r.readlines()
            for line in lines:
                line = line.split()
                self.filenames.append(line[0])
                if split == 'test':
                    label = line[0]
                else:
                    label = int(line[1])
                self.labels.append(label)
                if split == 'train':
                    text_label = line[0].split(os.sep)[2]
                    self.label_dict[label] = text_label

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return a single image and its corresponding label when given an index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        if self.transform is not None:
            image = self.transform(
                Image.open(os.path.join(self.root_dir, "images", self.filenames[idx])))
        else:
                image = Image.open(os.path.join(self.root_dir, "images", self.filenames[idx]))
        label = self.labels[idx]
        return image, label    



class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


# class MyConv(nn.Module):
#     def __init__(self, num_classes=100, base=64):
#
#         super().__init__()
#
#         # Create a feature space
#         self.stem = nn.Conv2d(in_channels=3, out_channels=base, kernel_size=3, stride=1, padding=1, bias=False)
#
#         self.stage1 = self.make_stage(base, base, stride=1)  # Output: (64, 128, 128)
#         self.stage2 = self.make_stage(base, base * 2, stride=2)  # Output: (128, 64, 64)
#         self.stage3 = self.make_stage(base * 2, base * 4, stride=2)  # Output: (256, 32, 32)
#         self.stage4 = self.make_stage(base * 4, base * 8, stride=2)  # Output: (512, 16, 16)
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(base * 8, num_classes)
#         self.flatten = nn.Flatten()
#
#         self._init_identity_last_bn()
#
#     def make_stage(self, in_channels, out_channels, stride=1):
#         return nn.Sequential(
#             PreActBasicBlock(in_channels, out_channels, stride),
#             PreActBasicBlock(out_channels, out_channels, stride=1)
#         )
#
#     def _init_identity_last_bn(self):
#         # Zero-init the LAST BN gamma in each block
#         for m in self.modules():
#             if isinstance(m, PreActBasicBlock):
#                 nn.init.zeros_(m.bn2.weight)
#
#     def forward(self, x, return_intermediate=False):
#         x = self.stem(x)
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = self.flatten(self.pool(x))
#         x = self.fc(x)
#         return x


class MyConv(nn.Module):
    def __init__(self, num_classes=100, base=64):

        super().__init__()

        # Create a feature space
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res_block1 = BasicBlock(base, base, stride=2)
        self.res_block2 = BasicBlock(base, base * 2, stride=1)

        self.res_block3 = BasicBlock(base * 2, base * 2, stride=1)
        self.res_block4 = BasicBlock(base * 2, base * 2, stride=2)

        self.res_block5 = BasicBlock(base * 2, base * 4, stride=1)
        self.res_block6 = BasicBlock(base * 4, base * 4, stride=2)

        self.res_block7 = BasicBlock(base * 4, base * 8, stride=1)
        self.res_block8 = BasicBlock(base * 8, base * 8, stride=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 8 * 4 * 4, base * 8 * 4),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(base * 8 * 4, num_classes)
        )

    def make_stage(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x, return_intermediate=False):
        x = self.stem(x)  # Output: (64, 64, 64)

        x = self.res_block1(x)  # Output: (64, 32, 32)
        x = self.res_block2(x)  # Output: (128, 32, 32)

        x = self.res_block3(x)  # Output: (128, 32, 32)
        x = self.res_block4(x)  # Output: (128, 16, 16)

        x = self.res_block5(x)  # Output: (256, 16, 16)
        x = self.res_block6(x)  # Output: (256, 8, 8)

        x = self.res_block7(x)  # Output: (512, 8, 8)
        x = self.res_block8(x)  # Output: (512, 4, 4)

        x = self.classifier(x)  # Output: (num_classes, 1, 1)
        return x

    
def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the CNN classifier on the validation set.

    Args:
        model (CNN): CNN classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)
            

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy

def train(model, train_loader, val_loader, optimizer, criterion, device,
          num_epochs):
    """
    Train the CNN classifer on the training set and evaluate it on the validation set every epoch.

    Args:
    model (CNN): CNN classifier to train.
    train_loader (torch.utils.data.DataLoader): Data loader for the training set.
    val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    criterion (callable): Loss function to use for training.
    device (torch.device): Device to use for training.
    num_epochs (int): Number of epochs to train the model.
    """

    # Place the model on device
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train() # Set model to training mode

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch +1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the gradients for every batch
                optimizer.zero_grad()

                # Compute the logits and loss for the batch
                logits = model(inputs)
                loss = criterion(logits, labels)

                # Compute the gradients of the loss
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            results = f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
            print(results)

            # Write to log file
            with open('training_log_train6_basic_transformations.txt', 'a') as f:
                f.write(results + '\n')


def test(model, test_loader, device):
    """
    Get predictions for the test set.

    Args:
        model (CNN): classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        all_preds = []

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)

            logits = model(inputs)

            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)
    return all_preds
            
            

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy

def main(args):
    base = 64
    num_epochs = 50

    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])
    
    ## Define data transformation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    # train_transform = transforms.Compose([
    #     # geometric & scale
    #     transforms.RandomResizedCrop(128, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #
    #     # photometric
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    #     transforms.RandomGrayscale(p=0.05),
    #     transforms.RandomPerspective(distortion_scale=0.15, p=0.1),
    #     transforms.RandomRotation(degrees=10),
    #
    #     # to tensor + normalize
    #     transforms.ToTensor(),
    #     transforms.Normalize(image_net_mean, image_net_std),
    #
    #     # cutout-like reg (must be AFTER ToTensor)
    #     transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), inplace=True),
    # ])
    #
    # val_transform = transforms.Compose([
    #     transforms.Resize(144),  # slightly bigger
    #     transforms.CenterCrop(128),  # deterministic eval crop
    #     transforms.ToTensor(),
    #     transforms.Normalize(image_net_mean, image_net_std),
    # ])


    data_root = 'data'
    
    # Create MiniPlaces dataset object
    miniplaces_train = MiniPlaces(data_root,
                                  split='train',
                                  transform=train_transform)
    miniplaces_val = MiniPlaces(data_root,
                                split='val',
                                transform=val_transform,
                                label_dict=miniplaces_train.label_dict)

    # Create the dataloaders
    
    # Define the batch size and number of workers
    batch_size = 64
    num_workers = 8

    lr = 0.05 * (batch_size / 256)
    # lr = 0.001
    weight_decay = 2e-4
    momentum = 0.9

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(miniplaces_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(miniplaces_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps')

    model = MyConv(num_classes=len(miniplaces_train.label_dict), base=base)
                   

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if not args.test:

        train(model, train_loader, val_loader, optimizer, criterion,
              device, num_epochs=num_epochs)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()}, 'model.ckpt')

    else:
        miniplaces_test = MiniPlaces(data_root,
                                     split='test',
                                     transform=val_transform)
        test_loader = DataLoader(miniplaces_test,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)        
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        preds = test(model, test_loader, device)
        write_predictions(preds, 'predictions.csv')

def write_predictions(preds, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for im, pred in preds:
            writer.writerow((im, pred))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', default='model.ckpt')
    args = parser.parse_args()
    main(args)
