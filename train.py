import time
import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def input_args():
    parser = argparse.ArgumentParser(
        prog='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', default='flowers',
                        type=str, help='set directory to save checkpoint')
    parser.add_argument('--arch', default='densenet', type=str,
                        help='choose model, densenet or vgg')
    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help='set learning rate')
    parser.add_argument('--hidden_units', default=None,
                        type=int, nargs='+',
                        help='list of ints to set size of hidden layers')
    parser.add_argument('--epochs', default=4,
                        type=int, help='set number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='enable GPU')

    return parser.parse_args()


def train_model(dataloader, model, criterion, optimizer, device, epochs):
    print_every = 40
    steps = 0
    start = time.time()

    for e in range(epochs):
        running_loss = 0

        for inputs, labels in dataloader['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track loss and accuracy on validation set
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0

                    for inputs, labels in dataloader['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model.forward(inputs)
                        valid_loss += criterion(outputs, labels).item()

                        ps = torch.exp(outputs)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                print(
                    'Epoch: {}/{}.. '.format(
                                        e+1, epochs
                                            ),
                    'Training Loss: {:.3f}.. '.format(
                                                running_loss/print_every
                                                    ),
                    'Valid Loss: {:.3f}.. '.format(
                                            valid_loss/len(
                                                dataloader['valid'])),
                    'Test Accuracy: {:.3f}'.format(
                                            accuracy/len(
                                                dataloader['valid'])))
                running_loss = 0

                model.train()

    total_time = time.time() - start

    print("\n** Total time to train model:", str(
        int((total_time / 3600))) + ":" +
        str(int((total_time % 3600) / 60)) + ":" + str(
            int((total_time % 3600) % 60)))

    return model


def test_model(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        'Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


def create_classifier(num_input_features, hidden_units, num_output_features):
    count = 0
    classifier = nn.Sequential()

    if hidden_units is None:
        classifier.add_module('fc0', nn.Linear(
                                num_input_features, num_output_features))
    else:
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        classifier.add_module('fc0', nn.Linear(
                                num_input_features, hidden_units[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('dropout0', nn.Dropout(p=0.5))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('dropout'+str(i+1), nn.Dropout(p=0.5))
            count = i
        classifier.add_module('fc'+str(count+1), nn.Linear(
                                hidden_units[-1], num_output_features))
        classifier.add_module('output', nn.LogSoftmax(dim=1))

    return classifier


def save_checkpoint(num_input_features, class_names, epochs, model, optimizer,
                    arch, hidden_units):
    checkpoint = {'input_size': num_input_features,
                  'output_size': 102,
                  'hidden_units': hidden_units,
                  'epochs': epochs,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_names': model.class_names,
                  'arch': arch}

    torch.save(checkpoint, 'checkpoint.pth')

    print('Checkpoint saved successfully!')


def main():
    args = input_args()
    dir = args.dir
    gpu = args.gpu
    arch = args.arch
    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs

    # Debug print
    print('*'*5+'Hyperparameters'+'*'*5)
    print(f'Data dir:      {dir}')
    print(f'Model:         {arch}')
    print(f'Hidden layers: {hidden_units}')
    print(f'Learning rate: {lr}')
    print(f'Epochs:        {epochs}')

    train_dir = dir + '/train'
    valid_dir = dir + '/valid'
    test_dir = dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_and_valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(
                                train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(
                                test_dir, transform=test_and_valid_transforms)
    valid_data = datasets.ImageFolder(
                                valid_dir, transform=test_and_valid_transforms)

    # Define the dataloaders
    dataloader = {
        'train': torch.utils.data.DataLoader(
            train_data, batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(
            test_data, batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(
            valid_data, batch_size=32, shuffle=True)
            }

    # Check for gpu arg and verify is available, otherwise set CPU
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Current device: {device}')

    # Check for pretrained model input
    if arch == 'densenet':
        model = models.densenet201(pretrained=True)
        num_input_features = model.classifier.in_features
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
        num_input_features = model.classifier[0].in_features
    else:
        print("Please choose 'densenet' or 'vgg'.")

    # Freeze params of pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Update classifier with created classifier

    classifier = create_classifier(num_input_features, hidden_units, 102)

    if arch == 'densenet':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    elif arch == 'vgg':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    else:
        pass

    # Classifier architecture
    print('*'*5+'Classifier'+'*'*5)
    print('The classifier architecture:')
    print(classifier)

    criterion = nn.NLLLoss()

    model.to(device)

    # Train model
    print('*'*5+'Training'+'*'*5)
    model = train_model(dataloader,
                        model,
                        criterion,
                        optimizer,
                        device,
                        epochs)

    model.class_to_idx = train_data.class_to_idx
    class_names = train_data.classes
    model.class_names = class_names

    # Test model
    print('*'*5+'Testing'+'*'*5)
    test_model(dataloader, model, device)

    # Save checkpoint
    print('*'*5+'Saving Checkpoint'+'*'*5)
    save_checkpoint(num_input_features,
                    class_names,
                    epochs,
                    model,
                    optimizer,
                    arch,
                    hidden_units)


if __name__ == '__main__':
    main()
