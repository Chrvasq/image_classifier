import numpy as np

import argparse
import json

from PIL import Image

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from train import create_classifier


def input_args():
    parser = argparse.ArgumentParser(
        prog='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path',
                        default='flowers/test/15/image_06351.jpg',
                        type=str, help='image path for test image')
    parser.add_argument('--top_k', default=5, type=int,
                        help='set number of top results')
    parser.add_argument('--category_names', default='cat_to_name.json',
                        type=str, help='label names file')
    parser.add_argument('--gpu', action='store_true', help='enable GPU')
    parser.add_argument('--checkpoint', default='checkpoint.pth',
                        type=str, help='checkpoint file')

    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # Load pre-trained model
    if checkpoint['arch'] == 'densenet':
        model = models.densenet201(pretrained=True)
    elif checkpoint['arch'] == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        print('Missing arch')

    # Freeze params of pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Create classifier
    classifier = create_classifier(checkpoint['input_size'],
                                   checkpoint['hidden_units'],
                                   checkpoint['output_size'])

    model.classifier = classifier
    model.load_state_dict(checkpoint['model'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epochs']
    class_names = checkpoint['class_names']

    return model, optimizer, epochs, class_names


def process_image(image):
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image)
    image = image / 256

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std
    image = image.transpose((2, 0, 1))

    return image


def predict(image_path, model, topk, device):
    # Set model to eval mode
    model.eval()
    model.to(device)

    # Process image
    img = Image.open(image_path)
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)

    inputs = img.to(device)
    output = model.forward(inputs.float())
    ps = torch.exp(output)
    topk = ps.cpu().topk(topk)

    return (i.data.numpy().squeeze().tolist() for i in topk)


def main():
    args = input_args()
    image_path = args.image_path
    gpu = args.gpu
    top_k = args.top_k
    checkpoint = args.checkpoint
    labels = args.category_names

    # Debug print
    print('*'*5+'Hyperparameters'+'*'*5)
    print(f'Image path:     {image_path}')
    print(f'Gpu:            {gpu}')
    print(f'Checkpoint:     {checkpoint}')
    print(f'Category names: {labels}')
    print(f'Top_k:          {top_k}')

    # Check for gpu arg and verify is available, otherwise set CPU
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Current device: {device}')

    model, _, _, class_names = load_checkpoint(checkpoint)

    # Label Mapping
    with open(labels, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(image_path, model, top_k, device)

    flowers = [cat_to_name[class_names[i]] for i in classes]

    # Prediction
    print('*'*5+'Predictions'+'*'*5)

    for prob, flower in zip(probs, flowers):
        print('{:20}: {:.2f}%'.format(flower, prob * 100))


if __name__ == '__main__':
    main()
