import numpy as np

import argparse
import json

from PIL import Image

import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim


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

    # Freeze params of pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    # Resize image
    width, height = image.size
    size = 256

    height = int(size) if height > width else int(max(height * size/width, 1))
    width = int(size) if height < width else int(max(height * size/width, 1))

    resized_image = image.resize((width, height))

    # Crop Image
    crop_size = 224
    crop_width, crop_height = resized_image.size
    x1 = (crop_width - crop_size) / 2
    x2 = (crop_height - crop_size) / 2
    x3 = x1 + crop_size
    x4 = x2 + crop_size

    crop_image = resized_image.crop((x1, x2, x3, x4))

    # Convert color channel values
    np_image = np.array(crop_image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, topk, device, labels):
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
    with torch.no_grad():
        output = model.forward(inputs.float())
        ps = torch.exp(output)
        probs, classes = torch.topk(ps, topk)

        index_to_class = {
                    model.class_to_idx[x]: x for x in model.class_to_idx}
        top_classes = [
                    index_to_class[each] for each in classes.cpu().numpy()[0]]
        flower_names = [labels.get(str(each)) for each in top_classes]

    return probs.cpu().numpy()[0], top_classes, flower_names


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

    model = load_checkpoint(checkpoint)

    # Label Mapping
    with open(labels, 'r') as f:
        cat_to_name = json.load(f)

    probs, _, flower_names = predict(
                                image_path, model, top_k, device, cat_to_name)

    # Prediction
    print('*'*5+'Predictions'+'*'*5)

    for prob, flower_name in zip(probs, flower_names):
        print('{:20}: {:.2f}%'.format(flower_name, prob * 100))


if __name__ == '__main__':
    main()
