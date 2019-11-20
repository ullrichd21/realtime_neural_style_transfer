import argparse
import os
import sys
import time
import re

import numpy as np
import cv2
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

def stylize(img):
    device = torch.device("cpu")

    content_image = img#, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load("./smodels/mosaic.pth")
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    return utils.return_image(output[0])

def main():
    # main_arg_parser = argparse.ArgumentParser(description="parser for neural style transfer")
    #
    # main_arg_parser.add_argument("--content-image", type=str, required=True,
    #                              help="path to content image you want to stylize")
    # main_arg_parser.add_argument("--output-image", type=str, required=True,
    #                              help="path for saving the output image")
    # main_arg_parser.add_argument("--model", type=str, required=True,
    #                              help="saved model to be used for stylizing the image.")
    # main_arg_parser.add_argument("--cuda", type=int, required=True,
    #                              help="set it to 1 for running on GPU, 0 for CPU")
    #
    # args = main_arg_parser.parse_args()
    #
    # if args.cuda and not torch.cuda.is_available():
    #     print("ERROR: cuda is not available, try running on CPU")
    #     sys.exit(1)

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = stylize(img)
        cv2.imshow("Style Transfer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # stylize(img)


if __name__ == "__main__":
    main()
