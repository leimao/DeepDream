

from deepdream import deepdream
import argparse
import numpy as np
from PIL import Image

def main():

    parser = argparse.ArgumentParser(description = 'Designate function and keywords')
    group = parser.add_mutually_exclusive_group()

    # List the available layers and the number of channels
    # No input required
    group.add_argument('-l','--list', action = 'store_true', 
        help = 'List the available layers and the number of channels')
    # Preview the feature pattern of the neural network
    # Inputs: layer name, channel number
    group.add_argument('-p','--preview', nargs = 3, metavar=('layer_name', 'channel_number', 'output_filename'), 
        help = 'Preview the feature pattern of the neural network')
    # Render the image with the features from the neural network
    # Inputs: image path, layer name, channel number
    group.add_argument('-r', '--render', nargs = 4, metavar=('image_path', 'layer_name', 'channel_number', 'output_filename'), 
        help = 'Render the image with the features from the neural network')
    # Customize the image with the features from guide images
    # Inputs: image path, guide image path, layer name, channel number
    # This function is currently unavailable
    # group.add_argument('-c','--customize', nargs = 4, metavar=('image_path', 'guide_image_path', 'layer_name', 'channel_number'), 
    #    help = 'Customize the image with the features from guide images')
    args = parser.parse_args()

    if args.list:
        dream = deepdream()
        dream.show_layers()

    if args.preview:
        layer = str(args.preview[0])
        channel = int(args.preview[1])
        output_filename = str(args.preview[2])

        dream = deepdream()
        dream_obj = dream.T(layer = layer)[:,:,:,channel]
        dream.render_naive(t_obj = dream_obj, output_filename = output_filename, iter_n = 30)

    if args.render:
        image_path = str(args.render[0])
        layer = str(args.render[1])
        channel = int(args.render[2])
        output_filename = str(args.render[3])

        dream = deepdream()
        img0 = Image.open(image_path)
        img0 = np.float32(img0)
        dream_obj = dream.T(layer = layer)[:,:,:,channel]
        dream.render_deepdream(t_obj = dream_obj, img0 = img0, output_filename = output_filename, iter_n = 30)


if __name__ == '__main__':

    main()