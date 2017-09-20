

from deepdream import deepdream
import argparse



def main():

    parser = argparse.ArgumentParser(description = 'Designate function and keywords')
    group = parser.add_mutually_exclusive_group()

    # List the available layers and the number of channels
    # No input required
    group.add_argument('-l','--list', action = 'store_true', help = 'List the available layers and the number of channels')
    # Render the image with the features from the neural network
    # Inputs: image path, layer name, channel number
    group.add_argument('-r', '--render', nargs = 2, help = 'Render the image with the features from the neural network')
    # Customize the image with the features from guide images
    # Inputs: image path, guide image path, layer name, channel number
    group.add_argument('-c','--customize', nargs = 4, help = 'Customize the image with the features from guide images')




    if arg.





    group.add_argument('-s','--search', action = 'store_true', help = 'search article')
    group.add_argument('-r','--retrieve', action = 'store_true', help = 'retrieve article')
    parser.add_argument('keywords', type = unicode, help = 'keywords')
    args = parser.parse_args()

    if args.search:
        wiki = WikiAPI()
        results = wiki.search(term = args.keywords, limit = 50)
        print ("Query: %s" %args.keywords)
        print ("Number of Terms Found: %d" %len(results))
        print ("Search Results:")
        for result in results:
            print(result)

    if args.retrieve:
        wiki = WikiAPI()
        article = wiki.retrieve(title = args.keywords)
        print article.heading.encode('gbk', 'ignore')
        print('-'*50)
        print article.summary.encode('gbk', 'ignore')
        print('-'*50)
        print article.content.encode('gbk', 'ignore')





















    dream = deepdream()
    
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    
    '''
    # Picking some feature channel to visualize
    channel = 139
    dream_obj = dream.T(layer = layer)[:,:,:,channel]
    print(dream_obj.get_shape())

    #dream.render_naive(t_obj = dream_obj, output_filename = 'render_naive_demo.jpeg')


    img0 = Image.open('inputs/leaves.jpg')
    img0 = np.float32(img0)

    # I am not sure why this 'mixed4c' layer will work since it is not exactly in the layers.
    # To check all the layer names: 
    # http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/index.html
    layer = 'mixed4c'
    dream_obj = tf.square(dream.T(layer = layer))

    dream.render_deepdream(t_obj = dream_obj, img0 = img0, output_filename = 'render_large_demo.jpeg')
    '''
    dream.customize_deepdream(layer = layer, output_filename = 'customization_demo')



if __name__ == '__main__':
    main()