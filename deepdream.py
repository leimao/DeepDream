'''
Google DeepDream Local API
Lei Mao
9/17/2017
Department of Computer Science
University of Chicago
Developed and Tested in TensorFlow 1.3 and Python 3.6
'''
import os
import sys
import zipfile
import numpy as np
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
from PIL import Image

NETWORK_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
NETWORK_FILE_ZIPPED = 'inception5h.zip'
SIZE_NETWORK_FILE_ZIPPED = 49937555
NETWORK_FILE = 'tensorflow_inception_graph.pb'
SIZE_NETWORK_FILE = 53884595
NETWORK_DIR = 'networks/'
DOWNLOAD_DIR = 'downloads/'
INPUT_DIR = 'inputs/'
OUTPUT_DIR = 'outputs/'
RAND_SEED = 0

# No random process
# np.random.seed(RAND_SEED)
# tf.set_random_seed(RAND_SEED)

# File operation functions

def make_directories(directories):
    '''
    Make directories for the files.
    '''
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder \"%s\" created." %directory.strip('/'))

last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):
  '''
  A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  '''
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def unzip_files(src, dst, foi = None, expected_size = None):
    '''
    Unzip all files from src to dst.
    '''
    with zipfile.ZipFile(src, 'r') as zip_ref:
            zip_ref.extractall(dst)
    # Check file of interest (foi) in the unzipped files
    if foi:
        foi_path = dst + foi
        statinfo = os.stat(foi_path)
        if statinfo.st_size == expected_size:
            print("File %s found and verified." %foi)
        else:
            raise Exception('Failed to verify ' + filename + ' extracted.')

        return foi_path

def download_networks(url, directory, filename, expected_size):
    '''
    Download files from internet.
    '''
    # Download
    file_path = directory + filename
    if not os.path.exists(file_path):
        print('Attempting to download: %s' % filename) 
        file_path, _ = urlretrieve(url, file_path, reporthook = download_progress_hook)
        print('\nDownload complete!')
    
    statinfo = os.stat(file_path)
    if statinfo.st_size == expected_size:
        print('File %s found and verified.' % filename)
    else:
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return file_path

def showarray(a):
    '''
    Show image from numpy array.
    '''
    im = Image.fromarray(a)
    im.show()

def savearray(a, file_path):
    '''
    Save image from numpy.
    '''
    im = Image.fromarray(a)
    im.save(file_path)

def visstd(a, s = 0.1):
    '''
    Normalize image range for visualization.
    '''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


class deepdream(object):

    def __init__(self):

        # The file path of model
        self.model = self.prepare_networks()
        # Initialize the model
        self.load_graph(model = self.model)
        # Transform the resize function
        # This is what they did, but I am not sure why.
        self.resize = self.tffunc(np.float32, np.int32)(self.resize)

    def prepare_networks(self):
        '''
        Prepare neural network files if not exist.
        '''
        # Make directories for files
        make_directories(directories = [NETWORK_DIR, DOWNLOAD_DIR, INPUT_DIR, OUTPUT_DIR])
        if not os.path.exists(NETWORK_DIR + NETWORK_FILE):
            # Download zipped networks files
            network_file_zipped = download_networks(url = NETWORK_URL, directory = DOWNLOAD_DIR, 
                filename = NETWORK_FILE_ZIPPED, expected_size = SIZE_NETWORK_FILE_ZIPPED)
            # Unzip network files
            network_file = unzip_files(src = network_file_zipped, dst = NETWORK_DIR, 
                foi = NETWORK_FILE, expected_size = SIZE_NETWORK_FILE)
        else:
            network_file = NETWORK_DIR + NETWORK_FILE

        return network_file
    
    def load_graph(self, model):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph = self.graph)
        with tf.gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Define input tensor
        self.t_input = tf.placeholder(np.float32, name='input')
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self.t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input':t_preprocessed})
        print('Model loading complete!')

        layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
        print('Number of layers in the model: %d.' %len(layers))
        print('Total number of feature channels in the model: %d.' %sum(feature_nums))

    def T(self, layer):
        '''
        Helper for getting layer output tensor.
        '''
        return self.graph.get_tensor_by_name("import/%s:0"%layer)

    def render_naive(self, t_obj, output_filename, img0 = None, iter_n = 20, step = 1.0):
        '''
        Render user's image with learned pattern from neural network, and output the image.
        But the image has to be the original size of t_input.
        '''
        # If no input image provided, generate a random image with noise.
        if img0 is None:
            img0 = np.random.uniform(size=(224,224,3)) + 100.0

        # Define the optimization objective
        t_score = tf.reduce_mean(t_obj)
        # Automatic differentiation
        t_grad = tf.gradients(t_score, self.t_input)[0]
        
        img = img0.copy()
        for _ in range(iter_n):
            g, _ = self.sess.run([t_grad, t_score], {self.t_input: img})
            # Normalizing the gradient, so the same step size should work 
            # for different layers and networks
            g /= g.std()+1e-8
            img += g*step

        # Image normalization
        img = visstd(img)
        # Image clip and rescale 256 color
        img = np.uint8(np.clip(img, 0, 1)*255)
        # Show image
        showarray(img)
        # Save image
        file_path = OUTPUT_DIR + output_filename
        savearray(img, file_path)
        print('Image \"%s\" saved.' %output_filename)

    def tffunc(self, *argtypes):
        '''
        Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session = kw.get('session'))
            return wrapper
        return wrap
    
    def resize(self, img, size):
        '''
        Resize image.
        '''
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    
    def calc_grad_tiled(self, img, t_grad, tile_size = 224):
        '''
        Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.
        '''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size = 2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = self.sess.run(t_grad, {self.t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0) 

    def render_deepdream(self, t_obj, output_filename, img0 = None, iter_n = 10, step = 1.5, octave_n = 4, octave_scale = 1.4):
        '''
        Render user's image with learned pattern from neural network, and output the image.
        The image could be any size.
        '''
        # If no input image provided, generate a random image with noise.
        if img0 is None:
            img0 = np.random.uniform(size = (224,224,3)) + 100.0

        # Define the optimization objective
        t_score = tf.reduce_mean(t_obj)
        # Automatic differentiation
        t_grad = tf.gradients(t_score, self.t_input)[0]
    
        # Split the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-self.resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        # Generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))


            # Image clip and rescale 256 color
            img = np.uint8(np.clip(img, 0, 255))
            # Show image
            showarray(img)

        # Save image
        file_path = OUTPUT_DIR + output_filename
        savearray(img, file_path)
        print('Image \"%s\" saved.' %output_filename)


    def render_deepdreamvideo():
        '''
        In progress.
        '''

    def get_layer_name_maximum_dot_product():
        '''
        Find the layer name of the maximum of the dot products between the corresponding layer tensors
        '''

    def get_features_from_graph(self, layer, img):
        '''
        Get deep features of image from the layer specified in the graph. 
        '''

        t_obj = self.T(layer = layer)
        g = self.sess.run(t_obj, {self.t_input: img})

        return g[0].transpose(2,0,1)

    def calc_grad_tiled_maximum_dot_product(self, layer, img, guide_features, tile_size = 224):
        '''
        Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.
        t_obj is selected to layer where the product of guide_feature and corresponding img features are maximum. 
        '''
        sz = tile_size
        h, w = img.shape[:2]
        print('########')
        print(img.shape)

        print('########')

        sx, sy = np.random.randint(sz, size = 2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                print(y)
                print(y+sz)
                print('########')
                print(img_shift.shape)
                print(sub.shape)
                print('########')


                sub_features = self.get_features_from_graph(layer = layer, img = sub)
                sub_features = sub_features.reshape(sub_features.shape[0], -1)

                dot_products = sub_features * (guide_features.T)
                maximum_index = dot_products.argmax(axis = 1)

                guide_features_correspondence = guide_features[maximum_index, :]
                
                t_obj = tf.reshape(tf.transpose(self.T(layer = layer), perm = [2,0,1]), [-1])
                t_score = tf.matmul(t_obj, (guide_features_correspondence.reshape(-1).T))
                # Automatic differentiation
                t_grad = tf.gradients(t_score, self.t_input)[0]

                g = self.sess.run(t_grad, {self.t_input:sub})
                grad[y:y+sz,x:x+sz] = g

        return np.roll(np.roll(grad, -sx, 1), -sy, 0) 


    def customize_deepdream(self, layer, output_filename, guide_img = None, img0 = None, iter_n = 10, step = 1.5, octave_n = 4, octave_scale = 1.4):
        '''
        guide_image has to be the same size as the input of the neural network.
        '''
        # If no input image provided, generate a random image with noise.
        if guide_img is None:
            guide_img = np.random.uniform(size = (224,224,3)) + 100.0
        # If no input image provided, generate a random image with noise.
        if img0 is None:
            img0 = np.random.uniform(size = (224,224,3)) + 100.0

        print(guide_img.shape)
        guide_features = self.get_features_from_graph(layer = layer, img = guide_img)
        guide_features = guide_features.reshape(guide_features.shape[0], -1)

        # Split the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-self.resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        # Generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = self.calc_grad_tiled_maximum_dot_product(layer = layer, img = img, guide_features = guide_features)
                img += g*(step / (np.abs(g).mean()+1e-7))


            # Image clip and rescale 256 color
            img = np.uint8(np.clip(img, 0, 255))
            # Show image
            showarray(img)

        # Save image
        file_path = OUTPUT_DIR + output_filename
        savearray(img, file_path)
        print('Image \"%s\" saved.' %output_filename)


def main():

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
