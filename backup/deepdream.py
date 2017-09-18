'''
Google DeepDream Local API
Lei Mao
9/17/2017
Department of Computer Science
University of Chicago
Developed and Tested in Python 3.6
'''
import os
from io import BytesIO
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

    return

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

    return

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


def prepare_networks():
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


class deepdream(object):

    def __init__(self):

        # The file path of model
        self.model = self.prepare_networks()
        # Initialize the model
        self.graph, self.sess, self.t_input = self.load_graph(model = self.model)




    def prepare_networks():
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
    

    def load_graph(model):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        graph = tf.Graph()
        sess = tf.InteractiveSession(graph = graph)
        with tf.gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Define input tensor
        t_input = tf.placeholder(np.float32, name='input')
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input':t_preprocessed})
        print('Model loading complete!')

        layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
        print('Number of layers: %d.' %len(layers))
        print('Total number of feature channels: %d.' %sum(feature_nums))

        return graph, sess, t_input
























# TensorFlow helper functions

def load_graph(model):
    '''
    Lode trained model.
    '''
    print('Loading model...')
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph = graph)
    with tf.gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Define input tensor
    t_input = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})
    print('Model loading complete!')

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
    print('Number of layers: %d.' %len(layers))
    print('Total number of feature channels: %d.' %sum(feature_nums))

    return graph, sess, t_input


def showarray(a):
    '''
    Show image from numpy array.
    '''
    im = Image.fromarray(a)
    im.show()
    return

def savearray(a, file_path):
    '''
    Save image from numpy.
    '''
    im = Image.fromarray(a)
    im.save(file_path)
    return

def visstd(a, s = 0.1):
    '''
    Normalize image range for visualization.
    '''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer, graph):
    '''
    Helper for getting layer output tensor.
    '''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_naive(t_obj, t_input, output_filename, graph, sess, img0 = None, iter_n = 20, step = 1.0):
    '''
    Render learned pattern in certain layer of originial input size, and output the learned pattern as image file.
    '''
    # If no input image provided, generate a random image with noise.
    if img0 == None:
        img0 = np.random.uniform(size=(224,224,3)) + 100.0

    # Define the optimization objective
    t_score = tf.reduce_mean(t_obj)
    # Automatic differentiation
    t_grad = tf.gradients(t_score, t_input)[0]
        
    img = img0.copy()
    for _ in range(iter_n):
        g, _ = sess.run([t_grad, t_score], {t_input: img})
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

    return


def main():

    # Make directories for files
    make_directories(directories = [NETWORK_DIR, DOWNLOAD_DIR, INPUT_DIR, OUTPUT_DIR])
    # Download zipped networks files
    network_file_zipped = download_networks(url = NETWORK_URL, directory = DOWNLOAD_DIR, 
        filename = NETWORK_FILE_ZIPPED, expected_size = SIZE_NETWORK_FILE_ZIPPED)
    # Unzip network files
    network_file = unzip_files(src = network_file_zipped, dst = NETWORK_DIR, 
        foi = NETWORK_FILE, expected_size = SIZE_NETWORK_FILE)
    # Initialize the model
    graph, sess, t_input = load_graph(model = network_file)

    # Render learned pattern in a naive way
    # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
    # to have non-zero gradients for features with negative initial activations.
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139 # picking some feature channel to visualize
    t_obj = T(layer = layer, graph = graph)[:,:,:,channel]
    render_naive(t_obj = t_obj, t_input = t_input, output_filename = 'render_naive_demo.jpeg', graph = graph, sess = sess)





if __name__ == '__main__':
    main()
