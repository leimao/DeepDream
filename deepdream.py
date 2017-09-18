'''
Google DeepDream Local API
Lei Mao
9/17/2017
Department of Computer Science
University of Chicago
Developed and Tested in Python 3.6
'''
import os
import sys
import zipfile
import numpy as np
from six.moves.urllib.request import urlretrieve
#import tensorflow as tf

NETWORK_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
NETWORK_FILE_ZIPPED = 'inception5h.zip'
SIZE_NETWORK_FILE_ZIPPED = 49937555
NETWORK_FILE = 'tensorflow_inception_graph.pb'
SIZE_NETWORK_FILE = 53884595
NETWORK_DIR = 'networks/'
DOWNLOAD_DIR = 'downloads/'
RAND_SEED = 0

np.random.seed(RAND_SEED)
#tf.set_random_seed(RAND_SEED)


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

    return graph, sess





def main():

    # Make directories for files
    make_directories(directories = [NETWORK_DIR, DOWNLOAD_DIR])
    # Download zipped networks files
    network_file_zipped = download_networks(url = NETWORK_URL, directory = DOWNLOAD_DIR, 
        filename = NETWORK_FILE_ZIPPED, expected_size = SIZE_NETWORK_FILE_ZIPPED)
    # Unzip network files
    network_file = unzip_files(src = network_file_zipped, dst = NETWORK_DIR, 
        foi = NETWORK_FILE, expected_size = SIZE_NETWORK_FILE)
    graph, sess = load_graph(model = network_file)



if __name__ == '__main__':
    main()
