# vgg16 tensorflow model source: https://github.com/tensorflow/models/tree/master/research/slim

import tensorflow as tf

from adapters.tensorflow_adapter import TensorflowAdapter

use_gpu = False

base_folder = '%DVT_ROOT%/../../models/vgg16_tensorflow/'
adapter = TensorflowAdapter(
    network_definition_filename=base_folder + './vgg.py',
    checkpoint_filename=base_folder + './vgg_16.ckpt',
    use_gpu=use_gpu,
    channels=3, height=224, width=224)

# input images
static_files_dir = '%DVT_ROOT%/' + './input_images/'

# UI customization
caffevis_label_layers    = ['fc8', 'prob']
caffevis_labels          = '%DVT_ROOT%/models/caffenet-yos/ilsvrc_2012_labels.txt'
caffevis_prob_layer      = 'prob'

def caffevis_layer_pretty_name_fn(name):
    return name.replace('pool','p').replace('norm','n')

# offline scripts configuration
caffevis_outputs_dir = base_folder + './outputs'
layers_to_output_in_offline_scripts = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                                       'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7',
                                       'fc8', 'prob']
