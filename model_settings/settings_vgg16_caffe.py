# vgg16 caffe model source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

from adapters.caffe_adapter import CaffeAdapter

use_gpu = False

base_folder = '%DVT_ROOT%/../../models/vgg16_caffe/'
adapter = CaffeAdapter(
    deploy_prototxt_filepath=base_folder + './VGG_ILSVRC_16_layers_deploy.prototxt',
    network_weights_filepath=base_folder + './VGG_ILSVRC_16_layers.caffemodel',
    data_mean_ref=(103.939, 116.779, 123.68),
    use_gpu=use_gpu)

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
