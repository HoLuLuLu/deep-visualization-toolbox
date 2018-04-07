
from adapters.base_adapter import BaseAdapter
from settings_misc import replace_magic_DVT_ROOT


class TensorflowAdapter(BaseAdapter):

    def __init__(self, network_definition_filename, checkpoint_filename, use_gpu, channels, height, width):
        """
        Ctor of TensorflowAdapter class
        """

        self._network_definition_filename = replace_magic_DVT_ROOT(network_definition_filename)
        self._checkpoint_filename = replace_magic_DVT_ROOT(checkpoint_filename)
        self._use_gpu = use_gpu
        self._batch_size = 1
        self._channels = channels
        self._height = height
        self._width = width

        pass

    def init_thread_specific(self):
        pass

    def load_network(self, settings):

        import tensorflow as tf
        slim = tf.contrib.slim

        # dynamically load network definition module
        import imp
        import os
        filepath = os.path.abspath(self._network_definition_filename)
        mod_name, file_ext = os.path.splitext(os.path.split(filepath)[-1])
        py_mod = imp.load_source(mod_name, filepath)

        # TODO: change if can this be deduced from the graph
        inputs_placeholder = tf.placeholder(tf.float32, [self._batch_size, self._height, self._width, self._channels], name='inputs')

        # TODO: change hardcoded network name
        net, end_points = py_mod.vgg_16(inputs_placeholder)

        # Get list of variables to restore (which contains only 'v2')
        variables_to_restore = slim.get_variables_to_restore()
        restorer = tf.train.Saver(variables_to_restore)

        if not self._use_gpu:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            self._session = tf.Session(config=config)
        else:
            self._session = tf.Session()

        # Restore variables from disk.
        restorer.restore(self._session, "/home/local/ANT/arikp/models/vgg16_tensorflow/vgg_16.ckpt")
        print("Model restored.")
        # Do some work with the model

        pass


    def get_data_mean(self):
        pass

    def get_layers_list(self):

        import tensorflow as tf

        # TODO: change hardcoded part
        prefix = 'vgg_16'

        layers_list = [n.name[(len(prefix)+1):-(1 + len('weights'))] for n in tf.get_default_graph().as_graph_def().node
                       if 'Variable' in n.op and n.name.endswith('/weights')]

        return layers_list

    def get_input_shape(self):
        return self._batch_size, self._channels, self._height, self._width

    def get_blob_info(self, layer_name):
        pass

    def get_layer_weights(self, layer_name):
        pass

    def get_layer_bias(self, layer_name):
        pass

    def get_layer_data(self, layer_name):

        pass

    def get_layer_diff(self, layer_name):
        pass

    def forward(self, img):
        pass

    def forward_all(self, data):
        pass

    def predict(self, im_batch, oversample=False):
        pass

    def backward_from_layer(self, start_layer_name, start_diff, diffs, zero_higher):
        pass

    def deconv_from_layer(self, start_layer_name, start_diff, diffs, zero_higher, deconv_type):
        pass

    def set_network_batch_size(self, batch_size):
        pass

    def get_channel_swap(self):
        """
        get list of channels for converting network inputs to RGB
        if the network was trained on BGR inputs, the return value should probably be [2, 1, 0]
        if the network was trained on siamese BGR inputs, the return value should probably be [2, 1, 0, 5, 4, 3]
        :return: list of channels for converting network inputs to RGB
        """
        # VGG was trained using BGR, I think..
        return [2, 1, 0]
