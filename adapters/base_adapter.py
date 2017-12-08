
class BaseAdapter(object):

    def __init__(self):
        pass

    def init_thread_specific(self):
        pass

    def load_network(self, settings):
        pass

    def get_data_mean(self):
        pass

    def get_layers_list(self):
        pass

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
