
class BaseAdapter(object):

    def __init__(self):
        pass

    def init_thread_specific(self):
        pass

    def load_network(self, settings):
        pass

    def get_layers_list(self):
        pass

    def get_blob_info(self, layer_name):
        pass
