
from adapters.base_adapter import BaseAdapter


class TensorflowAdapter(BaseAdapter):

    def __init__(self, network_definition_script, checkpoint_filename, use_gpu):
        """
        Ctor of TensorflowAdapter class
        """

        self._network_definition_script = network_definition_script
        self._checkpoint_filename = checkpoint_filename
        self._use_gpu = use_gpu

        pass

    def load_network(self, settings):



        pass


    pass

