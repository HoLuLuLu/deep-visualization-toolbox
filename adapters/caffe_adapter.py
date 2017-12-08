import sys
import os
import numpy as np

from adapters.base_adapter import BaseAdapter
from image_misc import get_tiles_height_width_ratio
from settings_misc import replace_magic_DVT_ROOT

class CaffeAdapter(BaseAdapter):

    def __init__(self,
                 deploy_prototxt_filepath,
                 network_weights_filepath,
                 data_mean_ref=None,
                 generate_channelwise_mean=False,
                 caffe_root=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../caffe'),
                 use_gpu=True,
                 gpu_id=0,
                 image_dims=None,
                 raw_scale=255.0,
                 input_scale=None,
                 channel_swap=None,
                 transpose=None):
        """
        Ctor of CaffeAdapter class
        :param deploy_prototxt_filepath: Path to caffe deploy prototxt file
        :param network_weights_filepath: Path to network weights to load.
        :param data_mean_ref: Reference to data mean, if any, to be subtracted from input image file / webcam image.
        :param generate_channelwise_mean: should we generate the channelwise average of the input mean file
        :param caffe_root: caffe root directory
        :param use_gpu: whether to use GPU mode (if True) or CPU mode (if False)
        :param gpu_id: ID of GPU to use
        :param image_dims: image dimensions
        :param raw_scale: raw scale, multiplies input BEFORE mean subtraction
        :param input_scale: input scale, multiplies input AFTER mean subtraction
        :param channel_swap: channel swap, default is None which will make automatic decision according to other
               settings the automatic setting is either (2,1,0) or (2,1,0,5,4,3) according to is_siamese value and
               siamese_input_mode
        :param transpose: transpose, used to convert HxWxK to KxHxW, when None uses caffe default which is (2,0,1)
               this parameter should rarely change
        Specify as string path to file or tuple of one value per channel or None.
        """

        # Check that caffe root directory actually exists
        if not os.path.exists(caffe_root):
            raise Exception('The Caffe directory specified in caffe_root parameter, %s, does not exist' % caffe_root)

        self._caffe_root = caffe_root
        sys.path.insert(0, os.path.join(self._caffe_root, 'python'))

        self._deploy_prototxt_filepath = replace_magic_DVT_ROOT(deploy_prototxt_filepath)
        self._network_weights_filepath = replace_magic_DVT_ROOT(network_weights_filepath)
        self._data_mean_ref = data_mean_ref
        if isinstance(self._data_mean_ref, basestring):
            self._data_mean_ref = replace_magic_DVT_ROOT(self._data_mean_ref)
        self._generate_channelwise_mean = generate_channelwise_mean

        self._use_gpu = use_gpu
        self._gpu_id = gpu_id
        self._image_dims = image_dims
        self._raw_scale = raw_scale
        self._input_scale = input_scale
        self._channel_swap = channel_swap
        self._transpose = transpose

        pass

    pass

    def init_thread_specific(self):

        import caffe

        # Set the mode to CPU or GPU. Note: in the latest Caffe
        # versions, there is one Caffe object *per thread*, so the
        # mode must be set per thread! Here we set the mode for the
        # CaffeProcThread thread; it is also set in the main thread.
        if self._use_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(self._gpu_id)
            print 'Loaded caffe in GPU mode, using device', self._gpu_id
        else:
            caffe.set_mode_cpu()
            print 'Loaded caffe in CPU mode'

        return

    def load_network(self, settings):

        import caffe

        self._settings = settings

        self.init_thread_specific()

        self._process_network_proto(settings)

        self._deduce_calculated_settings_without_network(settings, self._processed_deploy_prototxt_filepath)

        net = caffe.Classifier(
            self._processed_deploy_prototxt_filepath,
            self._network_weights_filepath,
            image_dims=self._image_dims,
            mean=None,  # assigned later
            raw_scale=self._raw_scale,
            input_scale=self._input_scale,
            channel_swap=self._calculated_channel_swap)

        self._deduce_calculated_settings_with_network(settings, net)

        if self._transpose:
            net.transformer.set_transpose(net.inputs[0], self._transpose)

        data_mean = CaffeAdapter._set_mean(self._data_mean_ref, self._generate_channelwise_mean, net)

        # keep net and data_mean to be used later
        self._net = net
        self._data_mean = data_mean

        # set network batch size to 1
        current_input_shape = self._net.blobs[self._net.inputs[0]].shape
        current_input_shape[0] = 1
        self._net.blobs[self._net.inputs[0]].reshape(*current_input_shape)
        self._net.reshape()

        # calc blob info
        self._calc_blob_info()

        return

    def get_data_mean(self):
        return self._data_mean

    def get_layers_list(self):

        layers_list = []

        for layer_name in list(self._net._layer_names):

            # skip inplace layers
            if len(self._net.top_names[layer_name]) == 1 and len(self._net.bottom_names[layer_name]) == 1 and \
                            self._net.top_names[layer_name][0] == self._net.bottom_names[layer_name][0]:
                continue

            layers_list.append(layer_name)

        return layers_list

    def _calc_blob_info(self):
        '''For each blob, save the number of filters and precompute
        tile arrangement (needed by CaffeVisAppState to handle keyboard navigation).
        '''
        self._net_blob_info = {}
        for key in self._net.blobs.keys():
            self._net_blob_info[key] = {}
            # Conv example: (1, 96, 55, 55)
            # FC example: (1, 1000)
            blob_shape = self._net.blobs[key].data.shape

            # handle case when output is a single number per image in the batch
            if (len(blob_shape) == 1):
                blob_shape = (blob_shape[0], 1)

            self._net_blob_info[key]['isconv'] = (len(blob_shape) == 4)
            self._net_blob_info[key]['data_shape'] = blob_shape[1:]  # Chop off batch size
            self._net_blob_info[key]['n_tiles'] = blob_shape[1]
            self._net_blob_info[key]['tiles_rc'] = get_tiles_height_width_ratio(blob_shape[1], self._settings.caffevis_layers_aspect_ratio)
            self._net_blob_info[key]['tile_rows'] = self._net_blob_info[key]['tiles_rc'][0]
            self._net_blob_info[key]['tile_cols'] = self._net_blob_info[key]['tiles_rc'][1]

        return

    def get_blob_info(self, layer_name):

        top_name = self._layer_name_to_top_name(layer_name)
        return self._net_blob_info[top_name]

    def _layer_name_to_top_name(self, layer_name):

        if self._net.top_names.has_key(layer_name) and len(self._net.top_names[layer_name]) >= 1:
            return self._net.top_names[layer_name][0]

        else:
            return None

    def get_layer_weights(self, layer_name):
        if self._net.params.has_key(layer_name):
            return self._net.params[layer_name][0].data
        else:
            return None

    def get_layer_bias(self, layer_name):
        if self._net.params.has_key(layer_name):
            return self._net.params[layer_name][1].data
        else:
            return None

    def get_layer_data(self, layer_name):
        top_name = self._layer_name_to_top_name(layer_name)
        if top_name in self._net.blobs:
            return self._net.blobs[top_name].data
        else:
            return None

    def get_layer_diff(self, layer_name):
        top_name = self._layer_name_to_top_name(layer_name)
        if top_name in self._net.blobs:
            return self._net.blobs[top_name].diff
        else:
            return None

    def forward(self, img):
        data_blob = self._net.transformer.preprocess('data', img)  # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis, :, :, :]  # e.g. (1, 3, 227, 227)
        output = self._net.forward(data=data_blob)
        return

    def forward_all(self, data):
        return self._net.forward_all(data=data)

    def predict(self, im_batch, oversample=False):
        return self._net.predict(im_batch, oversample=oversample)

    def backward_from_layer(self, start_layer_name, start_diff, diffs=None, zero_higher=False):
        """
        Backward pass starting from somewhere in the middle of the
        network, starting with the provided diffs.

        :param start_layer_name: layer at which to begin the backward pass
        :param start_diff: diff to set at start_name layer
        :param diffs: list of diffs to return in addition to bottom diffs
        :param zero_higher: whether or not to zero out higher layers to reflect the true 0 derivative or leave them alone to save time
        :return: n/a
        """

        self._net.backward_from_layer(start_layer_name, start_diff, diffs, zero_higher)
        return

    def deconv_from_layer(self, start_layer_name, start_diff, diffs=None, zero_higher=False, deconv_type='Zeiler & Fergus'):
        """
        Deconv pass starting from somewhere in the middle of the
        network, starting with the provided diffs.

        :param start_layer_name: layer at which to begin the deconv pass
        :param start_diff: diff to set at start_name layer
        :param diffs: list of diffs to return in addition to bottom diffs
        :param zero_higher: whether or not to zero out higher layers to reflect the true 0 derivative or leave them alone to save time
        :param deconv_type: either 'Zeiler & Fergus' or 'Guided Backprop'
        :return: n/a
        """

        self._net.deconv_from_layer(start_layer_name, start_diff, diffs, zero_higher, deconv_type)
        return

    def set_network_batch_size(self, batch_size):
        # set network batch size
        current_input_shape = self._net.blobs[self._net.inputs[0]].shape
        current_input_shape[0] = batch_size
        self._net.blobs[self._net.inputs[0]].reshape(*current_input_shape)
        self._net.reshape()

    def _process_network_proto(self, settings):

        self._processed_deploy_prototxt_filepath = self._deploy_prototxt_filepath + ".processed_by_deepvis"

        # check if force_backwards is missing
        found_force_backwards = False
        with open(self._deploy_prototxt_filepath, 'r') as proto_file:
            for line in proto_file:
                fields = line.strip().split()
                if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                    found_force_backwards = True
                    break

        # write file, adding force_backward if needed
        with open(self._deploy_prototxt_filepath, 'r') as proto_file:
            with open(self._processed_deploy_prototxt_filepath, 'w') as new_proto_file:
                if not found_force_backwards:
                    new_proto_file.write('force_backward: true\n')
                for line in proto_file:
                    new_proto_file.write(line)

        # run upgrade tool on new file name (same output file)
        upgrade_tool_command_line = self._caffe_root + '/build/tools/upgrade_net_proto_text.bin ' + \
                                    self._processed_deploy_prototxt_filepath + ' ' + \
                                    self._processed_deploy_prototxt_filepath
        os.system(upgrade_tool_command_line)

        return

    def _deduce_calculated_settings_without_network(self, settings, processed_deploy_prototxt_filepath):
        CaffeAdapter._set_calculated_siamese_network_format(settings)
        self._set_calculated_channel_swap(settings)
        CaffeAdapter._read_network_dag(settings, processed_deploy_prototxt_filepath)

    def _deduce_calculated_settings_with_network(self, settings, net):
        CaffeAdapter._set_calculated_is_gray_model(settings, net)
        self._set_calculated_image_dims(net)

    @staticmethod
    def _set_calculated_siamese_network_format(settings):

        settings._calculated_siamese_network_format = 'normal'

        for layer_def in settings.layers_list:
            if layer_def['format'] != 'normal':
                settings._calculated_siamese_network_format = layer_def['format']
                return

    def _set_calculated_channel_swap(self, settings):

        if self._channel_swap is not None:
            self._calculated_channel_swap = self._channel_swap

        else:
            if settings.is_siamese and settings.siamese_input_mode == 'concat_channelwise':
                self._calculated_channel_swap = (2, 1, 0, 5, 4, 3)

            else:
                self._calculated_channel_swap = (2, 1, 0)

    class LayerRecord:

        def __init__(self, layer_def):

            self.layer_def = layer_def
            self.name = layer_def.name
            self.type = layer_def.type

            # keep filter, stride and pad
            if layer_def.type == 'Convolution':
                self.filter = map(int, list(layer_def.convolution_param.kernel_size))
                if len(self.filter) == 1:
                    self.filter *= 2
                self.pad = map(int, list(layer_def.convolution_param.pad))
                if len(self.pad) == 0:
                    self.pad = [0, 0]
                elif len(self.pad) == 1:
                    self.pad *= 2
                self.stride = map(int, list(layer_def.convolution_param.stride))
                if len(self.stride) == 0:
                    self.stride = [1, 1]
                elif len(self.stride) == 1:
                    self.stride *= 2

            elif layer_def.type == 'Pooling':
                self.filter = map(int, [layer_def.pooling_param.kernel_size])
                if len(self.filter) == 1:
                    self.filter *= 2
                self.pad = map(int, [layer_def.pooling_param.pad])
                if len(self.pad) == 0:
                    self.pad = [0, 0]
                elif len(self.pad) == 1:
                    self.pad *= 2
                self.stride = map(int, [layer_def.pooling_param.stride])
                if len(self.stride) == 0:
                    self.stride = [1, 1]
                elif len(self.stride) == 1:
                    self.stride *= 2

            else:
                self.filter = [0, 0]
                self.pad = [0, 0]
                self.stride = [1, 1]

            # keep tops
            self.tops = list(layer_def.top)

            # keep bottoms
            self.bottoms = list(layer_def.bottom)

            # list of parent layers
            self.parents = []

            # list of child layers
            self.children = []

        pass


    @staticmethod
    def _read_network_dag(settings, processed_deploy_prototxt):
        from caffe.proto import caffe_pb2
        from google.protobuf import text_format

        # load prototxt file
        network_def = caffe_pb2.NetParameter()
        with open(processed_deploy_prototxt, 'r') as proto_file:
            text_format.Merge(str(proto_file.read()), network_def)

        # map layer name to layer record
        layer_name_to_record = dict()
        for layer_def in network_def.layer:
            if (len(layer_def.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer_def.include]):
                layer_name_to_record[layer_def.name] = CaffeAdapter.LayerRecord(layer_def)

        top_to_layers = dict()
        for layer in network_def.layer:
            # no specific phase, or TEST phase is specifically asked for
            if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
                for top in layer.top:
                    if top not in top_to_layers:
                        top_to_layers[top] = list()
                    top_to_layers[top].append(layer.name)

        # find parents and children of all layers
        for child_layer_name in layer_name_to_record.keys():
            child_layer_def = layer_name_to_record[child_layer_name]
            for bottom in child_layer_def.bottoms:
                for parent_layer_name in top_to_layers[bottom]:
                    if parent_layer_name in layer_name_to_record:
                        parent_layer_def = layer_name_to_record[parent_layer_name]
                        if parent_layer_def not in child_layer_def.parents:
                            child_layer_def.parents.append(parent_layer_def)
                        if child_layer_def not in parent_layer_def.children:
                            parent_layer_def.children.append(child_layer_def)

        # update filter, strid, pad for maxout "structures"
        for layer_name in layer_name_to_record.keys():
            layer_def = layer_name_to_record[layer_name]
            if layer_def.type == 'Eltwise' and \
               len(layer_def.parents) == 1 and \
               layer_def.parents[0].type == 'Slice' and \
               len(layer_def.parents[0].parents) == 1 and \
               layer_def.parents[0].parents[0].type in ['Convolution', 'InnerProduct']:
                layer_def.filter = layer_def.parents[0].parents[0].filter
                layer_def.stride = layer_def.parents[0].parents[0].stride
                layer_def.pad = layer_def.parents[0].parents[0].pad

        # keep helper variables in settings
        settings._network_def = network_def
        settings._layer_name_to_record = layer_name_to_record

        return

    @staticmethod
    def _set_calculated_is_gray_model(settings, net):
        if settings.is_gray_model is not None:
            settings._calculated_is_gray_model = settings.is_gray_model
        else:
            input_shape = net.blobs[net.inputs[0]].data.shape
            channels = input_shape[1]
            if channels == 1:
                settings._calculated_is_gray_model = True
            elif channels == 2 and settings.is_siamese:
                settings._calculated_is_gray_model = True
            elif channels == 3:
                settings._calculated_is_gray_model = False
            elif channels == 6 and settings.is_siamese:
                settings._calculated_is_gray_model = False
            else:
                settings._calculated_is_gray_model = None

    def _set_calculated_image_dims(self, net):
        '''
        set calculate image dimensions
        :param settings:
        :param net: loaded caffe network
        '''

        # if image dimensions were externally set, use them
        if self._image_dims is not None:
            self._calculated_image_dims = self._image_dims

        else:
            # otherwise, use network assigned shape
            input_shape = net.blobs[net.inputs[0]].data.shape
            self._calculated_image_dims = input_shape[2:4]

    @staticmethod
    def _set_mean(data_mean_ref, generate_channelwise_mean, net):

        if isinstance(data_mean_ref, basestring):
            # If the mean is given as a filename, load the file
            try:
                data_mean = CaffeAdapter._load_mean_file(data_mean_ref)
            except IOError:
                print '\n\nCound not load mean file:', data_mean_ref
                print 'Ensure that the values in settings.py point to a valid model weights file, network'
                print 'definition prototxt, and mean. To fetch a default model and mean file, use:\n'
                print '$ cd models/caffenet-yos/'
                print '$ ./fetch.sh\n\n'
                raise
            input_shape = net.blobs[net.inputs[0]].data.shape[-2:]  # e.g. 227x227
            # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
            excess_h = data_mean.shape[1] - input_shape[0]
            excess_w = data_mean.shape[2] - input_shape[1]
            assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(input_shape)
            data_mean = data_mean[:, (excess_h / 2):(excess_h / 2 + input_shape[0]),
                              (excess_w / 2):(excess_w / 2 + input_shape[1])]
        elif data_mean_ref is None:
            data_mean = None
        else:
            # The mean has been given as a value or a tuple of values
            data_mean = np.array(data_mean_ref)
            # Promote to shape C,1,1
            # while len(data_mean.shape) < 3:
            #     data_mean = np.expand_dims(data_mean, -1)

        if generate_channelwise_mean:
            data_mean = data_mean.mean(1).mean(1)

        if data_mean is not None:
            print 'Using mean with shape:', data_mean.shape
            net.transformer.set_mean(net.inputs[0], data_mean)

        return data_mean

    @staticmethod
    def _load_mean_file(data_mean_file):
        filename, file_extension = os.path.splitext(data_mean_file)
        if file_extension == ".npy":
            # load mean from numpy array
            data_mean = np.load(data_mean_file)
            print "Loaded mean from numpy file, data_mean.shape: ", data_mean.shape

        elif file_extension == ".binaryproto":

            # load mean from binary protobuf file
            import caffe
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open(data_mean_file, 'rb').read()
            blob.ParseFromString(data)
            data_mean = np.array(caffe.io.blobproto_to_array(blob))
            data_mean = np.squeeze(data_mean)
            print "Loaded mean from binaryproto file, data_mean.shape: ", data_mean.shape

        else:
            # unknown file extension, trying to load as numpy array
            data_mean = np.load(data_mean_file)
            print "Loaded mean from numpy file, data_mean.shape: ", data_mean.shape

        return data_mean

    @staticmethod
    def _shownet(net):
        '''Print some stats about a net and its activations'''

        print '%-41s%-31s%s' % ('', 'acts', 'act diffs')
        print '%-45s%-31s%s' % ('', 'params', 'param diffs')
        for k, v in net.blobs.items():
            if k in net.params:
                params = net.params[k]
                for pp, blob in enumerate(params):
                    if pp == 0:
                        print '  ', 'P: %-5s' % k,
                    else:
                        print ' ' * 11,
                    print '%-32s' % repr(blob.data.shape),
                    print '%-30s' % ('(%g, %g)' % (blob.data.min(), blob.data.max())),
                    print '(%g, %g)' % (blob.diff.min(), blob.diff.max())
            print '%-5s' % k, '%-34s' % repr(v.data.shape),
            print '%-30s' % ('(%g, %g)' % (v.data.min(), v.data.max())),
            print '(%g, %g)' % (v.diff.min(), v.diff.max())