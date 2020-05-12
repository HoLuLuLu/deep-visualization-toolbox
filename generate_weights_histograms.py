#! /usr/bin/env python

# this import must comes first to make sure we use the non-display backend
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys
import os

from misc import mkdir_p
from optimize_image import change_model_to_load, clean_temp_file
from settings_misc import load_network

all_weights_per_layer = {}


def calculate_weights_histogram_for_specific_layer(net, layer_name, output_dir, fig, ax, force=False):
    # generate weights histogram for layer
    hist_filename = os.path.join(output_dir, 'weight_histogram.png')

    weights = net.params[layer_name][0].data.flatten()
    hist, bin_edges = np.histogram(weights, bins=50)
    if not force and os.path.isfile(hist_filename):
        print 'The weights histogram for this layer already exists. Skipping...'
    else:
        weights_num = len(weights)

        width = 0.7 * (bin_edges[1] - bin_edges[0])
        center = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(center, hist, align='center', width=width, color='g')

        fig.suptitle('weights for layer %s\n %s weights used' % (layer_name, weights_num))
        ax.xaxis.label.set_text('weight value')
        ax.yaxis.label.set_text('count')

        # Save layer weights histogram to image file
        mkdir_p(output_dir)
        fig.savefig(hist_filename)

        ax.cla()

    return hist, bin_edges


def calculate_bias_histogram_for_specific_layer(net, layer_name, output_dir, fig, ax, force=False):
    # generate bias histogram for layer
    bias_filename = os.path.join(output_dir, 'bias_histogram.png')
    if not force and os.path.isfile(bias_filename):
        print 'The bias histogram for this layer already exists. Skipping...'
    else:
        bias = net.params[layer_name][1].data.flatten()
        hist, bin_edges = np.histogram(bias, bins=50)
        bias_num = len(bias)

        width = 0.7 * (bin_edges[1] - bin_edges[0])
        center = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(center, hist, align='center', width=width, color='g')

        fig.suptitle('bias for layer %s\n %s biases used' % (layer_name, bias_num))
        ax.xaxis.label.set_text('bias value')
        ax.yaxis.label.set_text('count')

        # Save layer bias histogram to image file
        fig.savefig(bias_filename)

        ax.cla()

        return hist, bin_edges


def calculate_weights_histogram_for_specific_unit(net, layer_name, channel_idx, output_dir, fig, ax,
                                                  hist_range=(None, None), force=False):
    hist_filename = os.path.join(output_dir, 'weight_histogram.png')
    if not force and os.path.isfile(hist_filename):
        print 'The weights histogram for unit %d already exists. Skipping...' % (channel_idx)
    else:
        # get vector of weights
        weights = net.params[layer_name][0].data[channel_idx].flatten()
        bias = net.params[layer_name][1].data[channel_idx]
        weights_num = len(weights)

        # create histogram
        if hist_range == (None, None):
            hist_range = (weights.min(), weights.max())
        hist, bin_edges = np.histogram(weights, bins=50, range=hist_range)

        # generate histogram image file
        width = 0.7 * (bin_edges[1] - bin_edges[0])
        center = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(center, hist, align='center', width=width, color='g')

        fig.suptitle('weights for unit %d, bias is %f\n %s weights used' % (channel_idx, bias, weights_num))
        ax.xaxis.label.set_text('weight value')
        ax.yaxis.label.set_text('count')

        # save weight histogram as image file
        mkdir_p(output_dir)
        fig.savefig(hist_filename)

        ax.cla()

        return hist, bin_edges


def calculate_weight_bias_histograms_for_net(net, settings, output_dir=None, scale_by_layer=False, do_print=True,
                                             force=False):
    if not output_dir:
        output_dir = settings.caffevis_outputs_dir

    layers = settings.layers_to_output_in_offline_scripts

    fig = plt.figure(figsize=(10, 10), facecolor='white', tight_layout=False)
    ax = fig.add_subplot(111)

    for layer_name in layers:
        if do_print:
            print "calculating weights and bias histogram for layer %s" % (layer_name)

        layer_output_dir = os.path.join(output_dir, layer_name)
        mkdir_p(layer_output_dir)
        try:
            all_weights_per_layer[layer_name] = np.stack(net.params[layer_name][0].data, net.params[layer_name][1].data)
            layer_hist, layer_bin_edges = calculate_weights_histogram_for_specific_layer(net, layer_name,
                                                                                         layer_output_dir, fig, ax,
                                                                                         force)
            calculate_bias_histogram_for_specific_layer(net, layer_name, layer_output_dir, fig, ax, force)

            for channel_idx in xrange(0, net.params[layer_name][0].shape[0]):
                n_channels = net.params[layer_name][0].shape[0]

                if do_print and channel_idx % 10 == 0:
                    print "calculating weights histogram for layer %s channel %d out of %d" % (
                        layer_name, channel_idx, n_channels)

                unit_output_dir = os.path.join(layer_output_dir, 'unit_%04d' % channel_idx)
                mkdir_p(unit_output_dir)
                if scale_by_layer and not layer_bin_edges is None:
                    layer_hist_range = (layer_bin_edges.min(), layer_bin_edges.max())
                else:
                    layer_hist_range = (None, None)
                calculate_weights_histogram_for_specific_unit(net, layer_name, channel_idx, unit_output_dir, fig, ax,
                                                              layer_hist_range, force)
        except KeyError as excep:
            print 'Unable to generate weights/bias histograms for this layer: ' + str(excep)

    if len(all_weights_per_layer.keys()) > 0:
        np.save(os.path.join(output_dir, 'all_weights_biases.npy'), all_weights_per_layer)


def calculate_weights_histograms():
    try:
        # if model in command line change to it
        if '--model' in sys.argv:
            change_model_to_load(sys.argv[sys.argv.index('--model') + 1])

        import settings

        parser = argparse.ArgumentParser(
            description='Generates weights histograms for layers and its units.')
        parser.add_argument('--force', action='store_true', default=False,
                            help='Activate force mode. Overwrites existing files')
        parser.add_argument('--scale_by_layer', action='store_true', default=False,
                            help='Every unit of a layer has the same scale for the histogram.')
        parser.add_argument('--outdir', type=str, default=settings.caffevis_outputs_dir,
                            help='Which output directory to use. Files are output into outdir/layer/unit_%%04d/{weights_histogram}.png')
        parser.add_argument('--model', type=str, default=None,
                            help='Name of the model you want to change to. This overwrites the settings made in files.')
        args = parser.parse_args()

        net, data_mean = load_network(settings)
        calculate_weight_bias_histograms_for_net(net, settings, args.outdir, args.scale_by_layer, force=args.force)

    finally:
        clean_temp_file()


if __name__ == "__main__":
    calculate_weights_histograms()
