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


def calculate_bias_weights_histogram_for_specific_layer(net, layer_name, output_dir, fig, ax, do_print=True):
    if do_print:
        print "calculating weights histogram for layer %s" % (layer_name)

    # generate weights histogram for layer
    weights = net.params[layer_name][0].data.flatten()
    hist, bin_edges = np.histogram(weights, bins=50)
    weights_num = len(weights)

    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(center, hist, align='center', width=width, color='g')

    fig.suptitle('weights for layer %s\n %s weights used' % (layer_name, weights_num))
    ax.xaxis.label.set_text('weight value')
    ax.yaxis.label.set_text('count')

    # Save layer weights histogram to image file
    mkdir_p(output_dir)
    fig.savefig(os.path.join(output_dir, 'weight_histogram.png'))

    ax.cla()

    if do_print:
        print "calculating bias histogram for layer %s" % (layer_name)

    # generate bias histogram for layer
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
    fig.savefig(os.path.join(output_dir, 'bias_histogram.png'))

    ax.cla()


def calculate_weights_histogram_for_specific_unit(net, layer_name, channel_idx, output_dir, fig, ax,
                                                  hist_range=(None, None), do_print=True):
    n_channels = net.params[layer_name][0].shape[0]

    if do_print and channel_idx % 10 == 0:
        print "calculating weights histogram for layer %s channel %d out of %d" % (
            layer_name, channel_idx, n_channels)

    # get vector of weights
    weights = net.params[layer_name][0].data[channel_idx].flatten()
    bias = net.params[layer_name][1].data[channel_idx]
    weights_num = len(weights)

    # create histogram
    if hist_range is (None, None):
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
    fig.savefig(os.path.join(output_dir, 'weight_histogram.png'))

    ax.cla()


def calculate_weight_bias_histograms_for_net(net, settings, output_dir=None):
    if not output_dir:
        output_dir = settings.caffevis_outputs_dir

    layers = settings.layers_to_output_in_offline_scripts

    fig = plt.figure(figsize=(10, 10), facecolor='white', tight_layout=False)
    ax = fig.add_subplot(111)

    for layer_name in layers:
        layer_output_dir = os.path.join(output_dir, layer_name)
        mkdir_p(layer_output_dir)
        calculate_bias_weights_histogram_for_specific_layer(net, layer_name, layer_output_dir, fig, ax)

        for channel_idx in xrange(0, net.params[layer_name][0].shape[0]):
            unit_output_dir = os.path.join(layer_output_dir, 'unit_%04d' % channel_idx)
            mkdir_p(unit_output_dir)
            calculate_weights_histogram_for_specific_unit(net, layer_name, channel_idx, unit_output_dir, fig, ax)


def calculate_weights_histograms():
    try:
        # if model in command line change to it
        if '--model' in sys.argv:
            change_model_to_load(sys.argv[sys.argv.index('--model') + 1])

        import settings

        parser = argparse.ArgumentParser(
            description='Generates weights histograms for layers and its units.')
        parser.add_argument('--gpu', action='store_true', default=settings.caffevis_mode_gpu, help='use gpu')
        parser.add_argument('--net_prototxt', type=str, default=settings.caffevis_deploy_prototxt,
                            help='network prototxt to load')
        parser.add_argument('--net_weights', type=str, default=settings.caffevis_network_weights,
                            help='network weights to load')
        parser.add_argument('--outdir', type=str, default=settings.caffevis_outputs_dir,
                            help='Which output directory to use. Files are output into outdir/layer/unit_%%04d/{weights_histogram}.png')
        parser.add_argument('--model', type=str, default=None,
                            help='Name of the model you want to change to. This overwrites the settings made in files.')
        args = parser.parse_args()

        settings.caffevis_deploy_prototxt = args.net_prototxt
        settings.caffevis_network_weights = args.net_weights

        net, data_mean = load_network(settings)
        calculate_weight_bias_histograms_for_net(net, settings, args.outdir)

    except Exception as exep:
        print str(exep)
    finally:
        clean_temp_file()


if __name__ == "__main__":
    calculate_weights_histograms()
