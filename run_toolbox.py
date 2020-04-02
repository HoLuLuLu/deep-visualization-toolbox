#! /usr/bin/env python

# this import must comes first to make sure we use the non-display backend
import matplotlib

matplotlib.use('Agg')

import os, argparse
from live_vis import LiveVis
from bindings import bindings
from optimize_image import change_model_to_load, clean_temp_file


def main(model=None):
    try:
        if model:
            change_model_to_load(model)

        try:
            import settings
        except:
            print '\nError importing settings.py. Check the error message below for more information.'
            print "If you haven't already, you'll want to open the settings_model_selector.py file"
            print 'and edit it to point to your caffe checkout.\n'
            raise

        if not os.path.exists(settings.caffevis_caffe_root):
            raise Exception('ERROR: Set caffevis_caffe_root in settings.py first.')

        lv = LiveVis(settings)

        help_keys, _ = bindings.get_key_help('help_mode')
        quit_keys, _ = bindings.get_key_help('quit')
        print '\n\nRunning toolbox. Push %s for help or %s to quit.\n\n' % (help_keys[0], quit_keys[0])
        lv.run_loop()
    except Exception as exep:
        print str(exep)
    finally:
        clean_temp_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the Deep Visualization Toolbox.')
    parser.add_argument('--model', type=str, default=None,
                        help='Name of the model you want to change to. This overwrites the settings made in files.')
    args = parser.parse_args()

    main(args.model)
