#!/usr/bin/env bash

python2 ./find_maxes/find_max_acts.py "$@"
python2 ./find_maxes/crop_max_patches.py "$@"
