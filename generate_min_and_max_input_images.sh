#!/usr/bin/env bash

python2 ./find_maxes/find_max_acts.py --search-min --N=10 "$@"
python2 ./find_maxes/crop_max_patches.py --search-min --N=10 "$@"
