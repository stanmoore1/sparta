#!/bin/bash
# rungdb.sh <bin>: run in.pos under gdb on $BIN/<bin>_serial, print fmask[0], n_start_of_step, n_end_of_step at post_run
source /home/user/sparta/.fix-review/tools/env.sh
w=$SP/v7/r1/10-5/gdb.$1; rm -rf $w; mkdir -p $w; cd $w; cp $FR/decks/r1/common/* $FR/decks/r1/10-5/* .
timeout 120 gdb -batch -x gdb.cmd --args $BIN/$1_serial -in in.pos -echo none 2>&1 | grep '^\$'
