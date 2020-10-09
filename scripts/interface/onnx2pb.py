#!/usr/bin/env python

import onnx
import sys

from onnx_tf.backend import prepare

onnx_model = onnx.load(sys.argv[1]+'.onnx')  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(sys.argv[1]+'.pb')  # export the model
