#!/bin/bash

python onnx2pb.py $1
python add_preprocess.py $1 $2
sh show.sh $1
