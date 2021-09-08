#!/bin/bash
WORKROOT=$(cd $(dirname $0); pwd)
cd $WORKROOT
#### python ####
export PYTHONPATH=$WORKROOT:$PYTHONPATH
#echo "PYTHONPATH=$PYTHONPATH"
## python 3.6/3.7 is recomended
PYTHON_BIN=`which python3`
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "running command: ($PYTHON_BIN $@)"
$PYTHON_BIN -u $@
exit $?
