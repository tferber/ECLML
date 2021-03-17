#!/bin/bash

export PYTHONPATH=/nfs/dust/belle2/user/ferber/pip3/lib/python3.6/site-packages:$PYTHONPATH
export PATH=/nfs/dust/belle2/user/ferber/pip3/bin:$PATH

export CODE=$PWD
export PYTHONPATH=$CODE:$PYTHONPATH
#export PYTHONPATH=$CODE/models:$PYTHONPATH
#export PYTHONPATH=$CODE/losses:$PYTHONPATH
#export PYTHONPATH=$CODE/datasets:$PYTHONPATH
#export PYTHONPATH=$CODE/metrics:$PYTHONPATH

#export PATH=$CODE/models:$PATH
#export PATH=$CODE/losses:$PATH
#export PATH=$CODE/datasets:$PATH
#export PATH=$CODE/metrics:$PATH
