#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

if [ ! -d $DATA_ROOT ]; then
    wget https://cloud.irit.fr/s/dCPHpY2u6vDMQFV/download
    tar xf download -C $BASH_ROOT
    rm download
fi
