#!/bin/bash

set -e
SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WDIR=`pwd`

# Extract data
cd $WDIR/data

# Get the data
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
rm -rf conala-corpus
unzip conala-corpus-v1.1.zip
rm -rf conala-corpus-v1.1.zip

wget https://www.dropbox.com/s/7l42q8foywuqu5y/parsed_so.zip?dl=1
rm -rf parsed_so.json
mv -f "parsed_so.zip?dl=1" parsed_so.zip
unzip parsed_so.zip

cd $WDIR
$SHELL
