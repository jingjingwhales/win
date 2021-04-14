#!/usr/bin/env bash

echo "start to set up virual env"

if [ -d venv ]; then
    echo "virtual env is already there"
else
    pip3 install virtualenv
    virtualenv -p $(which python3) venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    echo "virtual env is set up now, you can 'source venv/bin/activate'"
fi


