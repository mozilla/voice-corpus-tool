#!/bin/bash
base_dir=`dirname $0`/..
rm -rf "${base_dir}/venv"
virtualenv -p python3 "${base_dir}/venv"
source "${base_dir}/venv/bin/activate"
pip3 install -r "${base_dir}/requirements.txt"