#!/bin/sh
cd hacktrick_ai
pip install -e .
cd ../hacktrick_rl
pip install -e .

cd ./hacktrick_rl
[ ! -f data_dir.py ] && echo "import os; DATA_DIR = os.path.abspath('.')" >> data_dir.py

pip install protobuf
pip install python-socketio[asyncio_client]==4.6.0
pip install python-engineio==3.13.0