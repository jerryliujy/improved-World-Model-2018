sudo apt-get install build-essential python3-dev
sudo apt-get install libgl1 libgl1-mesa-glx
pip install setuptools wheel
sudo apt-get install swig
pip install -r requirements.txt

# install autorom
pip install "autorom[accept-rom-license]"
AutoROM --accept-license