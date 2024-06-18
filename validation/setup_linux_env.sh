#!/bin/bash

echo "Installing miniconda3 and pm2 ONLY!"

# Stop the script on any error
set -e

apt update
apt install nano
apt install vim npm -y
npm install pm2 -g
npm fund

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
bash /opt/miniconda-installer.sh -b -p /opt/miniconda3

./../../../opt/miniconda3/bin/conda init

echo "Done."