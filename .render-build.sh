#!/bin/bash
set -o errexit

# Install system dependencies for geospatial packages
apt-get update
apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt
