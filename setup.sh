#!/bin/bash

ZIP_FILE=download_folder.zip 

if [ -f "$ZIP_FILE" ]; then
    echo "Starting the unziping $ZIP_FILE"
    
    tar -xvzf $ZIP_FILE
    
else 
    echo "[ERROR]: Please download the $ZIP_FILE and rerun this script"
    echo "$ZIP_FILE is available here: https://uapt33090-my.sharepoint.com/:u:/g/personal/aleixomatos_ua_pt/EY81uznCss1FuR8l-58ruq0BfiqVBJT8GCHa1LOlt8bqCw?e=rpR5Jl"

    exit 1
fi

echo "Install python the requirements"
python install requirements.txt

echo "Manually Install mmnrm python library"
cd _other_dependencies/mmnrm/
rm -r ./dist
python setup.py sdist
pip install ./dist/mmnrm-0.0.2.tar.gz

echo "Manually Install nir python library"
cd ../nir/
rm -r ./dist
python setup.py sdist
pip install ./dist/nir-0.0.1.tar.gz

cd ../../