#/bin/bash
cd data
wget -O data.zip "https://owncloud.tuwien.ac.at/index.php/s/oyh3CeHneDlAewn/download"
unzip data.zip

cd ../src/densefusion
docker build -t "densefusion" .

cd ../ppf
docker build -t "ppf_recognizer" .

cd ../maskrcnn
docker build -t "maskrcnn" .
