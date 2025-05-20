#!/bin/bash
set -e

mkdir -p data
cd data

#get dataset
wget https://zenodo.org/records/8260741/files/p8_ee_tt_ecm380_10files.tar?download=1 -O p8_ee_tt_ecm380_10files.tar 
tar xf p8_ee_tt_ecm380_10files.tar

#get model files
wget https://huggingface.co/jpata/particleflow/resolve/main/clic/clusters/v2.3.0/pyg-clic_20250130_214007_333962/model_kwargs.pkl
wget https://huggingface.co/jpata/particleflow/resolve/main/clic/clusters/v2.3.0/pyg-clic_20250130_214007_333962/checkpoints/checkpoint-10-1.932789.pth

cd ..

#get MLPF model code
git clone --depth 1 --branch v2.3.0 https://github.com/jpata/particleflow.git

#process dataset to ML format
python3 particleflow/mlpf/data/key4hep/postprocessing.py --input data/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_1302.root --dataset clic --outpath data/p8_ee_tt_ecm380/
