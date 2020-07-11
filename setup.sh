conda create --name aqa python=3.7
conda activate aqa
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
python install -r requirements.txt
python -m spacy download en_core_web_lg
