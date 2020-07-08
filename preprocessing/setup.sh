pip install -r requirements.py
python -m spacy download en_core_web_lg
cd ../
zenodo_get 3779954
mv models pretrained_models
