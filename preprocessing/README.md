# Preprocessing Scripts

## Order

1. terms.py
2. coref.py
3. generate_oie6_input.py
4. [cd ~/oie6/] run.py --save models/results --mode splitpredict --inp ~/aqa/data/hotpot_qa/open_ie_data/sentences_test.txt --task oie --gpus 1 --out test_predictions.txt --rescoring --num_extractions 5
5. process_oie6_output.py
6. kg.py