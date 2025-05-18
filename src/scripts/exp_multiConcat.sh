
## Feature Extraction
python run_MultiFeatureExtract.py \
    src/configs/extract-multi.yaml

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=".:$PYTHONPATH"
morphoclass --verbose train \
    --features-dir output/multiconcat_features/ \
    --model-config src/configs/model-multiconcat.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/multiconcat_checkpoint/

morphoclass --verbose evaluate performance \
    output/multiconcat_checkpoint/checkpoint.chk \
    output/eval_multiconcat.html