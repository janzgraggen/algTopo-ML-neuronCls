
## Feature Extraction
python run_graph.py

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=".:$PYTHONPATH"
morphoclass --verbose train \
    --features-dir output/multiconcat_features/ \
    --model-config CLI/configs/model-multiconcat.yaml \
    --splitter-config CLI/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/multiconcat_checkpoint/

morphoclass --verbose evaluate performance \
    output/multiconcat_checkpoint/checkpoint.chk \
    output/eval_multiconcat.html