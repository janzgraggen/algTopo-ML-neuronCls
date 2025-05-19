
## Feature Extraction
python run_MultiFeatureExtract.py \
    src/configs/extract-multi.yaml

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=".:$PYTHONPATH"
morphoclass --verbose train \
    --features-dir output/features/features:kanari18_laylab \
    --model-config src/configs/model-multiconcat.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/results/features:kanari18laylab/ 

morphoclass --verbose evaluate performance \
    output/results/features:multiconcat/checkpoint.chk \
    output/results/features:multiconcat/eval_multiconcat.html