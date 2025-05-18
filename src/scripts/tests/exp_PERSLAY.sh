morphoclass --verbose train \
    --features-dir output/myown_features/ \
    --model-config src/configs/model-gnn.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/gnn_myfeatures_check/

morphoclass --verbose evaluate performance \
    output/gnn_myfeatures_check/checkpoint.chk \
    output/evaluation_report.html