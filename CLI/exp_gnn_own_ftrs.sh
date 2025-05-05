morphoclass train \
    --features-dir output/myown_features/ \
    --model-config CLI/configs/model-gnn.yaml \
    --splitter-config CLI/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/gnn_myfeatures_check/

morphoclass evaluate performance \
    output/gnn_myfeatures_check/checkpoint.chk \
    output/evaluation_report.html