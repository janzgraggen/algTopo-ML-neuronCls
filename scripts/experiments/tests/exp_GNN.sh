morphoclass --verbose extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    graph-rd\
    output/extract-features/pc-L5/apical/graph-rd/ --force

morphoclass --verbose train \
    --features-dir output/extract-features/pc-L5/apical/graph-rd/ \
    --model-config src/configs/model-gnn.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-graph-rd-gnn/

morphoclass --verbose evaluate performance \
    output/pc-L5-apical-graph-rd-gnn/checkpoint.chk \
    output/evaluation_report_gnn.html