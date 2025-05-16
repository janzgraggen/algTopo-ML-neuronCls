morphoclass --verbose extract-features \
    assets/datasets_structured_layer/pyramidal-cells/L5/dataset.csv \
    apical \
    image-tmd-rd \
    output/extract-features/pc-L5/apical/image-tmd-rd/ --force

morphoclass --verbose train \
    --features-dir output/extract-features/pc-L5/apical/image-tmd-rd/ \
    --model-config src/configs/model-cnn.yaml \
    --splitter-config src/configs/splitter-stratified-k-fold.yaml \
    --checkpoint-dir output/pc-L5-apical-image-tmd-rd-cnn/

morphoclass --verbose evaluate performance \
    output/pc-L5-apical-image-tmd-rd-cnn/checkpoint.chk \
    output/evaluation_report2.html