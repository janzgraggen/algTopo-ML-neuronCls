## 🧵 Fixing Segmentation Faults from PyTorch ReLU on CPU

On macOS (and some Linux setups), PyTorch’s CPU builds can invoke OpenMP or MKL routines that are **not entirely thread-safe**. If too many threads spin up, this may cause a **segmentation fault** — especially in simple calls like `ReLU()`.

### 🛠️ What to try:

Force PyTorch to use **single-threaded execution** for OpenMP and MKL routines by setting the following environment variables:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

then you can savely run

```bash
morphoclass train \
  --features-dir output/extract-features/pc-L5/apical/graph-rd/ \
  --model-config CLI/configs/model-gnn.yaml \
  --splitter-config CLI/configs/splitter-stratified-k-fold.yaml \
  --checkpoint-dir output/pc-L5-apical-graph-rd-gnn/
```