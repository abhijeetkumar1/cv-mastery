# CV Mastery

A hands-on computer vision course in notebooks: from pixels to CNNs, transformers, VLMs, GPU inference and document AI.
The full plan is in [PLAN.md](PLAN.md).

## Quick start

```bash
uv sync               # install the Python 3.12 environment (first time, or after dependency changes)
uv run jupyter lab    # open the notebooks
```

In VS Code or Cursor, pick the kernel **Python 3.12 (cv-mastery)**. Start with `notebooks/00_setup/00_setup.ipynb`.

Sample images, videos and datasets download automatically on first use and are cached in `data/`.

## Layout

| Folder | What's in it |
|---|---|
| `notebooks/` | one folder per module |
| `cvlib/` | reusable code built during the course (`env`, `data`, `viz`, `train`, ...) |
| `data/` | downloads and outputs (not committed) |
| `notes/` | your own cheat sheets |

## Progress

Tick each box when you've worked through the notebook (and done its experiments).

### Phase 1 · Foundations
- [ ] 00 · Setup: `00_setup/00_setup.ipynb`
- [ ] 01a · Image basics: `01_classic_cv/01a_image_basics.ipynb`
- [ ] 01b · Filters & shapes: `01_classic_cv/01b_filters_and_shapes.ipynb`
- [ ] 01c · Features & geometry: `01_classic_cv/01c_features_and_geometry.ipynb`
- [ ] 01d · Video & motion: `01_classic_cv/01d_video_and_motion.ipynb` (+ `webcam_motion.py`)
- [ ] 02a · Neural network from scratch: `02_nn_basics/02a_numpy_network.ipynb`
- [ ] 02b · PyTorch basics & the training loop: `02_nn_basics/02b_pytorch_basics.ipynb`

### Phase 2 · CNNs
- [ ] 03 · CNNs in depth: `03_cnn_depth/03_cnns_in_depth.ipynb`
- [ ] 04 · CNN architectures: `04_cnn_architectures/04_cnn_architectures.ipynb` (benchmark: also run on Kaggle for GPU numbers)
- [ ] 05a · Transfer learning: `05_transfer_learning/05a_transfer_learning.ipynb` (**run on Kaggle**; smoke-tests locally)
- [ ] 05b · Training craft: `05_transfer_learning/05b_training_craft.ipynb`
- [ ] 05c · Multi-label classification: `05_transfer_learning/05c_multilabel.ipynb` (fine-tuning part: Kaggle)

### Phase 3 · Detection & segmentation
- [ ] 06 · Object detection
- [ ] 07 · Segmentation

### Phase 4 · Transformers & embeddings
- [ ] 08 · Attention & Vision Transformers
- [ ] 09 · DETR family
- [ ] 10 · Self-supervised learning & CLIP
- [ ] 11 · Embeddings & retrieval
- [ ] 12 · Face recognition

### Phase 5 · Vision-language models
- [ ] 13 · VLMs: open source & frontier

### Phase 6 · Evaluate & optimize
- [ ] 14 · Model evaluation
- [ ] 15 · GPU inference optimization

### Phase 7 · Document AI
- [ ] 16 · OCR
- [ ] 17 · Document inputs & layout
- [ ] 18 · Document understanding (LayoutLMv3)
- [ ] 19 · Tables & forms
- [ ] 20 · OCR-free & VLM document AI
- [ ] 21 · Document AI in production

### Phase 8 · Capstone
- [ ] 22 · Optimized document intelligence platform
