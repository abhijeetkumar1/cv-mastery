# CV Mastery: Learning Plan

From "what is a pixel?" to production computer vision, VLMs and document AI.
Rule for every topic: **intuition first, math second, code third**, and every module ends with something you build.

---

## How it works

**Every notebook has the same sections:**
1. 🧠 **Intuition**: a plain-English analogy and a picture. No equations.
2. 📐 **Math**: only what's needed, explained line by line.
3. 🔨 **Build from scratch**: NumPy or raw PyTorch, so nothing is magic.
4. 🚀 **Use the real tool**: the same thing with the production library.
5. 📊 **Evaluate**: the right metric for the task, plus a look at where the model fails.
6. 🧪 **Experiment**: "change X, predict what happens, run it."
7. 📝 **Recap**: a short cheat sheet and interview-style questions.

**Where things run:**
- **Mac** (PyTorch on MPS): learning, from-scratch code, small training runs, inference.
- **Kaggle** (free NVIDIA GPU): heavy training, and everything CUDA-specific (TensorRT, Triton, vLLM).
- One notebook works in both places: the first cell detects the platform and sets paths and device.

**Tools used throughout:** OpenCV for image handling in every module, and a small `cvlib/` package of your own code (training loop, metrics, visualization, benchmarking) that grows as you go.

---

## Roadmap

| # | Module | Weeks | Runs on |
|---|---|---|---|
| | **Phase 1 · Foundations** | | |
| 00 | Setup | 1 | Mac |
| 01 | Classic CV with OpenCV | 1–2 | Mac |
| 02 | Neural network basics | 3 | Mac |
| | **Phase 2 · CNNs** | | |
| 03 | CNNs in depth | 4 | Mac |
| 04 | CNN architectures | 5 | Mac + Kaggle |
| 05 | Transfer learning & training craft | 6–7 | Mac + Kaggle |
| | **Phase 3 · Detection & segmentation** | | |
| 06 | Object detection | 8–9 | Kaggle |
| 07 | Segmentation | 10 | Mac + Kaggle |
| | **Phase 4 · Transformers & embeddings** | | |
| 08 | Attention & Vision Transformers | 11–12 | Mac + Kaggle |
| 09 | DETR family | 13 | Mac + Kaggle |
| 10 | Self-supervised learning & CLIP | 14 | Mac + Kaggle |
| 11 | Embeddings & retrieval | 15–16 | Mac + Kaggle |
| 12 | Face recognition | 17 | Mac + Kaggle |
| | **Phase 5 · Vision-language models** | | |
| 13 | VLMs: open source & frontier | 18–19 | Mac + Kaggle |
| | **Phase 6 · Evaluate & optimize** | | |
| 14 | Model evaluation | 20 | Mac |
| 15 | GPU inference optimization | 21–22 | Kaggle |
| | **Phase 7 · Document AI** | | |
| 16 | OCR | 23–24 | Mac + Kaggle |
| 17 | Document inputs & layout | 25–26 | Mac + Kaggle |
| 18 | Document understanding (LayoutLMv3) | 27–29 | Kaggle |
| 19 | Tables & forms | 30 | Mac + Kaggle |
| 20 | OCR-free & VLM document AI | 31 | Kaggle |
| 21 | Document AI in production | 32 | Mac |
| | **Phase 8 · Capstone** | | |
| 22 | Optimized document intelligence platform | 33–34 | Mac + Kaggle |

**Pace:** ~6–8 hrs/week → about 34 weeks.
**Fast track (~24 weeks)**, for document AI and the core skills: 00 → 01a–b → 02 → 03 → 04 → 05 → 06 → 08 → 09 (skim) → 11 → 13 → 14 → 15 → 16–22.

### Core skills → where they're covered

| Skill | Modules |
|---|---|
| CNNs, transfer learning | 03, 04, 05 |
| Attention, Vision Transformers | 08, 09 |
| Embeddings | 10, 11, 12 |
| Open-source & frontier VLMs | 13, 20 |
| Model evaluation | 📊 in every notebook, 14 |
| GPU inference: batching, quantization, mixed precision, ONNX, TensorRT | 15, 22 |

---

## Phase 1 · Foundations

### 00 · Setup
Python 3.12 environment with `uv` (your system Python 3.14 is too new for some CV libraries), PyTorch with MPS, core libraries, and a Kaggle account (phone-verified, so notebooks can use the internet).

### 01 · Classic CV with OpenCV
Four notebooks covering OpenCV's everyday toolbox.
- **01a · Image basics:** images as arrays, BGR vs. RGB, color spaces, histograms, resize/rotate/crop, perspective transforms.
- **01b · Filters & shapes:** convolution (🔨 by hand first), blur, edges (Canny), thresholding, morphology, contours, Hough lines and circles.
- **01c · Features & geometry:** corners, SIFT/ORB keypoints, feature matching, homography with RANSAC, template matching.
- **01d · Video & motion:** webcam input, background subtraction, optical flow, object tracking, Haar face detection.

**Build:** document scanner (reused in OCR later) · panorama stitcher · coin counter · webcam motion detector.

### 02 · Neural network basics
**Learn:** neurons, loss, gradient descent, backpropagation, softmax, overfitting, train/validation/test splits.
**Build:** 🔨 a 2-layer network in NumPy on MNIST, then the same in PyTorch. Its training loop becomes `cvlib/train.py`.

---

## Phase 2 · CNNs

### 03 · CNNs in depth
**Learn:** why convolution suits images · stride, padding, pooling · receptive field · backprop through a conv layer · counting parameters and FLOPs · 1×1, dilated and depthwise-separable convs · BatchNorm (and why it behaves differently in training vs. evaluation).
**Build:** 🔨 conv layer forward and backward in NumPy, checked against PyTorch · a CNN on CIFAR-10 with filter and feature-map visualizations.

### 04 · CNN architectures
**Learn:** the key idea behind each famous model: VGG → ResNet (skip connections) → MobileNet → EfficientNet → ConvNeXt. Why fewer FLOPs doesn't always mean faster.
**Build:** 🔨 ResNet and MobileNet blocks · a benchmark of 6 `timm` models (accuracy vs. speed) · 🔨 Grad-CAM heatmaps.

### 05 · Transfer learning & training craft
**Learn:**
- *Transfer learning:* training only a new final layer → partial fine-tuning → full fine-tuning → LoRA, and how dataset size and domain gap change the best choice.
- *Training craft:* augmentation, regularization, learning-rate schedules, mixed-precision training, class imbalance, experiment tracking, precision/recall/F1.
- *Multi-label classification:* when an image can have several labels at once ("beach + sunset + people"): sigmoid per label instead of softmax, binary cross-entropy loss, choosing a threshold per label, and metrics (per-label AP, micro/macro F1).

**Build:** a transfer learning study on Oxford Pets and an unusual dataset (medical or document images), comparing all four approaches at 1%, 10% and 100% of the data · a multi-label tagger on PASCAL VOC (20 object types, several per image) with tuned per-label thresholds.

---

## Phase 3 · Detection & segmentation

### 06 · Object detection
**Learn:** bounding boxes, IoU, non-max suppression · two-stage (Faster R-CNN) vs. one-stage (YOLO) detectors · anchors and feature pyramids · **mAP**, step by step.
**Build:** 🔨 IoU, NMS and mAP, checked against `pycocotools` · fine-tune Faster R-CNN · train YOLO on objects you label yourself and run it live on your webcam.

### 07 · Segmentation
**Learn:** semantic vs. instance segmentation · U-Net · Mask R-CNN · SAM (click to segment) · Dice and mIoU.
**Build:** 🔨 U-Net for pet segmentation · Mask R-CNN · a SAM click-to-segment demo.

---

## Phase 4 · Transformers & embeddings

### 08 · Attention & Vision Transformers
**Learn:**
- *Attention:* queries, keys and values; self- vs. cross-attention; multi-head; positional encodings (including RoPE); why attention is slow on long inputs and how FlashAttention and windowed attention fix it; attention blocks inside CNNs (SE).
- *Vision Transformers:* image → patches → tokens; why ViTs need more data than CNNs; DeiT and Swin; how CNNs and ViTs behave differently.

**Build:** 🔨 attention and 🔨 a mini-ViT from scratch · a speed/memory benchmark of your attention vs. PyTorch's built-in · fine-tune a ViT and compare with ResNet using the module 05 study · attention-map visualizations.

### 09 · DETR family
**Learn:** detection as "N slots that each find one object" · Hungarian matching · Deformable DETR · RT-DETR (real-time).
**Build:** 🔨 Hungarian matching and 🔨 a mini-DETR on synthetic shapes · fine-tune DETR and RT-DETR and compare them with your YOLO from module 06.

### 10 · Self-supervised learning & CLIP
**Learn:** learning without labels (SimCLR, MAE, DINOv2) · CLIP and SigLIP (images and text in one space) · zero-shot classification · detecting objects from a text description (Grounding DINO) · judging a model's features by training only a simple classifier on top, or by nearest neighbors.
**Build:** 🔨 SimCLR on a small dataset · CLIP text→image search over your photos · Grounding DINO demo.

### 11 · Embeddings & retrieval
**Learn:** what an embedding is and where it comes from · cosine vs. L2 distance · metric learning (triplet loss, InfoNCE, hard negatives) · vector search with FAISS (exact vs. approximate indexes) · uses: search, duplicate detection, clustering, anomaly detection · metrics: Recall@K, mAP@K.
**Build:** compare ResNet, ViT, CLIP, SigLIP and DINOv2 embeddings on the same retrieval task · 🔨 fine-tune with triplet loss · FAISS speed vs. accuracy benchmark · near-duplicate photo finder · anomaly detection on MVTec AD.

### 12 · Face recognition
**Learn:** detect → align → embed → compare · face detectors (YuNet, RetinaFace) · triplet loss and ArcFace · verification vs. identification, choosing a threshold · privacy and consent.
**Build:** 🔨 face alignment · 🔨 train ArcFace on a face dataset subset · webcam attendance system that rejects unknown faces · photo-album face clustering.

Use only your own photos, people who agree, and public research datasets. InsightFace weights are for non-commercial use only.

---

## Phase 5 · Vision-language models

### 13 · VLMs: open source & frontier
**Learn:**
- *How they're built:* vision encoder → connector → LLM; how images become tokens (and why that drives cost); training stages; why VLMs hallucinate.
- *The landscape:* open models (Qwen-VL, InternVL, Gemma 3, SmolVLM, Florence-2) vs. the latest GPT, Gemini and Claude models, compared on accuracy, cost, speed, privacy and license.
- *Using them:* structured JSON output, LoRA/QLoRA fine-tuning, quantized models, running locally on Mac (MLX/Ollama), serving with vLLM.
- *Evaluating them:* public benchmarks (MMMU, DocVQA, OCRBench, POPE for hallucination), why your own test set matters most, LLM-as-judge.

**Build:** 🔨 a tiny VLM (frozen vision encoder + projector + small LLM) · open vs. frontier comparison on your own 100-image test set · QLoRA fine-tune a small open VLM · run a VLM locally on your Mac.

---

## Phase 6 · Evaluate & optimize

### 14 · Model evaluation
**Learn:** choosing the metric that matches the real goal · train/test overlap and other leaks that inflate scores · results by slice (lighting, size, document type) · **confidence intervals and significance tests** for "is model A really better?" · whether confidence scores can be trusted (calibration) · robustness to blur, noise and compression · visual error analysis with FiftyOne.
**Build:** 🔨 `cvlib/eval.py` (confidence intervals, significance test, calibration, slice reports) · re-check earlier comparisons (ResNet vs. ViT, YOLO vs. RT-DETR) · find overlap between a public dataset's train and test sets using embeddings · robustness report for a CNN vs. a ViT.

**Metric cheat sheet**

| Task | Metrics |
|---|---|
| Classification | accuracy, precision/recall/F1, ROC/PR curves, calibration (ECE); multi-label: per-label AP, micro/macro F1 |
| Detection | IoU, mAP@0.5, mAP@0.5:0.95 |
| Segmentation | mIoU, Dice |
| Embeddings / retrieval | Recall@K, mAP@K, nearest-neighbor accuracy |
| Face verification | ROC, TAR@FAR |
| OCR | CER, WER |
| Document extraction | field-level F1, TEDS (tables), ANLS (document Q&A) |
| VLMs | exact match, ANLS, hallucination rate, LLM-as-judge |
| Inference | latency (p50/p95), throughput, GPU memory, accuracy change |

### 15 · GPU inference optimization
All on Kaggle's T4 GPU: TensorRT needs NVIDIA, and the T4 has fast FP16/INT8 cores.

**Learn:**
- *Measuring:* timing GPU code correctly, latency vs. throughput, finding the real bottleneck.
- *Batching:* why it helps, the latency/throughput tradeoff, dynamic batching, keeping the GPU fed.
- *Precision:* FP32 → FP16/BF16 (mixed precision) → INT8; how quantization works; quantizing after training vs. during training; distillation.
- *Runtimes:* `torch.compile`, ONNX and ONNX Runtime, TensorRT, Triton Inference Server, vLLM.

**Build:** 🔨 `cvlib/bench.py` · **the optimization ladder** for a ResNet-50 and a ViT: PyTorch FP32 → FP16 → `torch.compile` → ONNX Runtime → TensorRT FP16 → TensorRT INT8, recording speed, memory and accuracy at each step · 🔨 a simple INT8 quantizer · YOLO in TensorRT on video (end-to-end FPS) · Triton load test with and without dynamic batching · a quantized VLM served with vLLM.

---

## Phase 7 · Document AI

How the modules fit together:

```
PDF / photo / scan
      │
 17  has a text layer? ── yes ──► read text + positions directly
      │ no
 16  OCR: find text, read text
      │
 17  layout: blocks, reading order, special elements
      │
 18  understand: fields, key → value links, document type   (LayoutLMv3)
 19  tables, checkboxes, formulas, charts
 20  or skip OCR entirely                                    (Donut, VLMs)
      │
 21  validate, confidence, human review, PII redaction
      │
structured JSON
```

### 16 · OCR
**Learn:** OCR = find text → read text → clean up · preprocessing (reuses module 01) · text detectors (DBNet) · recognition with CRNN + CTC loss · TrOCR · toolkits: Tesseract, EasyOCR, PaddleOCR, docTR · when a VLM beats classic OCR · CER and WER.
**Build:** 🔨 CRNN + CTC on synthetic text · fine-tune TrOCR on handwriting · an OCR benchmark of 6 tools on the same receipts and photos · phone photo → scanner (01) → text.

### 17 · Document inputs & layout
**Learn:** digital vs. scanned PDFs (don't OCR text you can read directly) · reading words with positions (PyMuPDF, pdfplumber) · cleaning scans and photos (deskew, flatten curved pages, remove shadows) · layout detection (DocLayNet) · reading order with XY-cut · checkboxes, signatures, stamps, barcodes.
**Build:** smart PDF reader (text layer, with OCR fallback) · measure OCR improvement from page cleanup · 🔨 XY-cut reading order · train a layout detector and export it to TensorRT.

### 18 · Document understanding (LayoutLMv3)
**Learn:** why position matters as much as text · LayoutLM v1 → v2 → v3 · how LayoutLMv3's inputs are built (words, boxes, page image) · tasks: labeling fields, linking keys to values, document classification, document Q&A · datasets: FUNSD, CORD, SROIE, RVL-CDIP · entity-level F1.
**Build:** 🔨 LayoutLMv3 inputs by hand · fine-tune on FUNSD (fields + key→value links) and CORD/SROIE receipts · document classifier · **label 50–100 of your own documents and train on them**, the real "new document type" workflow.

### 19 · Tables & forms
**Learn:** finding tables vs. understanding their rows, columns and merged cells · Table Transformer · tools for digital PDFs (Camelot, pdfplumber) · TEDS metric · checkboxes · formulas → LaTeX · charts → data.
**Build:** 🔨 table extractor with OpenCV line detection · Table Transformer → pandas DataFrame, scored with TEDS · checkbox detector.

### 20 · OCR-free & VLM document AI
**Learn:** what you gain and lose by skipping OCR · Donut and Nougat · all-in-one PDF parsers (Docling, Marker, MinerU, olmOCR) · VLM extraction with a JSON schema and validation · search across documents with ColPali.
**Build:** fine-tune Donut and compare with your LayoutLMv3 pipeline · PDF parser comparison · VLM extraction checked with Pydantic · "chat with your PDFs" with answers cited to pages.

### 21 · Document AI in production
**Learn:** per-field accuracy · confidence scores and sending unsure fields to a person · checks like "line items add up to the total" · generating realistic fake documents for training · PII redaction · choosing a small model vs. a VLM per document type.
**Build:** evaluation harness for extraction · synthetic invoice generator (trained only on fake data, tested on real) · PII redaction tool · add validation and review flags to your extractor.

---

## Phase 8 · Capstone

### 22 · Optimized document intelligence platform
A Gradio app that takes a mixed pile of documents and returns structured data:

**upload → read or OCR → classify → route to the right extractor (LayoutLMv3, Donut or VLM) → tables → validate → human review for unsure fields → JSON/CSV**

It must include:
- **Evaluation:** per-field F1 with confidence intervals, results by document type, a gallery of failures.
- **Optimization:** TensorRT for the vision models, batched OCR, quantized VLM on vLLM, and pages/minute before vs. after.
- **A short write-up** comparing pipeline choices on accuracy, speed and cost.

---

## Project layout

```
cv-mastery/
├── PLAN.md
├── README.md          # progress checklist
├── pyproject.toml     # dependencies (uv)
├── notebooks/         # one folder per module: 00_setup/, 01_classic_cv/, ...
├── cvlib/             # your reusable code: train, viz, detection, embeddings, eval, bench, docs
├── data/              # not committed
└── notes/             # your own cheat sheets
```

## Kaggle tips
- Use the **T4** GPU (needed for module 15). Free quota: about 30 GPU hours/week, 12 hours per session.
- Attach datasets already on Kaggle instead of downloading them each run.
- Use **Save & Run All** for long training so it keeps running after you close the tab; saved models can be loaded into later notebooks.
- Upload `cvlib/` as a private Kaggle dataset so your code works the same on Kaggle and your Mac.
- Debug on a small sample locally, then run the full job on Kaggle.
- TensorRT engines only work on the GPU type they were built on.
