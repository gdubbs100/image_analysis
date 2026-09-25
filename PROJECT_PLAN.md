# Project Plan: Object Detection & Image Segmentation

A hands-on path from "I can run a model" to "I understand what it's doing and can adapt it."
Each stage has a goal, something to build, and questions to answer before moving on.

**Environment:** Python 3.11, Windows, CPU only (no NVIDIA GPU). Everything below runs on CPU;
training stages use small models/datasets, or free Colab GPUs when noted.

## The concepts in one minute

| Task | Output | Example |
|---|---|---|
| Classification | one label per image | "cat" |
| Object detection | boxes + labels + scores | "cat at (x,y,w,h), 0.92" |
| Semantic segmentation | a class label per pixel | all cat pixels are "cat" |
| Instance segmentation | a mask per object | cat #1 mask, cat #2 mask |
| Promptable segmentation | mask from a point/box/text prompt | SAM: click a thing, get its mask |

Key vocabulary to pick up along the way: bounding box, IoU, confidence threshold,
non-max suppression (NMS), anchor boxes, mAP, mask, Dice/IoU score, backbone, fine-tuning.

## Stage 0: Setup

- [ ] Create a virtual environment (`python -m venv .venv`) and add `.venv/` to `.gitignore`
- [ ] Install: `torch torchvision ultralytics opencv-python matplotlib jupyter pillow`
- [ ] Create folders: `images/` (your own test photos), `notebooks/`, `outputs/`
- [ ] Collect ~10 test photos: a street scene, a crowd, a cluttered desk, pets, something hard (occlusion, small objects, bad lighting)

**Done when:** `import torch, ultralytics` works and you have test images.

## Stage 1: Run a pretrained detector (an afternoon)

**Goal:** see what detection output looks like before learning how it works.

- [ ] Run YOLO (`yolo11n.pt`, the smallest) on your test images with `ultralytics`
- [ ] Draw boxes yourself with OpenCV/matplotlib from the raw results (don't just use `.plot()`)
- [ ] Print the raw outputs: boxes, class ids, confidences
- [ ] Vary the confidence threshold (0.1, 0.25, 0.5, 0.75) and watch false positives/negatives change
- [ ] Try a webcam or video file with the same model

**Questions to answer:**
- What are the 80 COCO classes, and what happens with objects that aren't among them?
- Where does it fail on your hard images, and why?
- How does speed differ between `yolo11n` and `yolo11m` on your CPU?

## Stage 2: Detection fundamentals (by hand)

**Goal:** implement the pieces that models hide, so the metrics stop being magic.

- [ ] Write `iou(box_a, box_b)` from scratch; test on hand-drawn cases
- [ ] Write NMS from scratch; compare your output to `torchvision.ops.nms`
- [ ] Load a small labeled set (e.g. 50 COCO val images) and compute precision/recall at IoU 0.5
- [ ] Compute mAP with `pycocotools` or `torchmetrics`, then read what each number means (mAP@0.5 vs mAP@0.5:0.95)
- [ ] Plot a precision-recall curve

**Questions to answer:**
- Why does NMS exist? What breaks if you turn it off?
- Why can a detector have high recall and low precision, and which do you care about for your use case?

## Stage 3: Compare detector families

**Goal:** understand the design space, not just one model.

- [ ] Run torchvision's **Faster R-CNN** (two-stage) and **SSD/RetinaNet** (one-stage) on the same images as YOLO
- [ ] Compare accuracy, speed, and failure modes side by side (a table in a notebook)
- [ ] Try an open-vocabulary detector (**OWL-ViT** or **Grounding DINO** via Hugging Face `transformers`): detect things by typing a text description
- [ ] Read one overview: how two-stage vs one-stage vs transformer (DETR) detectors differ

**Questions to answer:**
- When would you pick a two-stage over a one-stage detector?
- What does open-vocabulary detection give up compared to a fine-tuned YOLO?

## Stage 4: Run pretrained segmentation

**Goal:** see all three flavors of segmentation output.

- [ ] **Semantic:** torchvision DeepLabV3 (or SegFormer via `transformers`) — overlay a per-pixel class map
- [ ] **Instance:** YOLO-seg (`yolo11n-seg.pt`) or Mask R-CNN — get one mask per object, count objects
- [ ] **Promptable:** **Segment Anything (SAM / SAM 2)** — click a point or draw a box, get a mask; try "segment everything" mode
- [ ] Extract masks as NumPy arrays; compute area, centroid, bounding box from each mask
- [ ] Cut an object out of an image using its mask (background removal)

**Questions to answer:**
- What's the practical difference between semantic and instance segmentation? Give a case where each is the right tool.
- Why is SAM able to segment things it was never taught a name for?

## Stage 5: Fine-tune on your own data

**Goal:** the skill that matters most in practice: adapting a model to a new problem.

- [ ] Pick a small, concrete problem (ideas below) and gather 100-300 images
- [ ] Label with **Label Studio**, **CVAT**, or **Roboflow** (free tiers exist); or use a public dataset from Roboflow Universe
- [ ] Split train/val, check the YOLO dataset format (`images/`, `labels/`, `data.yaml`)
- [ ] Fine-tune `yolo11n` (detection) — use free **Google Colab** GPU if CPU is too slow
- [ ] Read the training curves and confusion matrix; find the worst validation images
- [ ] Repeat with segmentation (`yolo11n-seg`), optionally using SAM to speed up mask labeling

**Project ideas (pick one):**
- Count items in a photo (coins, screws, fruit, cars in a car park)
- Detect something specific in your home/garden/work (pets, birds at a feeder, tools)
- Segment a region of interest (leaf disease, road cracks, a product on a shelf)

**Questions to answer:**
- How does performance change with 25 vs 100 vs 300 training images?
- What does augmentation do, and when does it hurt?
- What does overfitting look like in these curves?

## Stage 6: Go deeper (choose your own adventure)

- [ ] **Understand the architecture:** read the YOLO and Mask R-CNN papers; trace a forward pass through torchvision's Faster R-CNN
- [ ] **Train from scratch** a tiny segmentation net (U-Net) on a small dataset (e.g. Oxford-IIIT Pet) in plain PyTorch
- [ ] **Tracking:** add object tracking on video (`yolo track` with ByteTrack) — count people crossing a line
- [ ] **Deployment:** export to ONNX and run with ONNX Runtime; measure CPU speedup
- [ ] **Build a small app:** Gradio or Streamlit UI: upload an image, choose a model, see boxes/masks
- [ ] **Evaluation rigor:** build a failure-analysis notebook grouping errors by size, occlusion, lighting

## Suggested repo layout

```
image_analysis/
  PROJECT_PLAN.md
  README.md
  images/            # your test photos (git-ignored if large)
  notebooks/
    01_pretrained_detection.ipynb
    02_iou_nms_map.ipynb
    03_detector_comparison.ipynb
    04_segmentation.ipynb
    05_finetuning.ipynb
  src/               # reusable helpers (drawing, metrics)
  data/              # datasets (git-ignored)
  outputs/           # results, weights (git-ignored)
```

## Resources

- Ultralytics docs (YOLO): https://docs.ultralytics.com
- torchvision detection/segmentation models: https://pytorch.org/vision/stable/models.html
- Segment Anything: https://github.com/facebookresearch/sam2
- Hugging Face vision tasks (object detection, image segmentation): https://huggingface.co/tasks
- COCO dataset & metrics: https://cocodataset.org
- Roboflow Universe (public labeled datasets): https://universe.roboflow.com
- CS231n lecture notes on detection/segmentation (Stanford): good for theory

## Progress log

| Date | Stage | Notes |
|---|---|---|
| | | |
