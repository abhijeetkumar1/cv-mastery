"""cvlib: reusable code written while working through the course.

Modules are added as the course progresses:
    env   - platform/device detection, seeding            (00)
    data  - cached downloads of sample images and videos   (01)
    viz   - showing images, training curves, GIFs           (01)
    train - the standard PyTorch training loop              (02, + mixed precision & schedules in 05)
    datasets - CIFAR-10, Imagenette, VOC, image caches       (03-05)
    loaders  - GPU-side batching and augmentation            (03, 05)
    models   - small reusable models (SimpleCNN)             (03)
    lora     - LoRA adapters for Linear/Conv2d               (05a)
    metrics  - precision/recall/F1, AP, multi-label metrics  (05b, 05c)
    tracking - a minimal experiment log                      (05b)
"""
