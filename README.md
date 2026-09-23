# Local EM Segmentation & 3D Reconstruction Hub

A local Python application for **electron microscopy image preprocessing, deep-learning segmentation, assisted annotation, local model training, and interactive 3D reconstruction**.

The application was developed primarily for the analysis of **Serial Block-Face Scanning Electron Microscopy (SBF-SEM)** datasets and is intended as a local research environment for working with large biological image stacks.

The current version combines several stages of the image-analysis workflow in a single Gradio-based interface:

1. Image preprocessing and automated segmentation
2. SAM-assisted annotation and ground-truth generation
3. Local U-Net training
4. Interactive 3D reconstruction and visualization

---

## Project status

> **Research prototype / work in progress**

The application is functional and is currently used for research and development, but it remains under active development.

The repository is made public primarily to demonstrate my work in:

- computational bioimaging;
- electron microscopy image analysis;
- deep-learning segmentation;
- annotation workflow development;
- quantitative 3D reconstruction;
- development of local tools for biological image analysis.

The current version should **not be considered production-ready software**.

Interfaces, dependencies, model handling, and parts of the workflow may change as development continues.

**No open-source license is currently provided.**

---

# Main features

## EM image preprocessing

The application contains a preprocessing pipeline designed for grayscale electron microscopy images.

The current workflow includes:

- support for 8-bit and 16-bit images;
- grayscale conversion where necessary;
- non-local means denoising;
- percentile-based intensity normalization;
- CLAHE local contrast enhancement;
- edge-preserving bilateral filtering;
- intensity inversion;
- final sharpening prior to segmentation.

Preprocessing parameters such as the **CLAHE clip limit** and **tile-grid size** can be adjusted directly from the graphical interface.

---

## Automated deep-learning segmentation

The segmentation module uses a **U-Net architecture with a ResNet18 encoder** for binary segmentation of ultrastructural features.

Current functionality includes:

- local model inference;
- automatic hardware selection;
- CUDA support when available;
- Apple MPS support when available;
- CPU fallback;
- configurable prediction-overlay color;
- visualization of both the preprocessed image and segmentation overlay.

The current segmentation model expects grayscale images and performs inference on images resized internally to **512 × 512 pixels**, after which the prediction is mapped back to the original image dimensions.

---

## SAM-assisted annotation

The application integrates the **Segment Anything Model (SAM)** to assist with generation of ground-truth masks.

The annotation interface supports:

- positive point prompts;
- negative point prompts;
- multiple SAM mask proposals;
- interactive selection of alternative mask scales;
- undo of the most recent prompt;
- clearing the current object;
- committing multiple segmented objects into a global mask;
- manual mask retouching;
- saving image/mask pairs for subsequent model training.

This module is intended to reduce the amount of fully manual annotation required when preparing microscopy datasets for supervised segmentation.

---

## Ground-truth dataset generation

Annotated data can be saved directly into a dataset structure containing:

```text
dataset/
├── images/
└── masks/
```

The original microscopy image and the corresponding binary mask are stored as paired samples that can subsequently be used by the local training module.

---

# Local network training

The application includes a built-in training interface for retraining the segmentation model from locally generated image/mask datasets.

The current training workflow includes:

- local image and mask directories;
- configurable number of epochs;
- configurable batch size;
- configurable learning rate;
- preprocessing using the same image-processing pipeline used during inference;
- preprocessing cache to reduce repeated computation across epochs;
- U-Net training using a combined:
  - Dice loss;
  - binary cross-entropy loss;
- Adam optimization;
- CUDA mixed-precision training when supported;
- automatic saving of trained network weights.

After training, the model weights are saved as:

```text
structure_model.pth
```

---

# 3D Volume Reconstruction

The application includes an interactive reconstruction module for converting a stack of 2D segmentation masks into 3D surface models.

The current workflow supports:

- loading ordered mask slices from a folder;
- binary mask preparation;
- reconstruction of volumetric objects from Z-stacks;
- optional filling of hollow structures;
- adaptive repair of gaps in object boundaries;
- preservation of neighboring object separation;
- connected-component detection at full resolution;
- configurable downsampling for mesh generation;
- configurable Z-axis scaling;
- marching-cubes surface extraction;
- individual coloring of reconstructed objects;
- interactive Plotly-based 3D visualization;
- selection and highlighting of individual structures;
- optional 3D bounding-box visualization;
- adjustable object opacity.

A major focus of the reconstruction workflow is avoiding artificial merging of nearby biological structures during mask filling and downsampling.

---

## Object-aware solidification

Ring-like structures such as myelin sheaths can create difficulties during volumetric reconstruction when segmentation masks contain small discontinuities.

The application therefore includes an object-aware mask-filling workflow that:

1. identifies separate 2D connected components;
2. processes individual components independently;
3. adaptively repairs discontinuities in object boundaries;
4. assigns spatial ownership using nearest-component information;
5. prevents filled regions from crossing into neighboring objects;
6. performs additional Z-direction correction after volume assembly.

This was implemented specifically to reduce erroneous merging of densely packed neighboring structures during reconstruction.

---

# Graphical interface

The application is built using **Gradio** and currently contains four main tabs.

## 1. Segmentation — Inference

Used for:

- uploading raw EM images;
- preprocessing;
- running the trained U-Net;
- displaying the processed image;
- visualizing the predicted segmentation overlay.

---

## 2. Smart SAM Annotation Studio

Used for:

- assisted annotation;
- positive and negative point prompts;
- SAM mask generation;
- mask refinement;
- manual correction;
- saving ground-truth datasets.

---

## 3. Local Network Training

Used for:

- selecting image and mask datasets;
- configuring training parameters;
- running local U-Net training;
- saving newly trained model weights.

---

## 4. 3D Volume Reconstruction

Used for:

- loading mask Z-stacks;
- reconstructing individual structures;
- configuring reconstruction resolution;
- correcting Z-axis scaling;
- rendering reconstructed surfaces;
- selecting and highlighting individual 3D objects.

---

# Technologies

The project currently uses:

- **Python**
- **PyTorch**
- **segmentation-models-pytorch**
- **Segment Anything Model (SAM)**
- **OpenCV**
- **NumPy**
- **SciPy**
- **scikit-image**
- **Plotly**
- **Gradio**

---

# Repository structure

```text
local-em-segmentation-hub/
│
├── app.py
│   Main application containing preprocessing,
│   segmentation, annotation, training, and
│   3D reconstruction functionality.
│
├── run_pipeline.sh
│   Starts the application and records the
│   running process ID.
│
├── stop_pipeline.sh
│   Safely stops the application started by
│   run_pipeline.sh.
│
├── requirements.txt
│   Python dependencies used by the application.
│
├── .gitignore
│   Excludes environments, datasets, model
│   checkpoints, microscopy data, and temporary files.
│
└── README.md
```

Model weights, datasets, raw microscopy images, and temporary output files are intentionally **not included** in the repository.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Nochebald/local-em-segmentation-hub.git
cd local-em-segmentation-hub
```

---

## 2. Create a virtual environment

```bash
python3 -m venv env
```

Activate it:

```bash
source env/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Installation time will depend on your system and on the PyTorch configuration being installed.

GPU-enabled PyTorch installations may require system-specific configuration depending on the CUDA environment.

---

# Model files

## Segmentation model

The application currently expects a trained U-Net model named:

```text
structure_model.pth
```

to be located in the project directory.

The trained model is **not currently distributed with this repository**.

The architecture currently used by the application is:

```text
U-Net
Encoder: ResNet18
Input channels: 1
Output classes: 1
```

---

## Segment Anything Model

The application uses the SAM ViT-B checkpoint:

```text
sam_vit_b_01ec64.pth
```

If this checkpoint is not found locally, the application will attempt to download it automatically when launched.

The SAM checkpoint is therefore not stored in this GitHub repository.

---

# Running the application

Make the launcher executable if necessary:

```bash
chmod +x run_pipeline.sh
chmod +x stop_pipeline.sh
```

Start the application:

```bash
./run_pipeline.sh
```

The launcher:

1. detects the directory containing the repository;
2. activates the local Python environment;
3. starts the application;
4. records the application process ID.

The Gradio interface should then open in your web browser.

---

## Running directly

The application can also be started manually:

```bash
source env/bin/activate
python app.py
```

---

# Stopping the application

If the application was started with:

```bash
./run_pipeline.sh
```

it can be stopped using:

```bash
./stop_pipeline.sh
```

The stop script uses the recorded process ID to stop the specific application instance rather than terminating arbitrary Python processes.

---

# Supported image formats

Different parts of the current application support common microscopy image formats including:

```text
.tif
.tiff
.png
.jpg
.jpeg
```

The software was primarily developed and tested using electron microscopy image datasets.

---

# Typical workflow

A possible workflow using the application is:

```text
Raw EM image
     │
     ▼
Image preprocessing
     │
     ▼
Initial U-Net segmentation
     │
     ├───────────────┐
     │               │
     ▼               ▼
SAM-assisted     Existing
annotation       ground truth
     │               │
     └───────┬───────┘
             ▼
      Training dataset
             │
             ▼
      Local U-Net training
             │
             ▼
      Segmentation model
             │
             ▼
       Mask Z-stack
             │
             ▼
      3D reconstruction
             │
             ▼
 Interactive visualization
```

---

# Hardware

The application automatically attempts to use the best available PyTorch backend:

```text
NVIDIA CUDA GPU
      ↓
Apple Metal / MPS
      ↓
CPU
```

GPU acceleration is particularly useful for model training and SAM inference.

3D reconstruction performance depends primarily on:

- number of mask slices;
- image dimensions;
- number of detected objects;
- selected downsampling factor;
- CPU performance;
- available memory.

---

# Current limitations

This project is still under development.

Current limitations include:

- the trained U-Net model is not distributed with the repository;
- installation has not yet been packaged into a standalone installer;
- the interface has primarily been tested in my own research environment;
- validation across different operating systems and hardware configurations is limited;
- preprocessing parameters may require adjustment for substantially different EM datasets;
- the segmentation workflow currently focuses on binary segmentation;
- documentation and automated testing are still being expanded;
- model management and selection are currently basic;
- the project has not yet been prepared as a stable public software release.

---

# Planned development

Possible future improvements include:

- improved model loading and model selection;
- support for multiple segmentation models;
- improved installation and environment management;
- expanded dataset management;
- additional segmentation quality metrics;
- improved progress reporting;
- automated validation;
- broader microscopy-format support;
- improved 3D reconstruction controls;
- export of reconstructed meshes;
- improved documentation;
- automated testing;
- interface refinements.

---

# Intended use

This project was developed as a **research and bioimage-analysis tool**.

It is currently intended primarily for:

- research development;
- microscopy image-analysis workflows;
- exploration of segmentation approaches;
- generation of training annotations;
- local model training;
- visualization of segmented 3D biological structures;
- demonstration of computational bioimaging methods.

It is **not intended for clinical or diagnostic use**.

---

# Development background

The project grew from my work on quantitative analysis of **SBF-SEM datasets of peripheral nerves and Dorsal Root Ganglia**, where segmentation, annotation, large image stacks, and reconstruction of individual ultrastructural components are recurring parts of the analysis workflow.

The aim is to progressively integrate these separate analysis steps into a single local environment that can be used without requiring the user to interact directly with each underlying Python component.

---

# Related work

My recent research includes work on:

- automated segmentation of peripheral nerve ultrastructure;
- SBF-SEM image preprocessing;
- stain-independent deep-learning segmentation;
- reconstruction of peripheral nerve and Dorsal Root Ganglion structures;
- quantitative morphometric analysis of axons and organelles;
- 3D visualization of biological ultrastructure.

Relevant publications will be linked here as the project documentation develops.

---

# Screenshots

Screenshots and example outputs will be added as development continues.

Planned examples include:

### Segmentation

```text
Raw EM image → Preprocessed image → U-Net prediction
```

### SAM annotation

```text
EM image → SAM prompts → Refined ground-truth mask
```

### Training

```text
Local dataset → U-Net training → Updated model
```

### 3D reconstruction

```text
Mask Z-stack → Individual object detection → Interactive 3D model
```

---

# Author

**Vitaly Borisovs, Ph.D.**

Research Associate  
University of Milan-Bicocca

Research interests:

- computational bioimaging;
- volume electron microscopy;
- SBF-SEM;
- biological image segmentation;
- deep learning;
- 3D reconstruction;
- quantitative morphometry.

GitHub:  
https://github.com/Nochebald

Portfolio:  
https://nochebald.github.io/

---

# License and reuse

**No open-source license is currently provided for this repository.**

The source code is publicly visible primarily for research demonstration and portfolio purposes.

The project remains under active development and is not currently released for unrestricted reuse, redistribution, or incorporation into other software projects.

Please contact the author regarding potential reuse or collaboration.

Copyright © 2026 Vitaly Borisovs.
