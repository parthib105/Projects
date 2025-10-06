# Cahn-Hilliard Image Segmentation

This project implements and evaluates an image segmentation model based on a modified Cahn-Hilliard (CH) partial differential equation (PDE). The model's performance is compared against classical thresholding methods (Otsu, Maximum Entropy) and the Chan-Vese level-set method across various image types, including noisy, natural, and clinical images.

## 📜 Proposed Model and Governing Equation

The segmentation model is described by a fourth-order PDE that governs the evolution of a phase field, $u(x,t)$, which represents the segmented image.

The governing equation is:
$$u_{t}=-\epsilon~\Delta^{2}u+\frac{1}{\epsilon}\Delta(4u(u-1)(u-\gamma))+\lambda(u_{0}-u)$$

where:
* **$u(x,t)$**: The phase field representing the image at a given point $x$ and time $t$.
* **$\epsilon > 0$**: A parameter that characterizes the width of the transition layer between phases (i.e., object and background).
* **$\gamma \in (0,1)$**: The threshold parameter that determines the segmentation level.
* **$\lambda > 0$**: A fidelity parameter that enforces closeness to the original image, $u_0(x)$.
* **$u_0(x)$**: The initial input image.

The evolution starts with the image itself, $u(x,0) = u_0(x)$, and assumes homogeneous Neumann boundary conditions, ensuring no flux across the image boundaries.

## 🧪 Experiments and Observations

The model was tested in four different scenarios to evaluate its robustness, adaptivity, and accuracy.

### **Experiment 1: Noisy Airplane Image**
* This experiment tested the model's robustness to noise by segmenting a noisy image of an airplane and comparing it to Otsu's and Kapur's Maximum Entropy methods.
* **Observation**: The Cahn-Hilliard method produced smoother and more accurate boundaries, demonstrating superior noise robustness compared to the classical methods.


### **Experiment 2: Snake Image**
* This experiment evaluated the effect of different thresholding strategies for the $\gamma$ parameter, using a fixed value, Otsu's threshold, and a Maximum Entropy threshold.
* **Observation**: Using an entropy-based threshold ($\gamma=T_{MaxEnt}$) can help enhance the separation of foreground and background when intensity distributions overlap. However, both Otsu and MaxEnt thresholds resulted in some background patches being incorrectly captured as foreground.


### **Experiment 3: Clinical Ear CT Slice**
* Here, the CH method was compared with Otsu thresholding and the Chan-Vese (level-sets) active contour model on a clinical ear CT slice.
* **Observation**: The CH method, when combined with the adaptivity of Otsu's threshold for $\gamma$, preserved anatomical detail while effectively reducing noise and spurious regions, outperforming the other methods.


### **Experiment 4: Clinical Angiography Image**
* This experiment focused on segmenting blood vessels, which are typically sparse and have low contrast, comparing the CH method with Otsu thresholding.
* **Observation**: The CH method's spatial regularization helps suppress noise and enhance the continuity of thin vessels. It produces a more accurate and visually coherent segmentation than Otsu's method alone, which can struggle with low-contrast, sparse features.


## 🚀 Setup and Usage

Follow these steps to set up the environment and run the experiments.

### **1. Clone the Repository**
```bash
git clone https://github.com/parthib105/Projects.git
cd Image-Segmentation-Cahn-Hilliard
```

### **2. Set up a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### **3. Install Dependencies**
The required packages are listed in requirements.txt.
```bash
pip install -r requirements.txt
```

### **4. Add data**
Place your input images into the `data/` directory

### **5. Run Experiments**
You can run any of the four experiments using the `main.py` script with the `--experiment` flag
```bash
# Run Experiment 1 (Noisy Airplane)
python src/main.py --experiment 1

# Run Experiment 2 (Snake)
python src/main.py --experiment 2

# Run Experiment 3 (Ear CT Slice)
python src/main.py --experiment 3

# Run Experiment 4 (Angiography)
python src/main.py --experiment 4
```
After running an experiment, the resulting plots will be saved in the `bash results/` directory, and a confirmation message will be printed to the console.

```bash
plots are saved to: results/experiment_1/
```
