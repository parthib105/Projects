# Cahn-Hilliard Image Segmentation

This project implements and evaluates an image segmentation model based on a modified Cahn-Hilliard (CH) partial differential equation (PDE). The model's performance is compared against classical thresholding methods (Otsu, Maximum Entropy) and the Chan-Vese level-set method across various image types, including noisy, natural, and clinical images.
---
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
---
## ⚙️ Algorithm Outline

The following algorithm outlines the steps for implementing the segmentation model using convexity splitting and the Fourier spectral method.

### Step 1: Input
Load the original image \( u_0(x) \).  
Set parameters:
- \( \epsilon \)
- \( \lambda \)
- Threshold \( \gamma \)
- Time step \( \Delta t \)
- Number of iterations \( N \)
- Grid spacings \( \Delta x, \Delta y \)

### Step 2: Preprocessing
Define the convex splitting constants:

\[
C_1 = \frac{3}{\epsilon}, \quad C_2 = 
\begin{cases}
3\lambda & \text{(Stage 1)} \\
0 & \text{(Stage 2)}
\end{cases}
\]

### Step 3: Spatial Discretization
a) Define a rectangular grid of size \( m \times n \).  
b) Compute the Fourier symbol \( M_{i,j} \) using:

\[
M_{i,j} = \frac{2}{\Delta x^2} \left[ \cos\left( \frac{2\pi i}{m} \right) - 1 \right] + \frac{2}{\Delta y^2} \left[ \cos\left( \frac{2\pi j}{n} \right) - 1 \right]
\]

### Step 4: Initialization
a) Set \( U^0 = u_0 \).  
b) Compute the discrete Fourier transform \( \widehat{U}^0 \) of \( U^0 \).  
c) Precompute the denominator in Fourier space:

\[
D = 1 + \Delta t \left( \epsilon M^2 - C_1 M + C_2 \right)
\]

### Step 5: Time-Stepping Loop
For \( k = 0, 1, 2, \ldots, N-1 \):

a) Compute the nonlinear term:

\[
W'(U^k) = 4 U^k (U^k - 1)(U^k - \gamma)
\]

b) Calculate the numerator in Fourier space:

\[
\text{Num} = \left( 1 - \Delta t C_1 M + \Delta t C_2 \right) \widehat{U}^k + \Delta t \left[ \frac{1}{\epsilon} \widehat{W'(U^k)} + \lambda (\widehat{u_0 - U^k}) \right]
\]

c) Update the Fourier coefficient:

\[
\widehat{U}^{k+1} = \frac{\text{Num}}{D}
\]

d) Inverse Fourier transform \( \widehat{U}^{k+1} \) to obtain \( U^{k+1} \) in the physical space.

### Step 6: Stage Process
Perform the above time-stepping for:

- **Stage 1**: Use parameter set \( (\epsilon, \lambda) = (0.01, 10^6) \) for a designated number of iterations.  
- **Stage 2**: Use the output of Stage 1 as the initial condition with parameters \( (\epsilon, \lambda) = (0.001, 0) \) for further iterations.

### Step 7: Output
The final solution \( U \) represents the segmented image.

--- 

Let me know if you'd like this wrapped in a full `README.md` template with a title, description, and usage example.
 
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

## 📚 References
[1] R. Vijayakrishna, B. V. Rathish Kumar, and A. Halim, A PDE Based Image Segmentation Using Fourier Spectral Method, Differential Equations and Dynamical Systems, 30(2):469-484, 2018.
