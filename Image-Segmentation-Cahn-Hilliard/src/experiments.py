import os
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import chan_vese

from .segmentation_utils import load_image_gray, kapur_entropy_threshold
from .ch_model import solve_cahn_hilliard_fourier
from .visualization import save_comparison_plot

def run_experiment_1(data_dir, results_dir):
    """Experiment on the noisy airplane image."""
    img_path = os.path.join(data_dir, 'noisy_airplane.png')
    img = load_image_gray(img_path)
    if img is None: return

    # Otsu's method
    otsu_thresh = threshold_otsu(img)
    otsu_seg = img > otsu_thresh

    # Kapur's Max Entropy method
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 1))
    hist_norm = hist / hist.sum()
    maxent_thresh = kapur_entropy_threshold(hist_norm, img.size)
    maxent_seg = img > maxent_thresh

    # Cahn-Hilliard method
    gamma = 0.1
    ch_params_s1 = {'eps1': 1e-4, 'lambda_val1': 0, 'C1_factor1': 3.0, 'C2_factor1': 3.0, 'dt1': 1.0, 'itmax1': 100}
    ch_params_s2 = {'eps2': 0.4, 'lambda_val2': 0, 'C1_factor2': 3.0, 'C2_factor2': 3.0, 'dt2': 1.0, 'itmax2': 50}
    ch_seg_u = solve_cahn_hilliard_fourier(img, gamma, **ch_params_s1, **ch_params_s2)
    ch_seg = ch_seg_u > gamma

    # Visualization
    images = [img, otsu_seg, maxent_seg, ch_seg]
    titles = ['Original Noisy Airplane', f'Otsu ({otsu_thresh:.2f})', f'MaxEnt ({maxent_thresh:.2f})', f'CH (γ={gamma:.1f})']
    main_title = "Fig 1 Replication: Noisy Airplane Image"
    save_path = os.path.join(results_dir, "fig1_noisy_airplane.png")
    save_comparison_plot(images, titles, main_title, save_path)

def run_experiment_2(data_dir, results_dir):
    """Experiment on the snake image with different gamma values."""
    img_path = os.path.join(data_dir, 'snake.png')
    img = load_image_gray(img_path)
    if img is None: return

    # Common CH parameters
    ch_params_s1 = {'eps1': 0.01, 'lambda_val1': 1e-6, 'C1_factor1': 3.0, 'C2_factor1': 3.0, 'dt1': 1.0, 'itmax1': 100}
    ch_params_s2 = {'eps2': 0.001, 'lambda_val2': 0.0, 'C1_factor2': 3.0, 'C2_factor2': 3.0, 'dt2': 1.0, 'itmax2': 100}
    
    # CH with fixed gamma
    gamma_fixed = 0.5
    ch_seg_u1 = solve_cahn_hilliard_fourier(img, gamma_fixed, **ch_params_s1, **ch_params_s2)
    ch_seg1 = ch_seg_u1 > 0.5

    # CH with Otsu gamma
    otsu_thresh = threshold_otsu(img)
    ch_seg_u2 = solve_cahn_hilliard_fourier(img, otsu_thresh, **ch_params_s1, **ch_params_s2)
    ch_seg2 = ch_seg_u2 > 0.5

    # CH with MaxEnt gamma
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 1))
    hist_norm = hist / hist.sum()
    maxent_thresh = kapur_entropy_threshold(hist_norm, img.size)
    ch_seg_u3 = solve_cahn_hilliard_fourier(img, maxent_thresh, **ch_params_s1, **ch_params_s2)
    ch_seg3 = ch_seg_u3 > 0.5

    # Visualization
    images = [img, ch_seg1, ch_seg2, ch_seg3]
    titles = ['Original Snake', f'CH (γ={gamma_fixed:.1f})', f'CH with Otsu γ ({otsu_thresh:.2f})', f'CH with MaxEnt γ ({maxent_thresh:.2f})']
    main_title = "Fig 2 Replication: Snake Image"
    save_path = os.path.join(results_dir, "fig2_snake.png")
    save_comparison_plot(images, titles, main_title, save_path)

def run_experiment_3(data_dir, results_dir):
    """Experiment on the Ear CT slice."""
    img_path = os.path.join(data_dir, 'ear_ct_slice.png')
    img = load_image_gray(img_path)
    if img is None: return
    
    # Otsu's method
    otsu_thresh = threshold_otsu(img)
    otsu_seg = img > otsu_thresh
    
    # Chan-Vese method
    cv_seg = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, dt=0.5, max_num_iter=200)

    # Cahn-Hilliard with Otsu gamma
    gamma = otsu_thresh
    ch_params_s1 = {'eps1': 0.01, 'lambda_val1': 1e-6, 'C1_factor1': 3.0, 'C2_factor1': 3.0, 'dt1': 1.0, 'itmax1': 100}
    ch_params_s2 = {'eps2': 0.001, 'lambda_val2': 0, 'C1_factor2': 3.0, 'C2_factor2': 3.0, 'dt2': 1.0, 'itmax2': 100}
    ch_seg_u = solve_cahn_hilliard_fourier(img, gamma, **ch_params_s1, **ch_params_s2)
    ch_seg = ch_seg_u > gamma
    
    # Visualization
    images = [img, otsu_seg, cv_seg, ch_seg]
    titles = ['Original Ear CT Slice', f'Otsu ({otsu_thresh:.2f})', 'Chan-Vese', f'CH with Otsu γ ({gamma:.2f})']
    main_title = "Fig 3 Replication: Ear CT Slice"
    save_path = os.path.join(results_dir, "fig3_ear_ct.png")
    save_comparison_plot(images, titles, main_title, save_path)
    
def run_experiment_4(data_dir, results_dir):
    """Experiment on the angiography image."""
    img_path = os.path.join(data_dir, 'angiography.png')
    img = load_image_gray(img_path)
    if img is None: return

    # Otsu's method
    otsu_thresh = threshold_otsu(img)
    otsu_seg = img > otsu_thresh

    # Cahn-Hilliard with Otsu gamma
    gamma = otsu_thresh
    ch_params_s1 = {'eps1': 5, 'lambda_val1': 1, 'C1_factor1': 3.0, 'C2_factor1': 3.0, 'dt1': 10, 'itmax1': 100}
    ch_params_s2 = {'eps2': 1e-7, 'lambda_val2': 1e6, 'C1_factor2': 3.0, 'C2_factor2': 3.0, 'dt2': 10, 'itmax2': 10}
    ch_seg_u = solve_cahn_hilliard_fourier(img, gamma, **ch_params_s1, **ch_params_s2)
    ch_seg = ch_seg_u > gamma

    # Visualization
    images = [img, otsu_seg, ch_seg]
    titles = ['Original Angiography', f'Otsu ({otsu_thresh:.2f})', f'CH with Otsu γ ({gamma:.2f})']
    main_title = "Fig 4 Replication: Angiography Image"
    save_path = os.path.join(results_dir, "fig4_angiography.png")
    save_comparison_plot(images, titles, main_title, save_path)