# src/ch-model.py
import numpy as np
import math

def solve_cahn_hilliard_fourier(u0_orig: np.ndarray, gamma,
                                eps1, lambda_val1, C1_factor1, C2_factor1, dt1, itmax1,
                                eps2, lambda_val2, C1_factor2, C2_factor2, dt2, itmax2,
                                dx=1.0, dy=1.0):
    """
    Solves the modified Cahn-Hilliard equation using a two-stage Fourier Spectral method.
    Returns the final continuous phase field `u` after stage 2.
    """
    u0 = u0_orig.copy()
    m, n = u0.shape
    u = u0.copy()

    # Create Fourier space Laplacian operator
    freq_m = np.fft.fftfreq(m, d=dx)
    freq_n = np.fft.fftfreq(n, d=dy)
    kx = 2 * np.pi * freq_m
    ky = 2 * np.pi * freq_n
    Kx, Ky = np.meshgrid(ky, kx)
    K_sq = Kx**2 + Ky**2
    M_operator = -K_sq
    M_sq_operator = K_sq**2

    # --- Stage 1 ---
    print(f"Running Stage 1: eps={eps1}, lambda={lambda_val1}, dt={dt1}, itmax={itmax1}")
    C1_s1 = C1_factor1 / eps1 if eps1 > 0 else 0
    C2_s1 = C2_factor1 * lambda_val1
    u_hat = np.fft.fft2(u)
    u0_hat = np.fft.fft2(u0)
    denominator_s1 = 1 + dt1 * (eps1 * M_sq_operator - C1_s1 * M_operator + C2_s1)

    for i in range(itmax1):
        F_u = 4 * u * (u - 1) * (u - gamma)
        F_u_hat = np.fft.fft2(F_u)
        
        term1_num_s1 = (1 - dt1 * C1_s1 * M_operator + C2_s1 * dt1) * u_hat
        term2_num_s1 = (dt1 / eps1) * M_operator * F_u_hat if eps1 > 0 else 0
        # Use adjusted lambda here too
        term3_num_s1 = dt1 * lambda_val1 * (u0_hat - u_hat)

        numerator_hat_s1 = term1_num_s1 + term2_num_s1 + term3_num_s1
        u_hat = numerator_hat_s1 / denominator_s1
        u = np.real(np.fft.ifft2(u_hat))
        u = np.clip(u, -0.5, 1.5)

    u_stage1 = u.copy()

    # --- Stage 2 ---
    print(f"Running Stage 2: eps={eps2}, lambda={lambda_val2}, dt={dt2}, itmax={itmax2}")
    C1_s2 = C1_factor2 / eps2 if eps2 > 0 else 0
    C2_s2 = C2_factor2 * lambda_val2
    u = u_stage1.copy()
    u_hat = np.fft.fft2(u)
    denominator_s2 = 1 + dt2 * (eps2 * M_sq_operator - C1_s2 * M_operator + C2_s2)

    for i in range(itmax2):
        F_u = 4 * u * (u - 1) * (u - gamma)
        F_u_hat = np.fft.fft2(F_u)
        
        term1_num_s2 = (1 - dt2 * C1_s2 * M_operator + C2_s2 * dt2) * u_hat
        term2_num_s2 = (dt2 / eps2) * M_operator * F_u_hat if eps2 > 0 else 0
        term3_num_s2 = dt2 * lambda_val2 * (u0_hat - u_hat) # This should be 0 if lambda_val2=0

        numerator_hat_s2 = term1_num_s2 + term2_num_s2 + term3_num_s2
        u_hat = numerator_hat_s2 / denominator_s2
        u = np.real(np.fft.ifft2(u_hat))
        u = np.clip(u, -0.5, 1.5)

    u_final = np.clip(u, 0, 1) # Final clip
    return u_final