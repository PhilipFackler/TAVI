import functools
from concurrent.futures import ProcessPoolExecutor
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numba import cuda
import cupy
from math import ceil
cupy.cuda.runtime.setDevice(1)

import os
from os.path import dirname, realpath
use_rocm = os.getenv('ROCM_PATH') is not None

import numba
if use_rocm:
    from numba import hip
    hip.pose_as_cuda()

from numba import cuda

if use_rocm:
    from math import cos, exp, pi, sin, sqrt

    @cuda.jit(device=True)
    def rsqrt(x):
        return 1 / sqrt(x)

    @cuda.jit(device=True)
    def sincospi(x):
        return (sin(pi*x), cos(pi*x))

else:
    from numba.cuda.libdevice import cos, exp, rsqrt, sin, sincospi, sqrt

from phonopy import load

import phonopyacc

class model_info:
    def __init__(self):
        filename = dirname(realpath(__file__)) + "/phonopy.yaml"
        self.mesh = [11, 11, 11]
        self.intens_data = phonopyacc.IntensitiesData(filename, self.mesh)
        self.temperature = 300
        self.cutoff = 8e-2

def init_model():
    global data
    data = model_info()

# -------------------------------------------------------
# user input model_disp and model_inten
# -------------------------------------------------------
def model_disp(vq1, vq2, vq3, data):
    """return energy for given Q points
    """
    Q_prim = cupy.dot(cupy.column_stack([vq1,vq2,vq3]),
        cupy.asarray(data.intens_data.phonon.primitive_matrix))
    freqs = data.intens_data.calc_run_qpoints_frequencies(Q_prim)
    return cupy.transpose(cupy.asarray(freqs))


def model_inten(vq1, vq2, vq3, data):
    """return intensity for given Q points
    """
    Q_prim = cupy.dot(cupy.column_stack([vq1,vq2,vq3]),
        cupy.asarray(data.intens_data.phonon.primitive_matrix))
    _, dsf = data.intens_data.calc_intensities(Q_prim, data.temperature, data.cutoff)
    return cupy.transpose(dsf)


# -------------------------------------------------------
# fake resolution matrix and resolution ellipses
# -------------------------------------------------------
def rotation_matrix_4d(theta_deg):
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0, 0, -s],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [s, 0, 0, c],
        ]
    )


def resolution_matrix(hkl, en):
    """Fake resoltuion matrix mat and prefactor r0
    r0 is a constant, rez_mat is a symmetric positive 4 by 4 matrix
    """

    sigma1, sigma2 = 0.3, 0.02
    sigma3 = sigma4 = 0.2
    angle = -80
    mat = np.diag([1 / sigma1**2, 1 / sigma3**2, 1 / sigma4**2, 1 / sigma2**2])

    rot = rotation_matrix_4d(angle)
    rez_mat = rot.T @ mat @ rot
    r0 = 1

    return tuple((hkl[i], en[j], r0, rez_mat) for i in range(np.shape(hkl)[0]) for j in range(np.size(en)))


def plot_rez_ellipses(ax):
    sigma1, sigma2 = 0.3, 0.02
    angle = 80
    for i in range(3):
        ax.add_artist(
            Ellipse(
                xy=(2, 0),
                width=sigma1 * 2 * (i + 1),
                height=sigma2 * 2 * (i + 1),
                angle=angle,
                edgecolor="w",
                facecolor="none",
                label=f"{i + 1}-sigma",
            )
        )


# -------------------------------------------------------
# functions required for resolution convolution
# -------------------------------------------------------


def quadric_proj(quadric: np.ndarray, idx: int) -> np.ndarray:
    """projects along one axis of the quadric

    dimensino of input arry is n by n
    dimensiont of output array is (n-1) by (n-1)
    """

    # delete if orthogonal
    if np.abs(qii := quadric[idx, idx]) < 1e-8:
        mask = np.arange(quadric.shape[0]) != idx
        return quadric[np.ix_(mask, mask)]

    # row/column along which to perform the orthogonal projection
    # symmetrise if not symmetric, normalise to indexed component
    vec = 0.5 * (quadric[idx, :] + quadric[:, idx]) / np.sqrt(qii)
    ortho_proj = quadric - np.outer(vec, vec)  # projected quadric

    # return np.delete(np.delete(ortho_proj, idx, axis=0), idx, axis=1)
    mask = np.arange(ortho_proj.shape[0]) != idx
    return ortho_proj[np.ix_(mask, mask)]


def incoh_sigma_en(mat: np.ndarray) -> float:
    """Incoherent sigma for energy"""

    elem = quadric_proj(quadric_proj(quadric_proj(mat, 2), 1), 0)[0, 0]

    return 1 / np.sqrt(np.abs(elem))


def incoh_sigma_qs(mat: np.ndarray) -> tuple[float, float, float]:
    """Incoherent sigmas for q1, q2 and q3"""

    mat_2 = quadric_proj(mat, 2)
    elem1 = abs(quadric_proj(mat_2, 1)[0, 0])
    elem2 = abs(quadric_proj(mat_2, 0)[0, 0])
    elem3 = abs(quadric_proj(quadric_proj(mat, 1), 0)[0, 0])

    return (1 / np.sqrt(elem1), 1 / np.sqrt(elem2), 1 / np.sqrt(elem3))


def coh_sigma(mat: np.ndarray, axis: int):
    """Coherent sigma along a given axis"""
    idx = int(axis)

    return 1 / np.sqrt(np.abs(mat[idx, idx]))

@cuda.jit
def _compute_weights_kernel(vqe, mat, wt):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= vqe.shape[2]:
        return

    for b in range(vqe.shape[1]):
        v = vqe[:, b, i]
        tmp = 0.0
        for k in range(4):
            for l in range(4):
                tmp += v[k] * mat[k, l] * v[l]
        wt[b, i] = tmp

    return

def compute_weights(vqe: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """calculate weight
    vqe has shape (4, num_bands, num_pts)
    mat has shape (4, 4)
    weights = np.einsum("ijk,il,ljk->jk", vqe, mat_qe, vqe)
    """
    mat_d = cupy.asarray(mat)
    weights = cupy.einsum("ijk,il,ljk->jk", vqe, mat_d, vqe)

    return weights


# @functools.cache
# def generate_meshgrid(num_of_sigmas=3, num_pts=(10, 10, 10)):
#     pts_qh, pts_qk, pts_ql = num_pts
#     qh = cupy.linspace(-num_of_sigmas, num_of_sigmas, pts_qh + 1)
#     qk = cupy.linspace(-num_of_sigmas, num_of_sigmas, pts_qk + 1)
#     ql = cupy.linspace(-num_of_sigmas, num_of_sigmas, pts_ql + 1)
#     return cupy.meshgrid(qh, qk, ql, indexing="ij")  # shape (3, N1, N2, N3)


# def generate_pts(sigma_qs, mat_hkl, num_of_sigmas=3, num_pts=(10, 10, 10)):
#     """Generate points in a 3D mesh, cut the points at the corners"""
#     (sigma_qh_incoh, sigma_qk_incoh, sigma_ql_incoh) = sigma_qs

#     vq_h, vq_k, vq_l = generate_meshgrid(num_of_sigmas, num_pts)
#     vq = cupy.ndarray((3, vq_h.shape[0], vq_h.shape[1], vq_h.shape[2]))
#     vq[0] = vq_h * sigma_qh_incoh
#     vq[1] = vq_k * sigma_qk_incoh
#     vq[2] = vq_l * sigma_ql_incoh

#     # -------- cut the corners based on distance --------
#     r_sq = cupy.einsum("i...,ij,j...->...", vq, cupy.asarray(mat_hkl), vq)
#     idx = r_sq < num_of_sigmas**2  # Ellipsoid mask
#     return (vq[0][idx], vq[1][idx], vq[2][idx]), idx

@cuda.jit
def _gen_pts_kernel(vq, n_sigmas, Nh, Nk, Nl, s_qh, s_qk, s_ql):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if i >= Nh or j >= Nk or k >= Nl:
        return

    a = -n_sigmas
    b = n_sigmas
    dh = (b - a) / (Nh - 1)
    dk = (b - a) / (Nk - 1)
    dl = (b - a) / (Nl - 1)
    vq[0,i,j,k] = (i*dh + a) * s_qh
    vq[1,i,j,k] = (j*dk + a) * s_qk
    vq[2,i,j,k] = (k*dl + a) * s_ql

    return

def generate_pts(sigma_qs, mat_hkl, n_sigmas, num_pts):
    """Generate points in a 3D mesh, cut the points at the corners"""
    L, M, N = (num_pts[0] + 1, num_pts[1] + 1, num_pts[2] + 1)
    vq = cupy.array([cupy.empty((L, M, N)), cupy.empty((L, M, N)),
        cupy.empty((L, M, N))])
    threads = (8, 8, 8)
    blocks = (ceil(L / 8), ceil(M / 8), ceil(N / 8))
    s_qh, s_qk, s_ql = sigma_qs
    _gen_pts_kernel[blocks, threads](vq, n_sigmas, L, M, N, s_qh, s_qk, s_ql)

    # -------- cut the corners based on distance --------
    r_sq = cupy.einsum("i...,ij,j...->...", vq, cupy.asarray(mat_hkl), vq)
    idx = r_sq < n_sigmas**2  # Ellipsoid mask
    return (vq[0][idx], vq[1][idx], vq[2][idx]), idx

def get_max_step(arr, axis: int):
    """Get max step along a given axis. Return zero if all NaN"""
    # shape of arr (num_bands, N1, N2, N3)
    diff_arr = np.abs(np.diff(arr, axis=axis))
    steps = np.nanmean(diff_arr, axis=axis)
    if np.isnan(steps).all():
        return 0.0

    return float(np.nanmax(steps))


def convolution(reso_params, data, energy_rez_factor=1 / 5, max_step=100):
    """Perform the convolution
    The maxium sampling box size in Q is (max_step, max_step ,max_step)

    Note:
        Increase the accuracy by decresing energy_rez_factor and incresing max_step
    """
    # ----------------------------------------------------
    # return np.nan if repo_params is None
    # ----------------------------------------------------
    if reso_params is None:
        return np.nan
    # ----------------------------------------------------
    # calculate resolution matrix for all points
    # ----------------------------------------------------
    (qh, qk, ql), en, r0, mat = reso_params
    print(f"Calculating (Q1, Q2, Q3, E) = ({qh:.2f}, {qk:.2f}, {ql:.2f}, {en:.2f})")
    mat_hkl = quadric_proj(mat, 3)
    # ----------------------------------------------------
    # calculate the incoherent sigmas for all Q and E directions
    # ----------------------------------------------------
    sigma_qs = incoh_sigma_qs(mat_hkl)
    sigma_en_incoh = incoh_sigma_en(mat)
    num_of_sigmas = 3
    min_en, max_en = en - num_of_sigmas * sigma_en_incoh, en + num_of_sigmas * sigma_en_incoh
    sigma_en_coh = coh_sigma(mat, 3)
    # define the energy resolution to be 1/5 of the coherent sigma in energy
    en_rez = sigma_en_coh * energy_rez_factor
    # ----------------------------------------------------
    # Calculate elemental volume
    # ----------------------------------------------------
    eigenvalues = np.linalg.eigvalsh(mat_hkl)
    eval_inv_sqrt = 1 / np.sqrt(eigenvalues)
    elem_vols = np.prod(eval_inv_sqrt) * (2 * num_of_sigmas) ** 3
    # ----------------------------------------------------
    # First round, coarse grid
    # ----------------------------------------------------
    pts = [10, 10, 10]
    (vqh, vqk, vql), idx = generate_pts(sigma_qs, mat_hkl, num_of_sigmas, tuple(pts))
    disp = model_disp(vqh + qh, vqk + qk, vql + ql, data)
    num_bands, num_pts = disp.shape

    # Retrun zero if all dispersion is outside the relevant energy window
    if np.max(disp) < min_en or np.min(disp) > max_en:
        return 0.0
    # ----------------------------------------------------
    # determine if sampled enough based on steps along energy
    # ----------------------------------------------------
    vq = cupy.array((vqh, vqk, vql))  # shape: (3, num_pts)
    vqe = cupy.empty((4, num_bands, num_pts))
    vqe[0:3] = vq[:, None, :]
    vqe[3] = disp - en
    weights = compute_weights(vqe, mat)  # shape: (num_bands, num_pts)
    # Return zero if everything is outside the 5-sigma volume
    if cupy.min(weights) > 5**3:
        return 0.0

    # ----------------------------------------------------
    # determine Q steps based on energy steps
    # ----------------------------------------------------
    disp_arr = np.full(shape=(num_bands,) + idx.shape, fill_value=np.nan)
    disp_h = cupy.asnumpy(disp)
    idx_h = cupy.asnumpy(idx)
    disp_arr[(slice(None),) + np.nonzero(idx_h)] = disp_h
    # Compute max energy steps
    steps = [get_max_step(disp_arr, axis=i) for i in (1, 2, 3)]
    # limit the maximum in case the dispersion is too steep
    for i, (step, pt) in enumerate(zip(steps, pts)):
        if step > en_rez:
            factor = step / en_rez
            pts[i] = int(np.min((pt * factor, max_step)))

    # ----------------------------------------------------
    # Enough sampled. Calculate weight from resolution function
    # ----------------------------------------------------
    (vqh, vqk, vql), idx = generate_pts(sigma_qs, mat_hkl, num_of_sigmas, tuple(pts))
    disp = model_disp(vqh + qh, vqk + qk, vql + ql, data)
    _, num_pts = disp.shape

    vq = cupy.array((vqh, vqk, vql))  # shape: (3, num_pts)
    vqe = cupy.empty((4, num_bands, num_pts))
    vqe[0:3] = vq[:, None, :]
    vqe[3] = disp - en

    weights = compute_weights(vqe, mat)  # shape: (num_bands, num_pts)
    # ----------------------------------------------------
    # Keep only the points within the 4D ellipsoid
    # ----------------------------------------------------
    idx_keep = np.any(weights < 5**3, axis=0)
    vq_filtered = vq[:, idx_keep]
    num_pts_keep = np.count_nonzero(idx_keep)
    percent_kep = num_pts_keep / np.prod(pts) * 100
    print(f"Number of pts inside the ellipsoid = {num_pts_keep}, percentage ={percent_kep:.3f}%")

    weights_filtered = cupy.exp(-weights[:, idx_keep] / 2)
    inten = model_inten(*vq_filtered, data)
    # normalization by elementary volume size
    elem_vols /= np.prod(pts)
    det = np.linalg.det(mat)
    inten_sum = cupy.sum(inten * weights_filtered) * elem_vols
    return cupy.asnumpy(r0 * inten_sum * np.sqrt(det) / (2 * np.pi) ** 2)


if __name__ == "__main__":
    # ----------------------------------------------------
    # points being measured
    # qe_mesh has the dimension (4, n_pts_of_measurement)
    # flatten for meshed measurement
    # ----------------------------------------------------
    q1_min, q1_max, q1_step = 2, 2.02, 0.02
    en_min, en_max, en_step = 1, 2, 1.0
    q2 = 0
    q3 = 0

    q1 = np.linspace(q1_min, q1_max, int((q1_max - q1_min) / q1_step) + 1)
    en = np.linspace(en_min, en_max, int((en_max - en_min) / en_step) + 1)

    # calculate resolution
    vq1, vq2, vq3 = np.meshgrid(q1, q2, q3, indexing="ij")
    q_list = np.stack((vq1.ravel(), vq2.ravel(), vq3.ravel()), axis=-1)
    reso_params = resolution_matrix(hkl=q_list, en=en)

    #data = model_info()

    t0 = time()
    # ------------------- multiprocessing ------------------
    #num_worker = 8
    #with ProcessPoolExecutor(max_workers=num_worker, initializer=init_model) as executor:
    #    results = executor.map(convolution, reso_params)
    #measurement_inten = np.asarray(list(results))
    # ------------------- single core ------------------
    data = model_info()
    sz = len(reso_params)
    measurement_inten = np.empty(shape=sz)
    for i in range(sz):
        measurement_inten[i] = convolution(reso_params[i], data)
    # --------------------------------------------------

    print(f"Convolution completed in {(t1 := time()) - t0:.4f} s")
    # total intensity should be close to S/2 *(q1_max - q1_min) * 2p*i
    total_intent = np.sum(measurement_inten) * q1_step * en_step / (q1_max - q1_min)

    # ----------------------------------------------------
    # plot 2D contour
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    vq1, ven = np.meshgrid(q1, en, indexing="ij")
    img = ax.pcolormesh(vq1, ven, measurement_inten.reshape(np.shape(vq1)), cmap="turbo", vmin=0, vmax=0.5)

    ax.grid(alpha=0.6)
    ax.set_xlabel("Q1")
    ax.set_ylabel("En")
    ax.set_xlim((q1_min, q1_max))
    ax.set_ylim((en_min, en_max))

    plot_rez_ellipses(ax)
    disp = cupy.asnumpy(model_disp(cupy.asarray(q1), cupy.zeros_like(q1),
        cupy.zeros_like(q1), data))
    for i in range(np.shape(disp)[0]):
        ax.plot(q1, disp[i], "-w")

    ax.legend()
    fig.colorbar(img, ax=ax)
    ax.set_title(
        f"1D FM chain S=1 J=-5, total intensity = {total_intent:.3f}"
        + f"\n3D Convolution for {len(q1) * len(en)} points, "
        + f"completed in {t1 - t0:.3f} s"
        # + " with {num_worker:1d} cores"
    )

    plt.show()
