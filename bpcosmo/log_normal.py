import jax
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d
from jax.config import config

config.update("jax_enable_x64", True)
from jax import jit
from functools import partial
import time

from .map_tools import MapTools

EPS = 1e-20
J = np.complex(0., 1.)


class FlatSkyMap:
  def __init__(self, N_Z_BINS, N_grid, theta_max, n_z, probe_list, cosmo_pars,
               shifts, var_gauss, precalculated):
    """
        :N_Z_BINS:      Number of redshift bins
        :N_grid:        Number of pixels on each side. At the moment, we assume square geometry
        :theta_max:     Each side of the square (in degrees)
        :n_z:           n(z) for each redshift bin. Provided as a list of z and n(z) in each redshift bin
        :cosmo_pars:    Cosmological parameters (Omega_m, A_s)
        :shifts:        
        :precalculated:
        """
    self.set_map_properties(N_Z_BINS, N_grid, theta_max)

    self.map_tool = MapTools(N_grid, theta_max)
    self.N_Y = self.map_tool.N_Y

    self.OmegaM_fid, self.As_fid, h, ns, Omega_b = cosmo_pars
    self.zs, self.n_zs = n_z
    self.probe_list = probe_list

    # TODO: replace this function
    Cl = get_spectra(N_Z_BINS,
                     self.zs,
                     self.n_zs,
                     self.probe_list,
                     self.OmegaM_fid,
                     self.As_fid,
                     h=h,
                     ns=ns,
                     Omega_b=Omega_b)

    self.Cl = Cl
    self.set_Cl_arr(Cl)

    self.Pl_theta, self.w_i, self.ls, self.lmax = precalculated
    self.shifts = shifts
    self.var_gauss = var_gauss
    self.pixel_window = self.pixel_window_func(self.ls)

    self.Cl = self.Cl[:, :, :self.lmax]
    self.Cl_g = self.cl2clg(self.Cl)
    self.set_Cl_arr(self.Cl_g)
    self.set_eigs(self.Cl_arr_real, self.Cl_arr_imag)

  def cl2clg(self, Cl):
    Cl_g = np.zeros((self.N_Z_BINS, self.N_Z_BINS, self.lmax))
    for i in range(self.N_Z_BINS):
      for j in range(i + 1):
        Cl_g[i, j] = self._cl2clg_single_bin(Cl[i, j], self.shifts[i],
                                             self.shifts[j])
        if (i != j):
          Cl_g[j, i] = Cl_g[i, j]
    Cl_g[:, :, 0] = 1e-15 * np.eye(self.N_Z_BINS)
    Cl_g[:, :, 1] = 1e-15 * np.eye(self.N_Z_BINS)
    return Cl_g

  def _cl2clg_single_bin(self, Cl, shift1, shift2=None):
    chi = 1. / (4. * np.pi) * np.sum(self.Pl_theta * Cl[:self.lmax] *
                                     (1 + 2. * self.ls[:self.lmax]),
                                     axis=1)
    if shift2 is None:
      shift2 = shift11
    chi_g = np.log(1. + chi / shift1 / shift2)
    Cl_g = (2. * np.pi) * np.sum(
        ((self.w_i * chi_g).reshape(-1, 1) * self.Pl_theta), axis=0)
    Cl_g[:2] = 1e-15
    return Cl_g

  def pixel_window_func(self, ls, anisotropic=False):
    Delta_Theta = self.theta_max / self.N_grid
    filter_arr = np.sinc(0.5 * self.ls * Delta_Theta / np.pi)
    return filter_arr

  def get_Cl_arr(self, Cl, beta=1.):
    ls = np.arange(len(Cl[0, 0]))
    lmax = ls[-1]
    self.ls = ls

    Cl_arr = np.zeros((self.N_Z_BINS, self.N_Z_BINS, self.N_grid, self.N_Y))

    for i in range(self.N_Z_BINS):
      for j in range(self.N_Z_BINS):
        Cl_arr[i, j] = 0.5 * self.interp_arr(ls / beta, Cl[i, j])

    Cl_arr_real = Cl_arr.copy()
    Cl_arr_imag = Cl_arr.copy()

    Cl_arr_imag[:, :, 0, 0] = 1e-20 * np.diag(np.ones(self.N_Z_BINS))
    Cl_arr_imag[:, :, 0, -1] = 1e-20 * np.diag(np.ones(self.N_Z_BINS))
    Cl_arr_imag[:, :, self.N_grid // 2,
                0] = 1e-20 * np.diag(np.ones(self.N_Z_BINS))
    Cl_arr_imag[:, :, self.N_grid // 2,
                -1] = 1e-20 * np.diag(np.ones(self.N_Z_BINS))

    Cl_arr_real[:, :, 0, 0] = 2. * Cl_arr_real[:, :, 0, 0]
    Cl_arr_real[:, :, 0, -1] = 2. * Cl_arr_real[:, :, 0, -1]
    Cl_arr_real[:, :, self.N_grid // 2,
                0] = 2. * Cl_arr_real[:, :, self.N_grid // 2, 0]
    Cl_arr_real[:, :, self.N_grid // 2,
                -1] = 2. * Cl_arr_real[:, :, self.N_grid // 2, -1]

    Cl_arr_imag = Cl_arr_imag * self.Omega_s
    Cl_arr_real = Cl_arr_real * self.Omega_s

    inv_Cl_arr_real = np.linalg.inv(Cl_arr_real.T).T
    inv_Cl_arr_imag = np.linalg.inv(Cl_arr_imag.T).T

    return Cl_arr_real, Cl_arr_imag, inv_Cl_arr_real, inv_Cl_arr_imag

  def set_Cl_arr(self, Cl, beta=1.):
    Cl_arr_real, Cl_arr_imag, inv_Cl_arr_real, inv_Cl_arr_imag = self.get_Cl_arr(
        Cl, beta)

    self.Cl_arr_imag = Cl_arr_imag
    self.Cl_arr_real = Cl_arr_real

    self.inv_Cl_arr_real = inv_Cl_arr_real
    self.inv_Cl_arr_imag = inv_Cl_arr_imag

  def set_eigs(self, Cl_arr_real, Cl_arr_imag):
    self.eig_val_real, eig_vec_real = np.linalg.eig(Cl_arr_real.T)
    self.eig_val_imag, eig_vec_imag = np.linalg.eig(Cl_arr_imag.T)
    self.R_real = np.swapaxes(eig_vec_real.T, 0, 1)
    self.R_imag = np.swapaxes(eig_vec_imag.T, 0, 1)
    self.R_real = self.eig_vec_normalize(self.R_real)
    self.R_imag = self.eig_vec_normalize(self.R_imag)
    self.R_real_T = np.swapaxes(self.R_real, 0, 1)
    self.R_imag_T = np.swapaxes(self.R_imag, 0, 1)

  def eig_vec_normalize(self, x):
    sign_matrix = (x[0] + 1e-25) / np.abs(x[0] + 1e-25)
    sign_matrix = sign_matrix[np.newaxis]
    return sign_matrix * x

  def set_map_properties(self, N_Z_BINS, N_grid, theta_max):
    self.N_Z_BINS = N_Z_BINS
    self.N_grid = N_grid
    self.theta_max = theta_max * np.pi / 180.
    self.Omega_s = self.theta_max**2
    self.PIXEL_AREA = (theta_max * 60. / N_grid)**2

    arcminute_in_radian = (1. / 60) * np.pi / 180.
    self.arcminute_sq_in_radian = arcminute_in_radian**2

  def interp_arr(self, ls, y):
    interp_func = interp1d(ls, y)
    return interp_func(self.map_tool.ell)

#=============== WRAP to Fourier object ====================

  @partial(jit, static_argnums=(0, ))
  def wrap_fourier(self, x):
    x_fourier = jnp.zeros((self.N_Z_BINS, 2, self.N_grid, self.N_Y))
    for n in range(self.N_Z_BINS):
      y = self.map_tool.array2fourier_plane(x[n])
      x_fourier.at[n].set(y)
    return x_fourier

  @partial(jit, static_argnums=(0, ))
  def reverse_wrap_fourier(self, F_x):
    x = jnp.zeros((self.N_Z_BINS, self.N_grid * self.N_grid - 1))
    for n in range(self.N_Z_BINS):
      y = self.map_tool.fourier_plane2array(F_x[n])
      x.at[n].set(y)
    return x

  @partial(jit, static_argnums=(0, ))
  def matmul(self, A, x):
    y = jnp.zeros(x.shape)
    for i in range(self.N_Z_BINS):
      a = jnp.sum(A[i, :] * x, axis=0)
      y.at[i].add(a)
    return y

  def init_field(self, scaling=1.):
    x = np.random.normal(size=(self.N_Z_BINS, self.N_grid**2 - 1))
    return scaling * self.x2field(x)
#============================================================================================

  def x2field(self, x):
    x = self.wrap_fourier(x)
    fieldG_Fourier = self.x2fieldG_fourier(x)
    fieldG_map = self.fieldG_fourier2map(fieldG_Fourier)
    fields = self.fieldGmap_2_fourier(fieldG_map, self.shifts, self.var_gauss)
    return fields

  def x2fieldG_fourier(self, x):
    field_real = self.matmul(self.R_real,
                             x[:, 0] * jnp.sqrt(self.eig_val_real.T))
    field_imag = self.matmul(self.R_imag,
                             x[:, 1] * jnp.sqrt(self.eig_val_imag.T))
    return jnp.swapaxes(jnp.array([field_real, field_imag]), 0, 1)

  @partial(jit, static_argnums=(0, ))
  def fieldG_fourier2map(self, fieldG_Fourier):
    fieldG_map = jnp.zeros(
        (self.N_Z_BINS, self.map_tool.N_grid, self.map_tool.N_grid))
    for n in range(self.N_Z_BINS):
      fieldG_l = self.map_tool.symmetrize_fourier(fieldG_Fourier[n])
      fieldG_map.at[n].add(self.map_tool.fourier2map(fieldG_l))
    return fieldG_map

  @partial(jit, static_argnums=(0, ))
  def fieldGmap_2_fourier(self, fieldG_map, shifts, var_gauss):
    fields = jnp.zeros((self.N_Z_BINS, 2, self.N_grid, self.N_Y))
    for i in range(self.N_Z_BINS):
      field_map = shifts[i] * (jnp.exp(fieldG_map[i] - 0.5 * var_gauss[i]) -
                               1.)
      fields.at[i].add(self.map_tool.map2fourier(field_map))
    return fields


#============================================================================================
