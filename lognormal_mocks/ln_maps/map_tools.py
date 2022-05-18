# This module contains useful operations done on the flat-sky maps

import numpy as np
import jax.numpy as jnp
import jax
from jax.config import config
from jax import jit
from functools import partial
config.update("jax_enable_x64", True)

J = np.complex(0., 1.)
EPS = 1e-20

class MapTools:
    def __init__(self, N_grid, theta_max):
        self.set_map_properties(N_grid, theta_max)
        self.set_fft_properties(N_grid, self.theta_max)
        self.imag_indices = self.get_imag_indices()
    
    @partial(jit, static_argnums=(0,))
    def do_fwd_KS(self, kappa_l):
        kappa_l = self.symmetrize_fourier(kappa_l)
        kappa_l_complex = kappa_l[0] + J * kappa_l[1] 

        F_gamma_1 = (self.ell_x**2 - self.ell_y**2) * kappa_l_complex / (self.ell**2 + EPS)
        F_gamma_2 = 2. * self.ell_x * self.ell_y    * kappa_l_complex / (self.ell**2 + EPS)
        
        gamma_1 =  jnp.fft.irfftn(F_gamma_1) / self.PIX_AREA
        gamma_2 =  jnp.fft.irfftn(F_gamma_2) / self.PIX_AREA
        
        return gamma_1, gamma_2    
    
    def do_KS_inversion(self, eps):
        
        A_ell = ((self.ell_x_full**2 - self.ell_y_full**2) - J * (2 * self.ell_x_full * self.ell_y_full)) \
                                            /(self.ell_full**2 + EPS)
        
        eps_1, eps_2 = eps
        eps_ell = self.PIX_AREA * jnp.fft.fftn(eps_1 + J * eps_2)
        kappa_ell = A_ell * eps_ell
        kappa_map_KS = jnp.fft.ifftn(kappa_ell).real /  self.PIX_AREA
        return kappa_map_KS
    
    @partial(jit, static_argnums=(0,))
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * jnp.fft.rfftn(x_map)
        return jnp.array([Fx_complex.real, Fx_complex.imag])
    
    @partial(jit, static_argnums=(0,))
    def fourier2map(self, Fx, conjugate=False):
        Fx         = self.symmetrize_fourier(Fx)
        Fx_complex = Fx[0] + J * Fx[1]
        if(conjugate):
            Fx_complex = jnp.conj(Fx_complex)
        return jnp.fft.irfftn(Fx_complex) /  self.PIX_AREA
    
    @partial(jit, static_argnums=(0,))
    def symmetrize_fourier(self, Fx):
        return jnp.array([(self.fourier_symm_mask * Fx[0]) + 
                          (~self.fourier_symm_mask * 
                           jnp.take(Fx[0], self.fourier_symm_flip_ind, axis=0))
                         ,(self.fourier_symm_mask * Fx[1]) - 
                          (~self.fourier_symm_mask * 
                           jnp.take(Fx[1], self.fourier_symm_flip_ind, axis=0))])

    def set_map_properties(self, N_grid, theta_max):
        self.N_grid     = N_grid
        self.theta_max  = theta_max * np.pi / 180.   # theta_max in radians
        self.Omega_s    = self.theta_max**2          # Area of the map        
        
        self.PIX_AREA = self.Omega_s / self.N_grid**2 # Pixel area in radian^2
        
    def set_fft_properties(self, N_grid, theta_max):
        lx = 2*np.pi*np.fft.fftfreq(N_grid, d=theta_max / N_grid)
        ly = 2*np.pi*np.fft.fftfreq(N_grid, d=theta_max / N_grid)

        N_Y = (N_grid//2 +1)
        self.N_Y = N_Y
        
        # mesh of the 2D frequencies
        self.ell_x = np.tile(lx[:, None], (1, N_Y))       
        self.ell_y = np.tile(ly[None, :N_Y], (N_grid, 1))
        self.ell = np.sqrt(self.ell_x**2 + self.ell_y**2)
        
        self.ell_x_full = np.tile(lx[:, None], (1, N_grid))       
        self.ell_y_full = np.tile(ly[None, :], (N_grid, 1))
        self.ell_full   = np.sqrt(self.ell_x_full**2 + self.ell_y_full**2)
        
        fourier_symm_mask = np.ones((N_grid, self.N_Y))
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask.astype(bool)        
        
        fourier_symm_mask_imag = fourier_symm_mask.copy()
        fourier_symm_mask_imag[0,-1]        = 0
        fourier_symm_mask_imag[self.N_Y-1,0]  = 0
        fourier_symm_mask_imag[self.N_Y-1,-1] = 0
        self.fourier_symm_mask_imag = fourier_symm_mask_imag.astype(bool)
        
        fourier_symm_flip_ind      = np.arange(N_grid)
        fourier_symm_flip_ind[1:]  = fourier_symm_flip_ind[1:][::-1]
        self.fourier_symm_flip_ind = fourier_symm_flip_ind
        
# ================== 1D array to Fourier plane ===============================             
    def set_fourier_plane_face(self, F_x, x):
        N = self.N_grid
        F_x = jax.ops.index_update(F_x, jax.ops.index[:,:,1:-1], x[:N**2 - 2*N].reshape(2,N,N//2-1))
        return F_x

    def set_fourier_plane_edge(self, F_x, x):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    
        F_x = jax.ops.index_update(F_x, jax.ops.index[:,1:N_Y-1,0],  x[N**2 - 2*N:N**2 - 2*N+2*N_edge].reshape((2,-1)))
        F_x = jax.ops.index_update(F_x, jax.ops.index[:,1:N_Y-1,-1], x[N**2 - 2*N+2*N_edge:-3].reshape((2,-1)))
        return F_x

    def set_fourier_plane_corner(self, F_x, x):    
        N = self.N_grid
        N_Y = N//2+1
               
        F_x = jax.ops.index_update(F_x, jax.ops.index[0,N_Y-1,-1] , x[-3])
        F_x = jax.ops.index_update(F_x, jax.ops.index[0,0,-1]    , x[-2])
        F_x = jax.ops.index_update(F_x, jax.ops.index[0,N_Y-1,0] , x[-1])
        return F_x
    
    @partial(jit, static_argnums=(0,))
    def array2fourier_plane(self, x):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    

        F_x_plane = jnp.zeros((2,N,N_Y))
        F_x_plane = self.set_fourier_plane_face(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_edge(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_corner(F_x_plane, x)

        F_x_plane = self.symmetrize_fourier(F_x_plane)        
        return F_x_plane
    
    def fourier_plane2array(self, Fx):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    

        x = jnp.zeros(shape=N*N-1)

        x = jax.ops.index_update(x, jax.ops.index[:N**2 - 2*N],                    Fx[:,:,1:-1].reshape(-1))
        x = jax.ops.index_update(x, jax.ops.index[N**2 - 2*N:N**2 - 2*N+2*N_edge], Fx[:,1:N_Y-1,0].reshape(-1))
        x = jax.ops.index_update(x, jax.ops.index[N**2 - 2*N+2*N_edge:-3],         Fx[:,1:N_Y-1,-1].reshape(-1))
        
        
        x = jax.ops.index_update(x, jax.ops.index[-3], Fx[0,N_Y-1,-1])
        x = jax.ops.index_update(x, jax.ops.index[-2], Fx[0,0,-1])        
        x = jax.ops.index_update(x, jax.ops.index[-1], Fx[0,N_Y-1,0])
        
        return x
    
    def get_imag_indices(self):
        x0 = np.zeros(self.N_grid**2-1)
        Fx = np.array(self.array2fourier_plane(x0))
        Fx[1] = 1
        Fx = jnp.array(Fx)
        x0 = np.array(self.fourier_plane2array(Fx)).astype(int)

        indices = np.arange(x0.shape[0])
        imag_indices_1d = indices[(x0 == 1)]

        return imag_indices_1d
    
# ================== POST-PROCESSING ===============================        
    def binned_Cl(self, kappa_ell_1, kappa_ell_2, ell_bins):
        """
        Calculates the binned (auto/cross)  power spectrum from the kappa_l's with a given l bins.
        """
        cross_Cl_bins = []
        ell_bin_centre = []
        for i in range(len(ell_bins) - 1):
            select_ell = (self.ell > ell_bins[i]) & (self.ell < ell_bins[i+1]) & self.fourier_symm_mask
            ell_bin_centre.append(np.mean(self.ell[select_ell]))
            # The factor of 2 needed because there are both real and imaginary modes in the l selection!
            cross_Cl = 2. * np.mean(kappa_ell_1[:,select_ell] * kappa_ell_2[:,select_ell])/self.Omega_s
            cross_Cl_bins.append(cross_Cl)
        return np.array(ell_bin_centre), np.array(cross_Cl_bins)
    
    def binned_cross_corr(self, kappa_ell_1, kappa_ell_2, ell_bins):
        _, Cl_12 = self.binned_Cl(kappa_ell_1, kappa_ell_2, ell_bins)
        _, Cl_11 = self.binned_Cl(kappa_ell_1, kappa_ell_1, ell_bins)
        _, Cl_22 = self.binned_Cl(kappa_ell_2, kappa_ell_2, ell_bins)
        return Cl_12 / np.sqrt(Cl_11 * Cl_22)
    
    def get_camb_Cl_average(self, ell_bins, Cl_interp):
        camb_Cl = []
        for i in range(len(ell_bins) - 1):
            select_ell = (self.ell > ell_bins[i]) & (self.ell < ell_bins[i+1]) & self.fourier_symm_mask
            Cl_average = np.mean(Cl_interp(self.ell[select_ell]))
            camb_Cl.append(Cl_average)
        return np.array(camb_Cl)
    
    def kappal2map(self, kappa_l):
        """
        Get the real space kappa maps from kappa_l. Returns a list of NxN map in each redshift bins.
        """
        N_Z_BINS = kappa_l.shape[0]
        kappa_map_list = []
        for n in range(N_Z_BINS):
            kappa_l_bin = kappa_l[n]
            kappa_map = self.fourier2map(kappa_l_bin)
            kappa_map_list.append(kappa_map)
        return kappa_map_list