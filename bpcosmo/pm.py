# This module contains utility functions to run a forward simulation
import jax
import jax.numpy as jnp

import jax_cosmo as jc

import numpyro
import numpyro.distributions as dist

from jax.experimental.ode import odeint
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.kernels import fftk
from jaxpm.lensing import density_plane

__all__ = [
    'get_density_planes',
]

def linear_field(mesh_shape, box_size, pk):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (box_size[0] * box_size[1] * box_size[2])

    field = numpyro.sample('initial_conditions',
                           dist.Normal(jnp.zeros(mesh_shape), 
                                       jnp.ones(mesh_shape)))

    field = jnp.fft.rfftn(field) * pkmesh**0.5
    field = jnp.fft.irfftn(field)
    return field

def get_density_planes(cosmology,
                   density_plane_width=100.,       # In Mpc/h
                   density_plane_npix=256,         # Number of pixels
                   density_plane_smoothing=3.,     # In Mpc/h
                   box_size=[400.,400.,4000.],     # In Mpc/h
                   nc=[64,64,640],
                   ):
  """Function that returns tomographic density planes 
  for a given cosmology from a lightcone.

  Args:
    cosmology: jax-cosmo object
    density_plane_width: width of the output density slices
    density_plane_npix: size of the output density slices
    density_plane_smoothing: Gaussian scale of plane smoothing
    box_size: [sx,sy,sz] size in Mpc/h of the simulation volume
    nc: number of particles/voxels in the PM scheme
  Returns:
    list of [r, a, plane], slices through the lightcone along with their
        comoving distance and scale factors.
  """
  # Planning out the scale factor stepping to extract desired lensplanes
  n_lens = int(box_size[-1] // density_plane_width)
  r = jnp.linspace(0., box_size[-1], n_lens+1)
  r_center = 0.5*(r[1:] + r[:-1])
  a_center = jc.background.a_of_chi(cosmology, r_center)

  # Create a small function to generate the matter power spectrum
  k = jnp.logspace(-4, 1, 128)
  pk = jc.power.linear_matter_power(cosmology, k)
  pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

  # Create initial conditions
  initial_conditions = linear_field(nc, box_size, pk_fn)

  # Create particles
  particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),axis=-1).reshape([-1,3])

  # Initial displacement
  cosmology._workspace = {} # FIX ME: this a temporary fix
  dx, p, f = lpt(cosmology, initial_conditions, particles, 0.01)

  # Evolve the simulation forward
  res = odeint(make_ode_fn(nc), [particles+dx, p], 
                jnp.concatenate([jnp.atleast_1d(0.01), a_center[::-1]]), 
                cosmology, rtol=1e-5, atol=1e-5)

  # Extract the lensplanes
  density_planes = []
  for i in range(len(a_center)):
      dx = box_size[0]/density_plane_npix
      dz = density_plane_width
      plane = density_plane(res[0][::-1][i],
                            nc,
                            (i+0.5)*density_plane_width/box_size[-1]*nc[-1],
                            width=density_plane_width/box_size[-1]*nc[-1],
                            plane_resolution=density_plane_npix,
                            smoothing_sigma=density_plane_smoothing/dx
                          )
      density_planes.append({'r':r_center[i], 
                         'a':a_center[i], 
                         'plane': plane, 
                         'dx': dx, 
                         'dz': dz})

  return density_planes
