import numpy as np
import h5py as h5
########## =================================================
import camb
from camb import model
########## =================================================
import pyccl
from pyccl.tracers import NumberCountsTracer, WeakLensingTracer
from pyccl.cls import angular_cl
########## =================================================


def get_sigma8(Omega_m, A_s, h=0.7, ns=0.97, Omega_b=0.046):
  pars = camb.CAMBparams()
  pars.set_cosmology(H0=100 * h,
                     ombh2=Omega_b * h * h,
                     omch2=(Omega_m - Omega_b) * h * h)
  pars.InitPower.set_params(As=A_s, ns=ns)
  #Note non-linear corrections couples to smaller scales than you want
  pars.set_matter_power(redshifts=[0.], kmax=2.0)
  #Linear spectra
  pars.NonLinear = model.NonLinear_none
  results = camb.get_results(pars)
  sig8 = np.array(results.get_sigma8())
  return sig8


def get_tracer(cosmo, probe, z, n_z):
  dn_dz = (z, n_z)
  if (probe == 'source'):
    tracer = WeakLensingTracer(cosmo, dndz=dn_dz)
  elif (probe == 'lens'):
    b_z = (z, np.ones(z.shape))
    tracer = NumberCountsTracer(cosmo, has_rsd=False, dndz=dn_dz, bias=b_z)
  return tracer


def get_spectra(N_Z_BINS,
                zs,
                n_zs,
                probe_list,
                Omega_m,
                A_s,
                h=0.7,
                ns=0.97,
                Omega_b=0.046):
  sigma8 = get_sigma8(Omega_m, A_s)[0]
  cosmo = pyccl.Cosmology(Omega_c=Omega_m - Omega_b,
                          Omega_b=Omega_b,
                          h=h,
                          n_s=ns,
                          sigma8=sigma8,
                          transfer_function='bbks')

  print(probe_list)
  ell_arr = np.arange(8100)

  Cl_arr = np.zeros((N_Z_BINS, N_Z_BINS, len(ell_arr)))
  for i in range(N_Z_BINS):
    for j in range(i + 1):
      probe1 = probe_list[i]
      probe2 = probe_list[j]

      z1 = zs[i]
      n_z1 = n_zs[i]
      dn_dz1 = (z1, n_z1)
      z2 = zs[j]
      n_z2 = n_zs[j]
      dn_dz2 = (z2, n_z2)

      tracer1 = get_tracer(cosmo, probe1, z1, n_z1)
      tracer2 = get_tracer(cosmo, probe2, z2, n_z2)

      cl = angular_cl(cosmo, tracer1, tracer2, ell_arr)

      Cl_arr[i, j] = cl
      Cl_arr[j, i] = cl
  Cl_arr[:, :, 0] = 1e-25 * np.eye(N_Z_BINS)
  Cl_arr[:, :, 1] = 1e-25 * np.eye(N_Z_BINS)

  return Cl_arr


def get_lognormal_precalculated(precalculated_file):
  with h5.File(precalculated_file, 'r') as f:
    Pl_theta = f['Pl_theta'][:]
    w_i = f['w_i'][:]
    ls = f['ls'][:]
    lmax = f['lmax'][()]
  return [Pl_theta, w_i, ls, lmax]
