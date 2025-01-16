################################################################################
#
# TPRF: Two-Particle Response Function (TPRF) Toolbox for TRIQS
#
# Copyright (C) 2023 by Hugo U. R. Strand
# Author: Hugo U. R. Strand
#
# TPRF is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TPRF is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TPRF. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import time
import numpy as np
import itertools

from h5 import HDFArchive

from triqs.gf import Gf, Fourier
from triqs.gf import make_gf_from_fourier

from triqs_tprf.lattice import lattice_dyson_g_wk

from triqs_tprf.rpa_tensor import kanamori_quartic_tensor

from triqs_tprf.bse import impurity_irreducible_vertex_Gamma
from triqs_tprf.dbse import solve_lattice_dbse_lomem
from triqs_tprf.dbse import impurity_reducible_vertex_F

from triqs_tprf.utilities import G2_loc_fixed_fermionic_window_python

from w2dyn_cthyb.converters_worm import p2_from_w2dyn_P2_worm_components
from w2dyn_cthyb.converters_worm import p2_remove_disconnected
from w2dyn_cthyb.converters_worm import p3_from_w2dyn_P3_worm_components
from w2dyn_cthyb.converters_worm import p3_w2dyn_to_triqs_freq_shift_alt
from w2dyn_cthyb.converters_worm import L_from_g3
from w2dyn_cthyb.converters_worm import g2_from_w2dyn_G2_worm_components

# DBSEP

# Implementation of dual Bethe-Salpeter equation for the polarization (DBSEP) according to [1] Krien, PHYSICAL REVIEW B 99, 235106 (2019)
# For the TPRF implementation of the dual Bethe-Salpeter equation for the susceptibility, see [2] van Loon and Strand, PHYSICAL REVIEW B 109, 155157 (2024)
# Erik van Loon, Lund University, 2025
# Work in progress

# A summary of the method is as follows:
# The dual Bethe-Salpeter equation (DBSE) is already more efficient than the BSE, since it uses Gdual=G-gloc instead of G,
# which decays as 1/nu^2 instead of 1/nu and therefore leads to a quickly decaying bubble and 1/nwf convergence of the DBSE,
# compared to 1/nwf for the ordinary BSE.
# The DBSE corresponds to a resummation of processes with local fermionic propagators, and this resummation leads to the desired replacement G->Gdual
# In the same spirit, a remaining bottleneck for the convergence is the fact that the vertex F in the DBSE is asymptotically constant.
# Krien's formula (here called DBSEP) is a resummation of the susceptibility in terms of processes with reducible local interactions. 
# It replaces F by Firr, where Firr has the property that it asymptotically decays instead of being constant. 
# Thus, all vertex corrections in the DBSEP converge with a higher power of 1/nwf than those in the DBSE
#
# Note: in the DBSEP, processes which don't involve Firr, only Lirr, also contribute asympotically, but they are computationally simpler. 
# The DBSEP has better formal convergence only if a those processes are taken into account using a sufficiently larger frequency box, which is not implemented here.
# Instead, the current implementation is expected to have the same power of nwf but better prefactor.

from triqs.gf import Gf, MeshProduct, Idx, MeshImFreq
from triqs_tprf.lattice import fourier_wk_to_wr, chi0r_from_gr_PH, chi0q_from_chi0r, chiq_sum_nu_from_chi0q_and_gamma_and_L_wn_PH

from triqs_tprf.linalg import product_PH, inverse_PH, identity_PH
from triqs_tprf.chi_from_gg2 import chi0_from_gg2_PH, chi_from_gg2_PH

from triqs_tprf.bse import get_chi0_nk_at_specific_w
from triqs_tprf.lattice_utils import add_fake_bosonic_mesh

import numpy as np

def impurity_polarization(chi_w,U_mat):
    # Construct desired impurity single-frequency objects

    # Rather painful, TPRF functions we want to use take wnn Gfs as input, so we need fake fermion grids
    # This should be improved in the future
    fake_fermion_mesh = MeshImFreq( beta=chi_w.mesh.beta, statistic='Fermion', n_iw=1)
    chi_wnn = Gf( mesh=MeshProduct( chi_w.mesh, fake_fermion_mesh, fake_fermion_mesh), target_shape=chi_w.target_shape )
    U_wnn = chi_wnn.copy()
    U_wnn.data[:] *= 0
    for w in U_wnn.mesh[0]:
        for n1 in U_wnn.mesh[1]:
            U_wnn[w,n1,n1] = U_mat
            chi_wnn[w,n1,n1] = chi_w[w]

    # Definition above Eq 8 in [1]:
    # chi = pi / (1 - U pi), so chi - chi U pi = pi. Due to geometric series, also chi - pi U chi = pi, sochi = pi (1+ U chi) and finally:
    # pi = chi / (1+U chi)
    # W  = U / (1 + U pi)
    pi_wnn = product_PH( chi_wnn, inverse_PH( identity_PH(chi_wnn)+product_PH(U_wnn, chi_wnn)  )  )
    W_wnn = product_PH( U_wnn, inverse_PH( identity_PH(pi_wnn)+product_PH(U_wnn, pi_wnn) ) )
    Upi_wnn = product_PH( U_wnn, pi_wnn)

    # Now, get back to single-frequency objects by removing the fake fermion grids
    pi_w = chi_w.copy()
    W_w = chi_w.copy()
    Upi_w = chi_w.copy()
    for w in pi_w.mesh:
        pi_w[w]  =  pi_wnn[w,Idx(0),Idx(0)]
        W_w[w]   =   W_wnn[w,Idx(0),Idx(0)]
        Upi_w[w] = Upi_wnn[w,Idx(0),Idx(0)]

    return pi_w, W_w, Upi_w

def pi_kw_to_chi_kw(pi_kw, U_mat):
    # Transform the lattice polarization to the lattice susceptibility

    chi_kw = pi_kw.copy()
    chi_kw.data[:] *= 0

    fake_fermion_mesh = MeshImFreq( beta=pi_kw.mesh[1].beta, statistic='Fermion', n_iw=1)

    for k in pi_kw.mesh[0]:
        pi_wnn = Gf( mesh=MeshProduct( pi_kw.mesh[1], fake_fermion_mesh, fake_fermion_mesh), target_shape=pi_kw.target_shape )
        U_wnn = pi_wnn.copy()
        U_wnn.data[:] *= 0
        for w in U_wnn.mesh[0]:
            for n1 in U_wnn.mesh[1]:
                U_wnn[w,n1,n1] = U_mat
                pi_wnn[w,n1,n1] = pi_kw[k,w]

        chi_wnn = product_PH( pi_wnn, inverse_PH( identity_PH(pi_wnn)-product_PH(U_wnn, pi_wnn) ) )

        for w in pi_kw.mesh[1]:
            chi_kw[k,w] = chi_wnn[w,Idx(0),Idx(0)]
    return chi_kw


def irreducible_L(L_wn, Upi_w):
    # Make irreducible vertex Lirr from reducible version L
    bmesh = L_wn.mesh[0]
    fmesh = L_wn.mesh[1]

    # Rather painful, TPRF functions take wnn Gfs as input, so we need to add an additional mesh
    L_wnn = Gf( mesh=MeshProduct( bmesh, fmesh, fmesh), target_shape=L_wn.target_shape )
    for w,n1 in L_wn.mesh:
        L_wnn[w,n1,n1] = L_wn[w,n1]
    Upi_wnn = L_wnn.copy()
    Upi_wnn.data[:] *= 0
    for w,n1 in L_wn.mesh:
        Upi_wnn[w,n1,n1] = Upi_w[w]


    # L = Lirr / (1-U pi), according to Eq 8 of [1]. Solving for Lirr gives
    # L - L U pi = Lirr
    Lirr_wnn = L_wnn - product_PH( L_wnn, Upi_wnn  )

    # Get rid of fake mesh
    Lirr_wn = L_wn.copy()
    for w,n in Lirr_wn.mesh:
        Lirr_wn[w,n] = Lirr_wnn[w,n,n]

    return Lirr_wn

def irreducible_F(F_wnn,W_w, L_wn):
    # Make irreducible vertex Firr from reducible version F
    # Eq 9 of [1]

    Firr_wnn = F_wnn.copy()
    for w,n1,n2 in F_wnn.mesh:
        Firr_wnn[w,n1,n2] = F_wnn[w,n1,n2] - np.einsum('abfe,efgh,dcgh->abcd', L_wn[Idx(w.index),Idx(n1.index)], W_w[Idx(w.index)], L_wn[Idx(-w.index),Idx(-n2.index-1) ].conj(), optimize=True ) 
        # Mirroring implemented using symmetry of L
        # Note the conventions of the L vertex in Refs [1,2]
        # TPRF's DBSE uses L as a vertex with the bosonic line on the right (Fig 2 of [2])
        # Eq 9 / Fig 1b of [1] have two L vertex, with the bosonic line on the right and left, respectively.
        # Eq 16 of [1] can be used to to map between right-sided and left-sided L-vertices, using complex conjugation
    return Firr_wnn


###

def load_h5(filename):
    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as a:
        p = a['p']
    return p

filename_sc  = 'data_sc.h5'
filename_chi = 'data_susc.h5'
filename_tri = 'data_triangle.h5'
filename_g2  = 'data_g2.h5'

print(f'--> Loading: {filename_sc}')
with HDFArchive(filename_sc, 'r') as a:
    p = a['ps'][-1]

# Remove small (1e-6) off diagonal terms in e_k and g_w by hand
#e_loc = np.sum(p.e_k.data, axis=0).real / p.e_k.data.shape[0]
#e_loc -= np.diag(np.diag(e_loc))
#p.e_k.data[:] -= e_loc[None, ...]

for i, j in itertools.product(range(2), repeat=2):
    if i != j:
        p.g_w[i, j] = 0.


# Interaction 
# Note: this TPRF implementation does the Fierz ambiguity nicely, so that Firr=0 asymptotically
p.U_mat = kanamori_quartic_tensor(1, p.U, p.U, 0, 0)
p.num_orbitals=1


# Impurity susceptibility (one frequency)
p_chi = load_h5(filename_chi)
p2 = p2_from_w2dyn_P2_worm_components(p_chi.GF_worm_components, p.num_orbitals)
p.g_tau = make_gf_from_fourier(p.g_w)
p.chi_imp_w = p2_remove_disconnected(p2, p.g_tau)
# Impurity polarization/screened interaction
p.pi_imp_w, p.W_w, p.Upi_w = impurity_polarization( p.chi_imp_w, p.U_mat)

# "Triangle" impurity two-particle Green's function (two frequencies)
p_tri = load_h5(filename_tri)
p3 = p3_from_w2dyn_P3_worm_components(p_tri.GF_worm_components, p.num_orbitals)
p3 = p3_w2dyn_to_triqs_freq_shift_alt(p3)
p.L_wn = L_from_g3(p3, p.g_w) # remove disconnected and amputate
# U-irreducible part
p.Lirr_wn = irreducible_L(p.L_wn, p.Upi_w)

# "Square" impurity two-particle Green's function (three frequencies)
p_g2 = load_h5(filename_g2)
p.g2_wnn = g2_from_w2dyn_G2_worm_components(
    p_g2.G2_worm_components, p.num_orbitals)

# Lattice dispersion and Green's function
g_wk = lattice_dyson_g_wk(mu=p.mu, e_k=p.e_k, sigma_w=p.sigma_w)

# DBSE, DBSEP calculations for varying frequency window

for nwf in [20, 18, 16, 14, 12, 10, 8, 6, 4]:
    print('='*72)
    print(f'nwf = {nwf}', flush=True)
    p.nwf = nwf
    g2_wnn = G2_loc_fixed_fermionic_window_python(p.g2_wnn, nwf=p.nwf)

    # G2 -> F
    p.F_wnn = impurity_reducible_vertex_F(p.g_w, g2_wnn)
    # U-irreducible part
    p.Firr_wnn = irreducible_F(p.F_wnn, p.W_w, p.Lirr_wn)

    p.chi_kw_dbse = solve_lattice_dbse_lomem(g_wk, p.F_wnn, p.L_wn, p.chi_imp_w)
    p.pi_kw = solve_lattice_dbse_lomem(g_wk, p.Firr_wnn, p.Lirr_wn, p.pi_imp_w) # Same function call, different arguments
    p.chi_kw_krien = pi_kw_to_chi_kw(p.pi_kw, p.U_mat)

    filename_out = f'data_bse_nwf_{nwf:03d}_nk_{p.n_k:03d}.h5'
    print(f'--> Saving: {filename_out}')
    with HDFArchive(filename_out, 'w') as a:
        a['p'] = p
