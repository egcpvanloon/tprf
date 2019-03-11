/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2019, The Simons Foundation and S. Käser
 * Authors: H. U.R. Strand, S. Käser
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

#include "../types.hpp"

namespace tprf {

  /** Random Phase Approximation (RPA) in the particle-hole channel
   
     Computes the equation

     .. math::
         \chi^{(\mathrm{s})}(\bar{a}b\bar{c}d) = \big(
         \mathbb{1} 
         - \chi^{(0)}(\bar{a}b\bar{B}A) U^{(\mathrm{s})}(A\bar{B}D\bar{C})
         \big)^{-1} \chi^{(0)}(\bar{C}D\bar{c}d)\,.
     
     @param chi0 bare particle-hole bubble :math:`\chi^{(0)}_{\bar{a}b\bar{c}d}(\mathbf{k}, i\omega_n)`
     @param U RPA static vertex as obtained from triqs_tprf.rpa_tensor.get_rpa_tensor :math:`U_{a\bar{b}c\bar{d}}`
     @return RPA suceptibility :math:`\chi_{\bar{a}b\bar{c}d}(\mathbf{k}, i\omega_n)`
  */

  chi_wk_t solve_rpa_PH(chi_wk_vt chi0, array_view<std::complex<double>, 4> U);

  /** Matrix RPA spin tensor
   

     Computes the equation

     .. math::
         \chi^{(\mathrm{s})}(\bar{a}b\bar{c}d) = \big(\mathbb{1} - \chi^{(0)}(\bar{a}b\bar{A}B) 
         U^{(\mathrm{s})}(\bar{A}B\bar{C}D)\big)^{-1}  \chi^{(0)}(\bar{C}D\bar{c}d)\,.
     
     @param chi0 spin-independent bare particle-hole bubble :math:`\chi^{(0)}_{\bar{a}b\bar{c}d}(\mathbf{k}, i\omega_n)`
     @param U the spin interaction tensor which can be obtained by triqs_tprf.rpa_tensor.get_rpa_us_tensor :math:`U^{(\mathrm{s})}_{\bar{a}b\bar{c}d}`
     @return The spin tensor as a rank 4 bosonic Matsubara frequency lattice Green's function :math:`\chi^{(\mathrm{s})}_{\bar{a}b\bar{c}d}(\mathbf{k}, i\omega_n)`
  */

  chi_wk_t solve_rpa_spin(chi_wk_vt chi0, array_view<std::complex<double>, 4> U);

  /** Matrix RPA charge tensor
   

     Computes the equation

     .. math::
         \chi^{(\mathrm{c})}(\bar{a}b\bar{c}d) = \big(\mathbb{1} - \chi^{(0)}(\bar{a}b\bar{A}B) 
         U^{(\mathrm{c})}(\bar{A}B\bar{C}D)\big)^{-1}  \chi^{(0)}(\bar{C}D\bar{c}d)\,.
     
     @param chi0 spin-independent bare particle-hole bubble :math:`\chi^{(0)}_{\bar{a}b\bar{c}d}(\mathbf{k}, i\omega_n)`
     @param U the charge interaction tensor which can be obtained by triqs_tprf.rpa_tensor.get_rpa_uc_tensor :math:`U^{(\mathrm{c})}_{\bar{a}b\bar{c}d}`
     @return The spin tensor as a rank 4 bosonic Matsubara frequency lattice Green's function :math:`\chi^{(\mathrm{c})}_{\bar{a}b\bar{c}d}(\mathbf{k}, i\omega_n)`
  */

  chi_wk_t solve_rpa_charge(chi_wk_vt chi0, array_view<std::complex<double>, 4> U);
  
} // namespace tprf
