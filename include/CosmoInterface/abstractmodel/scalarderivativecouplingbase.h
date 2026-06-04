#ifndef TEMPLAT_COSMOINTERFACE_ABSTRACTMODEL_SCALARDERIVATIVECOUPLINGBASE_H
#define TEMPLAT_COSMOINTERFACE_ABSTRACTMODEL_SCALARDERIVATIVECOUPLINGBASE_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

#include "CosmoInterface/couplingsmanager.h"
#include "TempLat/parameters/parameterparser.h"
#include "TempLat/util/templatarray.h"

#include <vector>

namespace TempLat
{
  template <typename T, size_t NS, typename SCALARDERIVATIVECOUPLINGS> class ScalarDerivativeCouplingBase
  {
  public:
    using ScalarDerivativeCouplings = typename SCALARDERIVATIVECOUPLINGS::template Container<T>;
    static constexpr bool IsDerivativeCoupled = SCALARDERIVATIVECOUPLINGS::howManyCouples() > 0;

    ScalarDerivativeCouplings derivativeCouplings;

    // Averages needed for derivative-coupling contributions to the scale-factor kernel.
    TempLatArray<T, NS * NS> derivativeCouplingPi2AvI;
    TempLatArray<T, NS * NS> derivativeCouplingPi2AvSI;
    TempLatArray<T, NS * NS> derivativeCouplingPi2AvIM;
    TempLatArray<T, NS * NS> derivativeCouplingPi2AvSIM;
    TempLatArray<T, NS * NS> derivativeCouplingGrad2AvI;
    TempLatArray<T, NS * NS> derivativeCouplingGrad2AvSI;

  protected:
    ScalarDerivativeCouplingBase(ParameterParser &parser)
    {
      auto derivativeCouplingCoefficients =
          parser.get<T, ScalarDerivativeCouplings::howManyCouples()>("lambda_DerivativeCoupling", 1);
      std::vector<T> unitCouplings(ScalarDerivativeCouplings::nGauge, T(1));
      derivativeCouplings.setEffectiveCharges(derivativeCouplingCoefficients, unitCouplings);
    }
  };

} // namespace TempLat

#endif
