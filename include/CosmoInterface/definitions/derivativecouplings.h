#ifndef COSMOINTERFACE_DEFINITIONS_DERIVATIVECOUPLINGS_H
#define COSMOINTERFACE_DEFINITIONS_DERIVATIVECOUPLINGS_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

#include "CosmoInterface/couplingsmanager.h"
#include "TempLat/util/rangeiteration/sum_in_range.h"
#include "TempLat/util/staticif.h"
#include "CosmoInterface/definitions/fieldfunctionals.h"
#include "CosmoInterface/definitions/gaugederivatives.h"

namespace TempLat
{
  class DerivativeCouplings
  {
  public:
    DerivativeCouplings() = delete;

    template <class Model, int N> static auto ScalarDerivativeSource(Model &model, Tag<N> n)
    {
      return ScalarDerivativeSourceInf(model, n) + ScalarDerivativeSourcepNGB(model, n);
    }

    template <class Model, int N> static auto ScalarDerivativeSourceInf(Model &model, Tag<N> n)
    {
      // Contributions where field n is the prefactor phi in phi * D_mu chi D^mu chi.
      auto sourceFromPrefactorField =
          Total(chi, 0, Model::Ns - 1,
                IfElse(Model::ScalarDerivativeCouplings::couples(n, chi),
                       1.0 / (model.derivativeCouplings(n, chi) * (Model::MPl/model.fStar)) 
                       * (pow(model.aI,model.alpha-3)* FieldFunctionals::pi2S(model,chi)
                       - pow(model.aI,model.alpha+1) * FieldFunctionals::grad2S(model,chi) ) , ZeroType()));
      
      return sourceFromPrefactorField;
    }

    template <class Model, int N> static auto ScalarDerivativeSourcepNGB(Model &model, Tag<N> n)
    {
      // Contributions where field n is the differentiated scalar chi in phi * D_mu chi D^mu chi.
      auto sourceFromDerivativeField =
          Total(phi, 0, Model::Ns - 1,
                IfElse(Model::ScalarDerivativeCouplings::couples(phi, n),
                      - 2.0 /(model.derivativeCouplings(phi, n)*(Model::MPl/model.fStar) + 2.0 * model.fldS(phi)) * (pow(model.aI,model.alpha-3) * model.piS(n) * model.piS(phi) - pow(model.aI,model.alpha+1) * GaugeDerivatives::GradientScalarProduct(model,n,phi)) 
                      - 1.0 /(1.0 + 2.0 * model.fldS(phi)/(model.derivativeCouplings(phi, n)*(Model::MPl/model.fStar))) * pow(model.aI,model.alpha+3) * Potential::derivS(model, n) 
                      + pow(model.aI,model.alpha+3) * Potential::derivS(model, n), ZeroType()));

       return sourceFromDerivativeField;
    }
  };

} // namespace TempLat

#endif
