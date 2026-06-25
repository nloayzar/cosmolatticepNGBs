#ifndef M2PHI2_HIGGS_PNGB_POLY2_H
#define M2PHI2_HIGGS_PNGB_POLY2_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

#include "CosmoInterface/cosmointerface.h"

namespace TempLat
{
  struct ModelPars : public TempLat::DefaultModelPars {
    static constexpr size_t NScalars = 3;
    static constexpr size_t NPotTerms = 4;

    // Scalar derivative coupling f(phi) * D_mu chi D^mu chi.
    // Ordering is (prefactor field, derivative-coupled field), row-major.
    using ScalarDerivativeCouplings =
        CouplingsManager<NScalars, NScalars, false, false, true, false, false, false, false, false, false>;
  };

#define MODELNAME m2phi2_higgs_pNGB_Poly2

  template <class R> using Model = MakeModel(R, ModelPars);

  class MODELNAME : public Model<MODELNAME>
  {
  private:
    double mass, phii, lambda, sigma, lambdah, masspNGB;
    double lambdaRescaled, sigmaRescaled, lambdahRescaled, masspNGBRescaled;

  public:
    using Model<MODELNAME>::derivativeCouplingFunction;
    using Model<MODELNAME>::derivativeCouplingFunctionDeriv;

    static constexpr size_t NDim = Model<MODELNAME>::NDim;

    MODELNAME(ParameterParser &parser, RunParameters<double> &runPar,
              device::memory::host_ptr<MemoryToolBox<NDim>> toolBox)
        : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL))
    {
      mass = parser.get<double>("mass");
      lambda = parser.get<double>("lambda");
      sigma = parser.get<double>("sigma");
      lambdah = parser.get<double>("lambdah");
      masspNGB = parser.get<double>("masspNGB");

      fldS0 = parser.get<double, 3>("initial_amplitudes");
      piS0 = parser.get<double, 3>("initial_momenta", {0, 0, 0});

      phii = fldS0[0];

      alpha = 0;
      fStar = phii;
      omegaStar = mass;

      lambdaRescaled = lambda * pow<2>(fStar / omegaStar);
      sigmaRescaled = sigma * fStar / pow<2>(omegaStar);
      lambdahRescaled = lambdah * pow<2>(fStar / omegaStar);
      masspNGBRescaled = masspNGB / omegaStar;

      setInitialPotentialAndMassesFromPotential();
    }

    auto derivativeCouplingFunction(Tag<0>, Tag<2>) // f(phi) = phi / Lambda + (phi / Lambda)^2.
    {
      auto Lambda = derivativeCouplings(0_c, 2_c) * (Model<MODELNAME>::MPl / fStar);
      auto phiOverLambda = fldS(0_c) / Lambda;
      return phiOverLambda + pow<2>(phiOverLambda);
    }

    auto derivativeCouplingFunctionDeriv(Tag<0>, Tag<2>) // df / dphi.
    {
      auto Lambda = derivativeCouplings(0_c, 2_c) * (Model<MODELNAME>::MPl / fStar);
      auto phiOverLambda = fldS(0_c) / Lambda;
      return (1.0 + 2.0 * phiOverLambda) / Lambda;
    }

    auto potentialTerms(Tag<0>) // Inflaton quadratic potential.
    {
      return 0.5 * pow<2>(fldS(0_c));
    }

    auto potentialTerms(Tag<1>) // Higgs quartic term.
    {
      return 0.25 * lambdahRescaled * pow<4>(fldS(1_c));
    }

    auto potentialTerms(Tag<2>) // pNGB mass term.
    {
      return 0.5 * pow<2>(masspNGBRescaled) * pow<2>(fldS(2_c));
    }

    auto potentialTerms(Tag<3>) // Inflaton-Higgs interactions.
    {
      return lambdaRescaled * pow<2>(fldS(0_c)) * pow<2>(fldS(1_c)) +
             sigmaRescaled * fldS(0_c) * pow<2>(fldS(1_c));
    }

    auto potDeriv(Tag<0>) // dV / dphi.
    {
      return fldS(0_c) + 2.0 * lambdaRescaled * fldS(0_c) * pow<2>(fldS(1_c)) +
             sigmaRescaled * pow<2>(fldS(1_c));
    }

    auto potDeriv(Tag<1>) // dV / dh.
    {
      return lambdahRescaled * pow<3>(fldS(1_c)) + 2.0 * lambdaRescaled * pow<2>(fldS(0_c)) * fldS(1_c) +
             2.0 * sigmaRescaled * fldS(0_c) * fldS(1_c);
    }

    auto potDeriv(Tag<2>) // dV / dchi.
    {
      return pow<2>(masspNGBRescaled) * fldS(2_c);
    }

    auto potDeriv2(Tag<0>) // d^2 V / dphi^2.
    {
      return 1.0 + 2.0 * lambdaRescaled * pow<2>(fldS(1_c));
    }

    auto potDeriv2(Tag<1>) // d^2 V / dh^2.
    {
      return 3.0 * lambdahRescaled * pow<2>(fldS(1_c)) + 2.0 * lambdaRescaled * pow<2>(fldS(0_c)) +
             2.0 * sigmaRescaled * fldS(0_c);
    }

    auto potDeriv2(Tag<2>) // d^2 V / dchi^2.
    {
      return pow<2>(masspNGBRescaled);
    }
  };
} // namespace TempLat

#endif // M2PHI2_HIGGS_PNGB_POLY2_H
