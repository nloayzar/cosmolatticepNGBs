#ifndef TANHP4_HIGGS_PNGB_H
#define TANHP4_HIGGS_PNGB_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

#include "CosmoInterface/cosmointerface.h"

namespace TempLat
{
  struct ModelPars : public TempLat::DefaultModelPars {
    static constexpr size_t NScalars = 3;
    static constexpr size_t NPotTerms = 4;

    // Scalar derivative coupling phi * D_mu chi D^mu chi.
    // Ordering is (prefactor field, derivative-coupled field), row-major.
    using ScalarDerivativeCouplings =
        CouplingsManager<NScalars, NScalars, false, false, true, false, false, false, false, false, false>;
  };

#define MODELNAME tanhp4_higgs_pNGB

  template <class R> using Model = MakeModel(R, ModelPars);

  class MODELNAME : public Model<MODELNAME>
  {
  private:
    double M, Lambda4, phii, omega, lambda, sigma, lambdah, masspNGB;
    double lambdaRescaled, sigmaRescaled, lambdahRescaled, masspNGBRescaled;

  public:
    static constexpr size_t NDim = Model<MODELNAME>::NDim;

    MODELNAME(ParameterParser &parser, RunParameters<double> &runPar,
              device::memory::host_ptr<MemoryToolBox<NDim>> toolBox)
        : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL))
    {
      M = parser.get<double>("M");
      Lambda4 = parser.get<double>("Lambda4");
      lambda = parser.get<double>("lambda");
      sigma = parser.get<double>("sigma");
      lambdah = parser.get<double>("lambdah");
      masspNGB = parser.get<double>("masspNGB");

      fldS0 = parser.get<double, 3>("initial_amplitudes");
      piS0 = parser.get<double, 3>("initial_momenta", {0, 0, 0});

      phii = fldS0[0];
      omega = sqrt(Lambda4) * phii / pow<2>(M);

      alpha = 1;
      fStar = phii;
      omegaStar = omega;

      lambdaRescaled = lambda * pow<2>(fStar / omegaStar);
      sigmaRescaled = sigma * fStar / pow<2>(omegaStar);
      lambdahRescaled = lambdah * pow<2>(fStar / omegaStar);
      masspNGBRescaled = masspNGB / omegaStar;

      setInitialPotentialAndMassesFromPotential();
    }

    auto potentialTerms(Tag<0>) // Inflaton tanh^4 potential.
    {
      return 0.25 * pow<4>(M / phii) * pow<4>(tanh(fldS(0_c) * phii / M));
    }

    auto potentialTerms(Tag<1>) // Higgs mass term.
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
      return pow<3>(M / phii) * pow<3>(tanh(fldS(0_c) * phii / M)) /
                 pow<2>(cosh(fldS(0_c) * phii / M)) +
             2.0 * lambdaRescaled * fldS(0_c) * pow<2>(fldS(1_c)) + sigmaRescaled * pow<2>(fldS(1_c));
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
      return pow<2>(M / phii) * (4.0 - cosh(2.0 * fldS(0_c) * phii / M)) *
                 pow<2>(tanh(fldS(0_c) * phii / M)) / pow<4>(cosh(fldS(0_c) * phii / M)) +
             2.0 * lambdaRescaled * pow<2>(fldS(1_c));
    }

    auto potDeriv2(Tag<1>) // d^2 V / dh^2.
    {
      return 3.0 * lambdahRescaled * pow<2>(fldS(1_c)) + 2.0 * lambdaRescaled * pow<2>(fldS(0_c)) + 2.0 * sigmaRescaled * fldS(0_c);
    }

    auto potDeriv2(Tag<2>) // d^2 V / dchi^2.
    {
      return pow<2>(masspNGBRescaled);
    }
  };
} // namespace TempLat

#endif // TANHP4_HIGGS_PNGB_H
