#ifndef M2PHI2_TRILINEAR_HIGGS_H
#define M2PHI2_TRILINEAR_HIGGS_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

#include "CosmoInterface/cosmointerface.h"

namespace TempLat
{
  struct ModelPars : public TempLat::DefaultModelPars {
    static constexpr size_t NScalars = 2;
    static constexpr size_t NPotTerms = 3;
  };

#define MODELNAME m2phi2_trilinear_higgs

  template <class R> using Model = MakeModel(R, ModelPars);

  class MODELNAME : public Model<MODELNAME>
  {
  private:
    double mass, phii, lambdah, sigma;
    double massRescaled, lambdahRescaled , sigmaRescaled;

  public:
    static constexpr size_t NDim = Model<MODELNAME>::NDim;

    MODELNAME(ParameterParser &parser, RunParameters<double> &runPar,
              device::memory::host_ptr<MemoryToolBox<NDim>> toolBox)
        : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL))
    {
      mass = parser.get<double>("mass");
      sigma = parser.get<double>("sigma");
      lambdah = parser.get<double>("lambdah");


      fldS0 = parser.get<double, 2>("initial_amplitudes");
      piS0 = parser.get<double, 2>("initial_momenta", {0, 0});

      phii = fldS0[0];

      alpha = 0;
      fStar = phii;
      omegaStar = mass;

      massRescaled = mass / omegaStar;
      sigmaRescaled = sigma * fStar / pow<2>(omegaStar);
      lambdahRescaled = lambdah * pow<2>(fStar / omegaStar);

      setInitialPotentialAndMassesFromPotential();
    }

    auto potentialTerms(Tag<0>) // Inflaton mass^2 phi^2 potential.
    {
      return 0.5 * pow<2>(massRescaled) * pow<2>(fldS(0_c));
    }

    auto potentialTerms(Tag<1>) // Trilinear inflaton-Higgs interaction.
    {
      return 0.5 * sigmaRescaled * fldS(0_c) * pow<2>(fldS(1_c));
    }

    auto potentialTerms(Tag<2>) // Higgs self-interaction.
    {
      return 0.25 * lambdahRescaled * pow<4>(fldS(1_c));
    }

    auto potDeriv(Tag<0>) // dV / dphi.
    {
      return pow<2>(massRescaled) * fldS(0_c) + 0.5 * sigmaRescaled * pow<2>(fldS(1_c));
    }

    auto potDeriv(Tag<1>) // dV / dh.
    {
      return lambdahRescaled * pow<3>(fldS(1_c)) + sigmaRescaled * fldS(0_c) * fldS(1_c);
    }

    auto potDeriv2(Tag<0>) // d^2 V / dphi^2.
    {
      return pow<2>(massRescaled);
    }

    auto potDeriv2(Tag<1>) // d^2 V / dh^2.
    {
      return 3.0 * lambdahRescaled * pow<2>(fldS(1_c)) + sigmaRescaled * fldS(0_c);
    }
  };
} // namespace TempLat

#endif // M2PHI2_TRILINEAR_HIGGS_H
