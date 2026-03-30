/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sharmonic.h>
#include <stdio.h>

int main() {

  double rsh = sharmonic_cartesian_real(4, 4, 0.5, 1.5, 2.5);

  printf("Cartesian real\n");
  printf("=================\n");
  double ref_rsh = 0.0143048168;
  double rdiff = fabs(rsh - ref_rsh);
  printf("Calculated = %e\n", rsh);
  printf("Reference  = %e\n", ref_rsh);
  printf("Difference = %e\n\n", rdiff);
  int ok = rdiff < 1e-10;

  
  double complex zsh = sharmonic_cartesian_complex(3, -2, 0.5, 1.5, 2.5);
  
  printf("Cartesian complex\n");
  printf("=================\n");
  double complex ref_zsh = -0.1974252283 + I*-0.1480689212;
  double zdiff = cabs(zsh - ref_zsh);
  printf("Calculated = %e + i*%e\n", creal(zsh), cimag(zsh));
  printf("Reference  = %e + i*%e\n", creal(ref_zsh), cimag(ref_zsh));
  printf("Difference = %e\n\n", zdiff);
  ok = ok && rdiff < 1e-10;
  
  return !ok;
}
