#include <inq/inq.hpp>

using namespace inq;
using namespace inq::magnitude;

template <class EnvType>
auto compute_GS(EnvType const &env, systems::ions const &ions, int const &nk) {

  auto electrons = systems::electrons(env.par(), ions,
                                      input::kpoints::grid({1, 1, 2}, false),
                                      options::electrons{}
                                          .cutoff(35.0_Ha)
                                          .extra_states(16)
                                          .temperature(300.0_K)
                                          .spin_non_collinear());
  ground_state::initial_guess(ions, electrons);
  auto result =
      ground_state::calculate(ions, electrons, options::theory{}.lda(),
                              inq::options::ground_state{}
                                  .steepest_descent()
                                  .energy_tolerance(1.e-8_Ha)
                                  .max_steps(10000)
                                  .mixing(0.1));
  // auto mag = observables::total_magnetization(electrons.spin_density());
  // std::cout << "mag = (" << mag[0] << ", " << mag[1] << ", " << mag[2] << ")"
  // << std::endl;
}

int main(int argc, char **argv) {
  auto env = input::environment{};

  auto a = 2.866_A;
  auto nk = 2;
  systems::ions ions(systems::cell::lattice({a / 2.0, a / 2.0, a / 2.0},
                                            {-a / 2.0, a / 2.0, a / 2.0},
                                            {-a / 2.0, -a / 2.0, a / 2.0}));
  ions.insert_fractional("Fe", {0.0, 0.0, 0.0});
  compute_GS(env, ions, nk);
}
