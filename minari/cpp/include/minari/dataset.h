#include <vector>
#include <memory>

namespace minari {

  using namespace std;

  struct CTransition {
    vector<int> observation_shape;
    int action_size;
    unsigned char* observation_i;
    float* observation_f;
    int action_i;
    float* action_f;
    float reward;
    unsigned char* next_observation_i;
    float* next_observation_f;
    float termination;
    float truncation;
    shared_ptr<CTransition> prev_transition;
    shared_ptr<CTransition> next_transition;
  };

}
