#ifndef __TUNER_HPP__
#define __TUNER_HPP__

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <map>

using namespace std;

struct TuningParams {
    float kp;
    float ki;
    float kd;
    float kv; 
    float ka; 
};

struct TuningBounds {
    float min_val;
    float max_val;
};

class Tuner{
  
  public:

    Tuner(TuningParams start);
    void reset();
    void log_data(float target, float actual);
    bool full_buffer() { return _target_history.size() > 100; }
    float kinematic_cost();
    TuningParams process_iteration();
    TuningParams apply_adam();
    TuningParams get_p() const { return _p_base; }
    float get_last_cost() const { return _current_cost; }
    void set_bounds(map<string, TuningBounds> new_bounds);

  private:
    enum State{
      EVAL_BASE,
      EVAL_KP,
      EVAL_KI,
      EVAL_KD,
      EVAL_KV,
      EVAL_KA,
      APPLY_ADAM
    };

    State _current = EVAL_BASE;
    TuningParams _p_base;
    float _alpha = 0.05f;
    float _beta_1 = 0.9f;
    float _beta_2 = 0.99f;
    float _epsilon = 1e-8f;
    int _t_step = 0;
    float _current_cost;

    TuningParams _alphas = {1.0f, 0.1f, 0.001f, 0.1f, 0.001f}; // learning rates
    TuningParams _deltas = {0.5f, 0.05f, 0.001f, 0.05f, 0.001f};

    TuningParams m = {0, 0, 0, 0, 0};
    TuningParams v = {0, 0, 0, 0, 0};
    float _delta = 0.01f;
    float _cost_base = 0.0f;
    float _cost_kp = 0.0f, _cost_ki = 0.0f, _cost_kd = 0.0f, _cost_kv = 0.0f, _cost_ka = 0.0f;
    vector<float> _target_history;
    vector<float> _real_history;
    map<string, TuningBounds> _bounds;
 
};

#endif