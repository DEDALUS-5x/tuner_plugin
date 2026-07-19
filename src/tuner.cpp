#include "tuner.hpp"


Tuner::Tuner(TuningParams start){

  _p_base = start;
  _bounds["kp"] = {0.0f, 120.0f};
  _bounds["ki"] = {0.0f, 2.0f};
  _bounds["kd"] = {0.0f, 10.0f};
  _bounds["kv"] = {0.0f, 100.0f};
  _bounds["ka"] = {0.0f, 2.0f};
}

void Tuner::set_bounds(map<string, TuningBounds> new_bounds) {
    _bounds = new_bounds;
}

void Tuner::reset(){
  _target_history.clear();
  _real_history.clear();
  _target_history.reserve(5000); 
  _real_history.reserve(5000);
}

void Tuner::log_data(float target, float actual){

  _target_history.push_back(target);
  _real_history.push_back(actual);
}

float Tuner::kinematic_cost(){
  if (!full_buffer()) return 1e9f;

  float total_cost = 0.0f;
  for (size_t i = 2; i < _target_history.size(); ++i) {
      float err = std::abs(_target_history[i - 1] - _real_history[i]);
      // kinematic for error weight
      float vel = std::abs(_target_history[i-1] - _target_history[i-2]);
      // penalize more as faster is the machine
      float kinematic_weight = 1.0f + (vel * 10.0f); 
      // integral time absolute error
      total_cost += err * kinematic_weight * (float)i; 
  }
  return total_cost / _target_history.size();
}

TuningParams Tuner::process_iteration(){

  _current_cost = kinematic_cost();
  std::cout << "Iteration state: " << _current << " | Cost: " << _current_cost << std::endl;

  TuningParams next_test_params = _p_base;

  switch (_current) {
      case EVAL_BASE:
          _cost_base = _current_cost;
          next_test_params.kp += _delta;
          _current = EVAL_KP;
          break;

      case EVAL_KP:
          _cost_kp = _current_cost;
          next_test_params.ki += _delta;
          _current = EVAL_KI;
          break;

      case EVAL_KI:
          _cost_ki = _current_cost;
          next_test_params.kd += _delta;
          _current = EVAL_KD;
          break;

      case EVAL_KD:
          _cost_kd = _current_cost;
          next_test_params.kv += _delta;
          _current = EVAL_KV;
          break;

      case EVAL_KV:
          _cost_kv = _current_cost;
          next_test_params.ka += _delta;
          _current = EVAL_KA;
          break;

      case EVAL_KA:
          _cost_ka = _current_cost;
          _current = APPLY_ADAM;
          return apply_adam(); // Applica e ricomincia
  }

  return next_test_params;
}

TuningParams Tuner::apply_adam(){
  _t_step++;
  TuningParams grad;
  grad.kp = (_cost_kp - _cost_base) / _delta;
  grad.ki = (_cost_ki - _cost_base) / _delta;
  grad.kd = (_cost_kd - _cost_base) / _delta;
  grad.kv = (_cost_kv - _cost_base) / _delta;
  grad.ka = (_cost_ka - _cost_base) / _delta;

  // lambda for ADAM single parameter
  auto update_param = [&](float& p, float g, float& m_val, float& v_val, string name) {

    // moments update
    m_val = _beta_1 * m_val + (1.0f - _beta_1) * g;
    v_val = _beta_2 * v_val + (1.0f - _beta_2) * (g * g);

    // bias correction and boundaries
    float m_hat = m_val / (1.0f - pow(_beta_1, _t_step));
    float v_hat = v_val / (1.0f - pow(_beta_2, _t_step));
    if (_bounds[name].max_val > 0.0001f) {
      p -= _alpha * m_hat / (sqrt(v_hat) + _epsilon);
      p = max(_bounds[name].min_val, min(p, _bounds[name].max_val));
    } else {
      p = 0.0f;
    }
  };

  update_param(_p_base.kp, grad.kp, m.kp, v.kp, "kp");
  update_param(_p_base.ki, grad.ki, m.ki, v.ki, "ki");
  update_param(_p_base.kd, grad.kd, m.kd, v.kd, "kd");
  update_param(_p_base.kv, grad.kv, m.kv, v.kv, "kv");
  update_param(_p_base.ka, grad.ka, m.ka, v.ka, "ka");

  cout << "[AdamTuner] ADAM fineshed: new parms: " 
            << "Kp=" << _p_base.kp << " Kv=" << _p_base.kv << endl;

  _current = EVAL_BASE;
  return _p_base;
}