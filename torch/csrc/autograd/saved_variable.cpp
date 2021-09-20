#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/anomaly_mode.h>

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

// For FlashNeuron headers
#include <c10/core/TensorOptions.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <thread>
#include <queue>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <THC/THCGeneral.h>
#include <c10/core/ScalarType.h>

#include <c10/cuda/CUDACachingAllocator.h>

namespace torch { namespace autograd {

FlashNeuronEngine FNEngine;

SavedVariable::SavedVariable(const Variable& variable, bool is_output, bool is_inplace_view) {
  if (variable.defined()) {
    was_default_constructed_ = false;
    output_nr_ = variable.output_nr();
    requires_grad_ = variable.requires_grad();
    has_grad_fn_ = !variable.is_leaf();
    is_inplace_view_ = is_inplace_view;
    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    data_ = variable.tensor_data();
    // TODO(albanD) This needs to be updated when moving to multiple levels
    const auto& fw_grad = variable.fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
      fw_grad_ = std::make_shared<ForwardGrad>();
      fw_grad_->set_value(fw_grad, /* level */ 0);
    }
    if (variable.is_leaf()) {
      grad_accumulator_ = impl::grad_accumulator(variable);
    } else if (!is_output) {
      grad_fn_ = variable.grad_fn();
    } else if (is_inplace_view) {
      weak_grad_fn_ = variable.grad_fn();
    }
    version_counter_ = impl::version_counter(variable);
    saved_version_ = version_counter_.current_version();
  }
}

SavedVariable::SavedVariable(const c10::optional<Variable>& variable, bool is_output, bool is_inplace_view)
  : SavedVariable(variable.has_value() ? *variable : Variable(), is_output, is_inplace_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  auto grad_fn = is_inplace_view_ ? weak_grad_fn_.lock() : grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    grad_fn = std::move(saved_for);
  }

  if (saved_version_ != version_counter_.current_version()) {
    std::stringstream message;
    message << "one of the variables needed for gradient computation has been "
        "modified by an inplace operation: [" << data_.toString() << " "
        << data_.sizes() << "]";
    if (grad_fn) {
        message << ", which is output " << output_nr_
            << " of " << grad_fn->name() << ",";
    }
    message << " is at version " << version_counter_.current_version()
        << "; expected version " << saved_version_ << " instead.";
    if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
            "that failed to compute its gradient, with torch.autograd."
            "set_detect_anomaly(True).";
    }
    else {
        message << " Hint: the backtrace further above shows the operation "
            "that failed to compute its gradient. The variable in question "
            "was changed in there or anywhere later. Good luck!";
    }
    throw std::runtime_error(message.str());
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data_, requires_grad_);
  }
  impl::set_version_counter(var, saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  impl::set_grad_accumulator(var, grad_accumulator_);

  // NB: var here is never a view so there is no need to make anything special
  // for the case where the saved Tensor was a view. This whole argument relies
  // on the fact that the Tensor returned by this function is never
  // modified in-place.
  if (fw_grad_ && !fw_grad_->empty()) {
    // TODO(albanD) This needs to be updated when moving to multiple levels
    auto new_fw_grad = fw_grad_->value(/* level */ 0);
    var.set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
  }

  return var;
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the saved intermediate "
    "results have already been freed. Specify retain_graph=True when calling "
    ".backward() or autograd.grad() the first time.";

/****************
 *  FlashNeuron
 ****************/

// Dictionaries for offloading/prefetching
static std::map<Oid, std::vector<PFInfo>> op_tensor_list;

// static std::map<Oid, c10::StreamId> stream_occupied;

static at::Tensor target_tensor[NUM_TENSOR];
static bool target_tensor_valid[NUM_TENSOR] = {false};

static bool is_tensor_offloaded[NUM_TENSOR] = {false};
static bool is_tensor_prefetched[NUM_TENSOR] = {false};
static int lifetime_forward[NUM_TENSOR] = {-1};
static int lifetime_backward[NUM_TENSOR] = {-1};

static double liveness_time[NUM_TENSOR] = {0.0};
static double liveness_size[NUM_TENSOR] = {0.0};
static bool liveness_csr[NUM_TENSOR] = {false};
static bool liveness_fp[NUM_TENSOR] = {false};

static double last_time_slot = 0;

static double accumSize = 0;
static double ssd_w = 3072;
static double ssd_r = 10240;
static double mem_wr = 12288;

static int* pref_order;
static int pref_order_idx;
static int pref_order_last;

void FlashNeuronEngine::offloading_scheduler(double freeSize) {
  double accumTime = 0;
  double delay_time[NUM_TENSOR] = {0};
  double real_trans_time[NUM_TENSOR] = {0};
  double real_trans_start[NUM_TENSOR] = {0};
  double remainSize = accumSize - freeSize;
  accumSize = 0;
  // int cur_back_num = at::globalContext().FNGlobal.curBackNum();

  for (int i = 0; i < NUM_TENSOR-1; i++)
    liveness_time[i + 1] += liveness_time[i];

  last_time_slot += liveness_time[NUM_TENSOR - 1];

  std::cout << "last time: " << last_time_slot << std::endl;

  for (int i = 0; i < NUM_TENSOR; i++) {
    if (at::native::fn_memorymanager.is_using_ssd())
      real_trans_time[i] = liveness_size[i] * 1000 / ssd_w;
    else
      real_trans_time[i] = liveness_size[i] * 1000 / mem_wr;

    if (at::native::fn_memorymanager.is_fp16() && liveness_fp[i])
      real_trans_time[i] = real_trans_time[i] / 2;

    if (at::native::fn_memorymanager.is_csr() && liveness_csr[i])
      real_trans_time[i] = real_trans_time[i] / 2;
  }

  if (remainSize <= 0) {
    std::cout << "Nothing" << std::endl;
    return;
  }

  int previous_i = 0;

  for (int i = 0; i < NUM_TENSOR; i++) {
    if (liveness_size[i] > 1) {
      at::native::fn_memorymanager.liveness_result[i] = true;
      remainSize -= liveness_size[i];

      previous_i = i;
      break;
    }
  }

  for (int i = previous_i + 1; i < NUM_TENSOR; i++) {
    if (liveness_size[i] > 1) {
      at::native::fn_memorymanager.liveness_result[i] = true;
      remainSize -= liveness_size[i];

      double delay_maybe = liveness_time[previous_i] + real_trans_time[previous_i] + delay_time[previous_i] - liveness_time[i];

      if (delay_maybe <= 0)
        delay_time[i] = 0;
      else
        delay_time[i] = delay_maybe;

      previous_i = i;

      if (remainSize < 0) {
        break;
      }
    }
  }

  bool timeout = (liveness_time[previous_i] + real_trans_time[previous_i] + delay_time[previous_i]) > last_time_slot;

  if (previous_i + 1 == NUM_TENSOR) {
    exit(1);
  }

  if (timeout) {
    for (int i = previous_i + 1; i < NUM_TENSOR; i++) {
      int delete_i = 0;
      for (delete_i = previous_i; delete_i >= 0; delete_i--) {
        if (liveness_size[delete_i] > 1 && at::native::fn_memorymanager.liveness_result[delete_i] &&
            !liveness_csr[delete_i] && !liveness_fp[delete_i]) {
          break;
        }
      }

      // if (delete_i == -1 && !at::globalContext().FNGlobal.isBERT()) {
      if (delete_i == -1) {
        for (delete_i = previous_i; delete_i >= 0; delete_i--) {
          if (liveness_size[delete_i] > 1 && at::native::fn_memorymanager.liveness_result[delete_i] && !liveness_csr[delete_i]) {
            break;
          }
        }
      }

      double delete_size = liveness_size[delete_i];

      int delete_previous_i = 0;
      if (delete_i == -1) {
        break;
      }

      at::native::fn_memorymanager.liveness_result[delete_i] = false;

      int add_i = 0;

      while (delete_size > 0) {
        for (delete_previous_i = delete_i; delete_previous_i >= 0; delete_previous_i--) {
          if (at::native::fn_memorymanager.liveness_result[delete_previous_i]) {
            break;
          }
        }

        for (add_i = previous_i + 1; add_i < NUM_TENSOR; add_i++) {
          if (liveness_size[add_i] > 1 && liveness_csr[add_i] && !at::native::fn_memorymanager.liveness_result[add_i]) {
            break;
          }
        }

        if (add_i == NUM_TENSOR) {
          for (add_i = previous_i + 1; add_i < NUM_TENSOR; add_i++) {
            if (liveness_size[add_i] > 1 && liveness_fp[add_i] && !at::native::fn_memorymanager.liveness_result[add_i]) {
              break;
            }
          }
        }

        if (add_i == NUM_TENSOR) {
          break;
        }

        at::native::fn_memorymanager.liveness_result[add_i] = true;
        for (int test = delete_i; test <= add_i; test++) {
          if (at::native::fn_memorymanager.liveness_result[test]) {
            if (delete_previous_i != -1) {
              double delay_maybe = liveness_time[delete_previous_i] + real_trans_time[delete_previous_i] + delay_time[delete_previous_i] - liveness_time[test];
              if (delay_maybe <= 0) {
                delay_time[test] = 0;
              } else {
                delay_time[test] = delay_maybe;
              }
            } else {
              delay_time[test] = 0;
            }

            delete_previous_i = test;
          }
        }

        i = add_i;
        delete_size -= liveness_size[add_i];
      }

      bool timeout = (liveness_time[add_i] + real_trans_time[add_i] + delay_time[add_i]) > last_time_slot;
      if (!timeout) {
        break;
      }
    }

    if (remainSize > 0) {
      for (int i = 0; i < NUM_TENSOR; i++) {
        if (liveness_size[i] > 1 && !at::native::fn_memorymanager.liveness_result[i] && liveness_csr[i]) {
          remainSize -= liveness_size[i];
          at::native::fn_memorymanager.liveness_result[i] = true;
        }

        if (remainSize < 0) break;
      }
    }

    if (remainSize > 0) {
      for (int i = 0; i < NUM_TENSOR; i++) {
        if (liveness_size[i] > 1 && !at::native::fn_memorymanager.liveness_result[i] && liveness_fp[i]) {
          remainSize -= liveness_size[i];
          at::native::fn_memorymanager.liveness_result[i] = true;
        }

        if (remainSize < 0) break;
      }
    }
  }
}

void FlashNeuronEngine::offLoad(at::Tensor t, Oid oid, SavedVariable* fetch_loc, bool isOutput) {

  if (at::globalContext().FNGlobal.isForward()) {
    if (!at::native::fn_memorymanager.is_fn()) {
      *fetch_loc = SavedVariable(t, isOutput);
      return;
    }

    auto tid = t.unsafeGetIntrusivePtr()->tensor_id;
    if (tid == -1) {
      at::globalContext().FNGlobal.setNewTid(t);
      tid = t.unsafeGetIntrusivePtr()->tensor_id;

      if (at::native::fn_memorymanager.is_debug())
        std::cout << "Missing allocator in oid: " << oid << ", " << tid << std::endl;
        // std::cout << "New oid: " << oid << ", tid: " << tid << std::endl;
    }

    if (at::native::fn_memorymanager.is_debug())
      std::cout << "Offload try oid: " << oid << ", tid: " << tid << ", csr: " << at::native::fn_memorymanager.relu_thru << std::endl;

    insertToPFDict_(oid, fetch_loc, tid);
    if (is_tensor_offloaded[tid]) {
      at::native::fn_memorymanager.relu_thru = false;
      return;
    } else {
      is_tensor_offloaded[tid] = true;
    }

    c10::TensorOptions opt = c10::TensorOptions();
    opt = opt.device(c10::Device(c10::DeviceType::CPU));
    opt = opt.dtype(t.dtype());
    opt = opt.pinned_memory(true);

    if (at::native::fn_memorymanager.is_using_ssd()) {
      // at::native::fn_memorymanager.set_dir(tid, at::native::p2pdsa_gputossd);
      // at::native::p2pdsa_cpl *p_cpl = new at::native::p2pdsa_cpl;
      // at::native::fn_memorymanager.set_cpl_addr(tid, at::native::p2pdsa_gputossd, (void *)p_cpl);
    }

    if (at::globalContext().FNGlobal.isOnDemand()) {
      float elapsed = 0;
      if (accumSize > 0) {
        elapsed = at::native::fn_memorymanager.timeEnd();

        if (at::native::fn_memorymanager.is_debug())
          std::cout << "elapsed: " << elapsed << std::endl;
      }

      accumSize += (double)t.nbytes() / 1024 / 1024;

      liveness_time[oid] += elapsed;
      liveness_size[tid] = (double)t.nbytes() / 1024 / 1024;

      if (at::native::fn_memorymanager.is_csr()) {
        liveness_csr[tid] = at::native::fn_memorymanager.relu_thru;
      } else {
        liveness_csr[tid] = false;
      }
      at::native::fn_memorymanager.relu_thru = false;

      if (at::native::fn_memorymanager.is_fp16() && t.element_size() >= 4) {
        liveness_fp[tid] = true;
      } else {
        liveness_fp[tid] = false;
      }

      target_tensor[tid] = t.FN_to(opt, false, liveness_csr[tid], true, c10::MemoryFormat::Contiguous);
      target_tensor_valid[tid] = true;

      while (at::native::fn_memorymanager.event_arr_d2h[tid]) {
        if (at::native::fn_memorymanager.is_using_ssd()) {
          at::native::fn_memorymanager.Arcp2pCompletion();
        }
      }

      lifetime_forward[tid] = oid;

      if (at::native::fn_memorymanager.is_debug())
        std::cout << "Demand offload oid: " << oid << ", tid: " << tid << ", csr: " << liveness_csr[tid] << ", " << at::native::fn_memorymanager.event_arr_d2h[tid] << std::endl;

    } else {
      if (at::native::fn_memorymanager.liveness_result[tid]) {
        if (lifetime_forward[tid] == oid) {
          target_tensor[tid] = t.FN_to(opt, false, liveness_csr[tid], true, c10::MemoryFormat::Contiguous);
          target_tensor_valid[tid] = true;
        }
      } else {
        *fetch_loc = SavedVariable(t, isOutput);
        return;
      }

      if (at::native::fn_memorymanager.is_debug())
        std::cout << "Offload oid: " << oid << ", tid: " << tid << ", csr: " << liveness_csr[tid] << ", " << at::native::fn_memorymanager.event_arr_d2h[tid] << std::endl;
    }
  }

  if (at::native::fn_memorymanager.is_using_ssd())
    at::native::fn_memorymanager.Arcp2pCompletion();

  if (at::globalContext().FNGlobal.isOnDemand() && at::globalContext().FNGlobal.isForward()) {
    at::native::fn_memorymanager.timeStart();
  }
}

void FlashNeuronEngine::joinOffload() {
  if (at::globalContext().FNGlobal.isOnDemand()) {
    last_time_slot = at::native::fn_memorymanager.timeEnd();
  }

  pref_order = at::globalContext().FNGlobal.getBackwardOrder();
  pref_order_last = at::globalContext().FNGlobal.getBackwardLastIdx();
  pref_order_idx = 0;
}

void FlashNeuronEngine::preFetchSync(Oid oid, bool isOutput) {
  if (oid == 0 || !at::native::fn_memorymanager.is_fn() ||
      (op_tensor_list.find(oid) == op_tensor_list.end())) {
    return;
  }

  cudaStreamSynchronize(at::native::fn_memorymanager.fn_stream);

  auto fetch_vec = op_tensor_list[oid];
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto fetch_loc = it->first;
    auto tid = it->second;

    if (is_tensor_prefetched[tid] == true) {
      *fetch_loc = SavedVariable(target_tensor[tid], isOutput);
      continue;
    }

    if (target_tensor_valid[tid] == false) {
      std::cerr << "sync tensor dictionary lookup miss" << std::endl;
      return;
    }

    while (!is_tensor_prefetched[tid]) {
      if (at::native::fn_memorymanager.is_debug())
        std::cout << "preFetchSync waiting" << std::endl;

      // volatile at::native::p2pdsa_cpl *p_flu_cpl = (volatile at::native::p2pdsa_cpl *)at::native::fn_memorymanager.get_cpl_addr(tid, at::native::p2pdsa_gputossd);
      // volatile at::native::p2pdsa_cpl *p_pre_cpl = (volatile at::native::p2pdsa_cpl *)at::native::fn_memorymanager.get_cpl_addr(tid, at::native::p2pdsa_ssdtogpu);

      void* fp_result = at::native::fn_memorymanager.get_fp16_addr(tid);
      size_t numel = at::native::fn_memorymanager.get_numel(tid);
      int resize = at::native::fn_memorymanager.get_resize(tid);
      int elem = at::native::fn_memorymanager.get_elem(tid);

      if (at::native::fn_memorymanager.is_using_ssd()) {
/*
        if (p_pre_cpl != NULL && false == p_pre_cpl->requested) {
          int resize = at::native::fn_memorymanager.get_resize(tid);
          size_t bit_elements, pos_elements, pos_elements_before;
          bit_elements = (size_t)((numel + 1024 - 1) / 1024) * 32;
          pos_elements_before = (size_t)((numel + 32 - 1) / 32);
          int count = 0;
          while (pos_elements_before != 0) {
            pos_elements_before = pos_elements_before >> 1;  count++;
          }
          pos_elements = 1 << count;

          cudaStreamSynchronize(str);

          if (at::native::fn_memorymanager.is_csr() && resize > 0) {
            void* bit = at::native::fn_memorymanager.get_bit_addr(tid);
            void* pos = at::native::fn_memorymanager.get_pos_addr(tid);

            at::native::fn_memorymanager.p2p_free(fp16, sizeof(__half) * resize);
            at::native::fn_memorymanager.p2p_free(bit, sizeof(unsigned int) * bit_elements);
            at::native::fn_memorymanager.p2p_free(pos, sizeof(unsigned int) * pos_elements);
          } else if (at::native::fn_memorymanager.is_fp16() && resize == 0) {
            at::native::fn_memorymanager.p2p_free(fp16, sizeof(__half) * numel);
          } else {
            at::native::fn_memorymanager.p2p_free(fp16, numel);
          }

          delete p_flu_cpl;
          delete p_pre_cpl;

          at::native::fn_memorymanager.set_cpl_addr(tid, at::native::p2pdsa_gputossd, NULL);
          at::native::fn_memorymanager.set_cpl_addr(tid, at::native::p2pdsa_ssdtogpu, NULL);

          is_tensor_prefetched[tid] = true;
        }
        at::native::fn_memorymanager.Arcp2pCompletion();
*/
      } else {
        if (target_tensor[tid].device().type() == c10::DeviceType::CUDA) {
          if (fp_result != NULL) {
            if (resize != -1)
              at::native::fn_memorymanager.p2p_free(fp_result, elem * resize);
            else
              at::native::fn_memorymanager.p2p_free(fp_result, elem * numel);
          }

          size_t bit_elements, pos_elements, pos_elements_before;
          void* bit = at::native::fn_memorymanager.get_bit_addr(tid);
          void* pos = at::native::fn_memorymanager.get_pos_addr(tid);

          if (bit != NULL) {
            bit_elements = (size_t)((numel + 1024 - 1) / 1024) * 32;
            at::native::fn_memorymanager.p2p_free(bit, sizeof(unsigned int) * bit_elements);
          }

          if (pos != NULL) {
            pos_elements_before = (size_t)((numel + 32 - 1) / 32);
            int count = 0;
            while (pos_elements_before != 0) {
              pos_elements_before = pos_elements_before >> 1;  count++;
            }
            pos_elements = 1 << count;

            at::native::fn_memorymanager.p2p_free(pos, sizeof(unsigned int) * pos_elements);
          }

          is_tensor_prefetched[tid] = true;
          at::native::fn_memorymanager.event_arr_h2d[tid] = false;

        }
      }
    }

    *fetch_loc = SavedVariable(target_tensor[tid], isOutput);
  }
}

void FlashNeuronEngine::insertToPFDict_(Oid oid, SavedVariable* loc, Tid tid) {
  auto it = op_tensor_list.find(oid);
  if (it != op_tensor_list.end()) {
    op_tensor_list[oid].emplace_back(loc, tid);
  } else {
    std::vector<PFInfo> tmp;
    tmp.emplace_back(loc, tid);
    op_tensor_list.insert(std::pair<Oid, std::vector<PFInfo>>(oid, tmp));
  }
}

void FlashNeuronEngine::dropTensor(Oid oid) {
  if (op_tensor_list.find(oid) == op_tensor_list.end()) {
    return;
  }

  auto fetch_vec = op_tensor_list[oid];
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto fetch_loc = it->first;
    auto tid = it->second;
    std::cout <<  "dropTensor try: " << tid << std::endl;

    if (at::globalContext().FNGlobal.isOnDemand()) {
      at::Tensor& tref = target_tensor[tid];
      c10::TensorOptions opt = c10::TensorOptions();
      opt = opt.device(c10::Device(c10::DeviceType::CPU));
      opt = opt.dtype(tref.dtype());
      opt = opt.pinned_memory(true);

      while (at::native::fn_memorymanager.event_arr_h2d[tid]) {
        if (at::native::fn_memorymanager.is_debug()) {
          std::cout << "dropTensor event_arr_h2d: " <<  tid << std::endl;
          break;
        }

        if (at::native::fn_memorymanager.is_using_ssd()) {
//          at::native::fn_memorymanager.Arcp2pCompletion();
        }
      }

      if (at::native::fn_memorymanager.is_using_ssd()) {
/*
        at::native::fn_memorymanager.set_dir(tid, at::native::p2pdsa_gputossd);
        at::native::p2pdsa_cpl *p_cpl = new at::native::p2pdsa_cpl;
        at::native::fn_memorymanager.set_cpl_addr(tid, at::native::p2pdsa_gputossd, (void *)p_cpl);
*/
      }

      tref = tref.FN_to(opt, false, false, true, c10::MemoryFormat::Contiguous);
      is_tensor_prefetched[tid] = false;

      while (at::native::fn_memorymanager.event_arr_d2h[tid]) {
        if (at::native::fn_memorymanager.is_debug()) {
          std::cout << "dropTensor event_arr_d2h: " <<  tid << std::endl;
          break;
        }

        if (at::native::fn_memorymanager.is_using_ssd()) {
//          at::native::fn_memorymanager.Arcp2pCompletion();
        }
      }
    } else {
      // int cur_back_num = at::globalContext().FNGlobal.curBackNum();
      if ((oid == lifetime_backward[tid]) && target_tensor_valid[tid]) {
        target_tensor_valid[tid] = false;
        fetch_loc->reset_data();
        c10::cuda::CUDACachingAllocator::emptyCache();

        if (at::native::fn_memorymanager.is_debug())
          std::cout << "Tensor drop: " << tid << ", " << oid << std::endl;
      }
    }
  }
}

bool FlashNeuronEngine::preFetch(Oid try_oid) {
  Oid oid = -1;

  if (try_oid == 0) {
    return false;
  } else if (try_oid == -1) {
    if (pref_order_last >= pref_order_idx) {
      oid = pref_order[pref_order_idx];
    }
  } else {
    oid = try_oid;
  }

  if (at::native::fn_memorymanager.is_debug())
    std::cout << "Prefetch try oid: " << oid << std::endl;

  if (op_tensor_list.find(oid) == op_tensor_list.end()) {
    return true;
  }

  if (at::globalContext().FNGlobal.isOnDemand()) {
    at::globalContext().FNGlobal.pushBackwardOrder(oid);
  }

  auto fetch_vec = op_tensor_list[oid];

  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;

    if (target_tensor_valid[tid] == false) {
      continue;
    }

    at::Tensor& tref = target_tensor[tid];
    c10::TensorOptions opt = c10::TensorOptions();
    opt = opt.device(c10::Device(c10::DeviceType::CUDA));
    opt = opt.dtype(tref.dtype());

    if (tref.device().type() == c10::DeviceType::CPU) {
      if (at::native::fn_memorymanager.is_using_ssd()) {
/*
        if (at::globalContext().FNGlobal.isOnDemand()) {
          while (at::native::fn_memorymanager.event_arr_d2h[tid]) {
            at::native::fn_memorymanager.Arcp2pCompletion();
          }
        } else {
          if (at::native::fn_memorymanager.event_arr_d2h[tid]) {
            return false;
          }
        }
        at::native::fn_memorymanager.set_dir(tid, at::native::p2pdsa_ssdtogpu);
        at::native::p2pdsa_cpl *p_cpl = new at::native::p2pdsa_cpl;
        p_cpl->requested = true;
        at::native::fn_memorymanager.set_cpl_addr(tid, at::native::p2pdsa_ssdtogpu, (void *)p_cpl);
*/
      }

      if (at::native::fn_memorymanager.is_debug())
        std::cout << "Prefetch oid: " << oid << ", tid: " << tid << ", " << at::native::fn_memorymanager.event_arr_h2d[tid] << std::endl;

      tref = tref.FN_to(opt, false, liveness_csr[tid], true,  c10::MemoryFormat::Contiguous);

      if (at::globalContext().FNGlobal.isOnDemand()) {
        lifetime_backward[tid] = oid;
        cudaStreamSynchronize(at::native::fn_memorymanager.fn_stream);

        while (at::native::fn_memorymanager.event_arr_h2d[tid]) {
          at::native::fn_memorymanager.Arcp2pCompletion();
        }
      }
    }
  }

  return true;
}

void FlashNeuronEngine::resetCppEngine() {
  for(auto i = 0; i < NUM_TENSOR; i ++) {
    if (is_tensor_prefetched[i] == true) {
      target_tensor[i].reset();
    }
  }

  memset(target_tensor_valid, 0, sizeof(bool) * NUM_TENSOR);
  memset(is_tensor_offloaded, 0, sizeof(bool) * NUM_TENSOR);
  memset(is_tensor_prefetched, 0, sizeof(bool) * NUM_TENSOR);

  memset(liveness_time, 0, sizeof(double) * NUM_TENSOR);
  memset(liveness_size, 0, sizeof(double) * NUM_TENSOR);

  op_tensor_list.clear();
  // stream_occupied.clear();

  // at::globalContext().FNGlobal.resetGlobalTid();
  // at::globalContext().FNGlobal.resetGlobalOid();

  double accum_sum = 0;
  for(int i = 0; i < NUM_TENSOR; i++) {
    if (at::native::fn_memorymanager.feature_map_accum[i] > 0) {
      accum_sum += at::native::fn_memorymanager.feature_map_accum[i];
    }

    at::native::fn_memorymanager.feature_map_accum[i] = 0;
  }

  at::native::fn_memorymanager.gradient_map_accum = 0;
  at::native::fn_memorymanager.weight_accum = 0;
  at::native::fn_memorymanager.misc_accum = 0;
}

}} // namespace torch::autograd
