#pragma once

#include <ATen/ATen.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

// For p2p library
#include <c10/core/Storage.h>
#include <dlfcn.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/autograd/saved_variable.h>

#define BLK_SZ ((size_t)1 << 12)
#define NUM_TENSOR 32768
#define NUM_OP 32768

using namespace std;


namespace at { namespace native {

using namespace at::cuda;

/*
typedef struct {
  uint64_t tid;
  uint64_t numel;
  uint64_t ntpb;
  void *dst;
  void *src;
  void *ptr;
} arcp2p_info;

#include "arcp2p.h"
using arcp2p_type1_fn = arcp2p * (*)(const char *[PATH_LENGTH], int);
using arcp2p_type2_fn = int      (*)(arcp2p *);
using arcp2p_type3_fn = int      (*)(arcp2p *, uint64_t, uint64_t);
using arcp2p_type4_fn = int      (*)(arcp2p *, uint64_t, uint64_t, uint64_t, arcp2p_cpl *, arcp2p_dir);
*/

class FN_memory {
 public:
  FN_memory();
  ~FN_memory();

  cudaStream_t fn_stream;
  cudaEvent_t startEvent;
  cudaEvent_t endEvent;

  int global_tensor_id_;
  // int cur_back_num;
  bool liveness_result[NUM_TENSOR] = {false};

  bool hard_training;
  bool relu_thru;
  bool mapping;
  bool* event_arr_d2h;
  bool* event_arr_h2d;

  void device_malloc(void** gpu_ptr, size_t size);
  void device_malloc_reverse(void** gpu_ptr, size_t size);

  void device_free(void* addr, size_t size);

  size_t device_occupancy_size();
  size_t p2p_occupancy_size();

  double device_occupancy();
  double device_occupancy_future(size_t size);
  double p2p_occupancy();

  void p2p_malloc(void** gpu_ptr, size_t size);
  void p2p_free(void* addr, size_t size);

  void* get_device_addr();
  uint64_t get_device_sz();

  void* get_fp16_addr(int tid);
  void set_fp16_addr(int tid, uint64_t addr);

  void* get_bit_addr(int tid);
  void set_bit_addr(int tid, uint64_t addr);

  void* get_pos_addr(int tid);
  void set_pos_addr(int tid, uint64_t addr);

  int get_resize(int tid);
  void set_resize(int tid, int resize);

  size_t get_numel(int tid);
  void set_numel(int tid, size_t numel);

  int get_elem(int tid);
  void set_elem(int tid, int elem);

/*
  void* get_cpl_addr(int tid, arcp2p_dir dir);
  void set_cpl_addr(int tid, arcp2p_dir dir, void *addr);

  uint64_t* get_offset_ptr(int tid);

  arcp2p_dir get_dir(int tid);
  void set_dir(int tid, arcp2p_dir dir);
*/

  // [JS] P2P library
  void Arcp2pSetting(int flags);
//  int  Arcp2pBarMapping(uint64_t, uint64_t);
//  void Arcp2pSubmission(uint64_t, uint64_t, uint64_t *, arcp2p_cpl *, arcp2p_dir, c10::Storage *, arcp2p_info *, cudaStream_t);
//  bool Arcp2pReqEmpty();
  void Arcp2pCompletion(bool prefCall);
//  void Arcp2pSynchronize();

  // Flag check
  bool is_timer();
  bool is_fn();
  bool is_fp16();
  bool is_csr();
  bool is_using_ssd();
  bool is_debug();

  float runTime;
  void timeStart();
  float timeEnd();

  int* pref_it;
  int pref_end;
  int pref_idx;

  int on_the_fly;

  double dev_freeBlk;
  double p2p_freeBlk;

  double feature_map_accum[NUM_TENSOR] = {0};
  double gradient_map_accum;
  double weight_accum;
  double misc_accum;

  size_t init_4m;
  size_t init_8m;
  size_t init_16m;
  size_t init_32m;
  size_t init_64m;
  size_t init_128m;

  struct timeval tv1, tv2;
  struct timeval startTime, endTime;

 private:
  void* deviceAddr;
//  bool* deviceTable;
  short* deviceTable;
  int* devStartBlk;
  int* devStartBlk_rev;
  unsigned int* device_page_map;
  unsigned int* device_page_map_rev;
  int devMaxBlk;

  void* p2pAddr;
  bool* p2pTable;
  int* p2pStartBlk;
  unsigned int* p2p_page_map;
  int p2pMaxBlk;

  int devBlk_0_4;
  int devBlk_4_16;
  int devBlk_16_64;
  int devBlk_64_128;
  int devBlk_128;

  int devBlk_0_4_rev;
  int devBlk_4_16_rev;
  int devBlk_16_64_rev;
  int devBlk_64_128_rev;
  int devBlk_128_rev;

  int p2pBlk_0_4;

  uint64_t* fp16_ptr_arr;
  uint64_t* bit_ptr_arr;
  uint64_t* pos_ptr_arr;
  int* resize_arr;
  size_t* numel_arr;
  int* elem_arr;
/*
  uint64_t* cpl_flu_ptr_arr;
  uint64_t* cpl_pre_ptr_arr;
  uint64_t* offset_arr;
  arcp2p_dir* dir_arr;

  // [JS] P2P library
  arcp2p_type1_fn arcp2p_initialize;
  arcp2p_type2_fn arcp2p_release;
  arcp2p_type3_fn arcp2p_bar_attach;
  arcp2p_type2_fn arcp2p_bar_detach;
  arcp2p_type4_fn arcp2p_transfer;
  arcp2p_type2_fn arcp2p_completion;
  arcp2p_type2_fn arcp2p_synchronize;

  arcp2p *arc_handle;
  uint64_t last_allocated_offset;
*/

  bool isTimer;
  bool isFN;
  bool isFP16;
  bool isCSR;
  bool isUsingSSD;
  bool isTesla;
  bool isDebug;

  uint64_t device_sz;
  uint64_t max_device;

  uint64_t p2p_sz;
  uint64_t max_p2p;
};

extern FN_memory fn_memorymanager;

}} // namespace at::native
