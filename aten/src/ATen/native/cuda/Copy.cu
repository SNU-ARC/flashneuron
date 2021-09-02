#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <THC/THC.h>

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_version.h>
#endif


#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#include <ATen/native/cuda/flashneuron.h>
#include <c10/cuda/CUDACachingAllocator.h>

#define nTPB 512
#define per_threads 256
#define nthreads 256
#define nblocks 256

#define find(n) (32 * (unsigned int)(n / 1024) + (n % 32))
#define mask(n) (0x80000000 >> (unsigned int)((n % 1024) / 32))


namespace at {
namespace native {

using namespace at::cuda;

__global__ void half_scale(float *din, __half *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize)  dout[idx] = __float2half(din[idx]);
}

__global__ void float_scale(__half *din, float *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize)  dout[idx] = __half2float(din[idx]);
}

__global__ void double_scale(__half *din, double *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize)  dout[idx] = (double)__half2float(din[idx]);
}

__global__ void zero_mask(float *din, unsigned int *bit, unsigned int *pos, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize) {
    if (din[idx] != 0.0f) {
      atomicAdd(&bit[find(idx)], mask(idx));
      atomicAdd(&pos[(unsigned int)(idx / 32)], 1);
    }
  }
}

__global__ void pos_first(unsigned int* pos, int asize) {
  int total_idx = nblocks * nthreads;

  for (int j = 0; j < (asize / per_threads / total_idx + 1); j++) {
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if ((global_idx + 1) * per_threads - 1 <= asize) {
      for (int i = 0; i < per_threads; i++) {
        int idx = global_idx * per_threads + i;
        if (idx % per_threads != 0) {
          pos[idx] += pos[idx - 1];
        }
      }
    }
  }
}

__global__ void pos_second(unsigned int* pos, unsigned int* opos, int asize) {
  int total_idx = nblocks * nthreads;

  for (int j = 0; j < (asize / per_threads / total_idx + 1); j++) {
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if ((global_idx + 1) * per_threads - 1 <= asize) {
      unsigned int temp = 0;

      for (int i = 0; i < global_idx; i++) {
        int idx = (i + 1) * per_threads - 1;
        temp += pos[idx];
      }

      for (int i = 0; i < per_threads; i++) {
        int idx = (global_idx) * per_threads + i;
        opos[idx] = pos[idx] + temp;
      }
    }
  }
}

__global__ void zero_insert_double(unsigned int *bit, unsigned int *nz_pos, float* din, double *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize) {
    int count = -1;
    if ((unsigned int)(bit[find(idx)] & mask(idx)) > 0) {
      for (int i = (int)(idx / 32) * 32; i < idx + 1; i++) {
        unsigned int mask = bit[find(i)] & mask(i);
        if (mask > 0)  count += 1;
      }
    }

    if (count == -1)  dout[idx] = 0.0;
    else {
      if ((unsigned int)(idx / 32) == 0) {
        dout[idx] = (double)din[count + 0];
      } else {
        dout[idx] = (double)din[count + nz_pos[(unsigned int)(idx / 32) - 1]];
      }
    }
  }
}

__global__ void zero_insert_float(unsigned int *bit, unsigned int *nz_pos, float* din, float *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize) {
    int count = -1;
    if ((unsigned int)(bit[find(idx)] & mask(idx)) > 0) {
      for (int i = (int)(idx / 32) * 32; i < idx + 1; i++) {
        unsigned int mask = bit[find(i)] & mask(i);
        if (mask > 0)  count += 1;
      }
    }

    if (count == -1)  dout[idx] = 0.0f;
    else {
      if ((unsigned int)(idx / 32) == 0) {
        dout[idx] = din[count + 0];
      } else {
        dout[idx] = din[count + nz_pos[(unsigned int)(idx / 32) - 1]];
      }
    }
  }
}

struct is_not_zero {
  __host__ __device__
  bool operator()(const float x) {
    return (x != 0);
  }
};

struct is_not_zero_double {
  __host__ __device__
  bool operator()(const double x) {
    return (x != 0);
  }
};

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter, bool non_blocking) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool memcpy_eligible = same_type && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  CUDAGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // cudaMemcpyAsync on the default stream.
  CUDAStream copy_stream = getCurrentCUDAStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    CUDAEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(getCurrentCUDAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    void *dst = iter.data_ptr(0);
    void *src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Perform the copy
      AT_CUDA_CHECK(cudaMemcpyAsync(
          dst, src, size,
          cudaMemcpyDeviceToDevice,
          copy_stream));
    }
  } else {
    auto dtype = iter.dtype(0);
    if (isQIntType(dtype)) {
      AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
      });
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          kHalf, kBool, kBFloat16, dtype, "copy_", [&] {
            gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
          });
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    CUDAEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(getCurrentCUDAStream(dst_device.index()));
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_cuda() && src_device.is_cuda());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use cudaMemcpyAsync
    return false;
  } else if (dst_device.is_cuda() && src_device.is_cuda()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return THCState_getPeerToPeerAccess(
        globalContext().getTHCState(), src_device.index(), dst_device.index());
}

static void copy_kernel_cuda(TensorIterator& iter, bool non_blocking) {
  AT_ASSERT(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // Type conversions are performed on the CPU for CPU-GPU copies and on
    // the src device for GPU-GPU copies.
    if (iter.device_type(0) == kCUDA) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_cuda() && src_device.is_cuda()) {
    copy_device_to_device(iter, non_blocking);
    return;
  }

  // Copy between CPU and GPU
  cuda::OptionalCUDAGuard device_guard;
  cudaMemcpyKind kind;
  if (dst_device.is_cuda() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = cudaMemcpyHostToDevice;
  } else if (dst_device.is_cpu() && src_device.is_cuda()) {
    device_guard.set_device(src_device);
    kind = cudaMemcpyDeviceToHost;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  CUDAStream stream = getCurrentCUDAStream();

  if (non_blocking) {
    AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    void* ptr = (dst_device == kCPU ? dst : src);
    AT_CUDA_CHECK(THCCachingHostAllocator_recordEvent(ptr, stream));
  } else {
#if HIP_VERSION >= 301
    AT_CUDA_CHECK(hipMemcpyWithStream(dst, src, nbytes, kind, stream));
#else
    AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  }
}

static void FN_copy_kernel_cuda(TensorIterator& iter, bool non_blocking, int tid, bool is_csr) {
  /* SSD flag only concerned if 'dir' information is properly saved at fn_memorymanager
   * only this case is from offload and prefetch.
   */
  // arcp2p_dir dir = fn_memorymanager.get_dir(tid);
  // bool ssd_flag  = fn_memorymanager.is_using_ssd() && (dir != arcp2p_unused);
  bool ssd_flag  = false;

  /* fp16 & csr option is delivered by flag setting
   * Note. FP16 should be set when csr is set.
   *       So we don't case about FP16=false & CSR=true case
   */
  bool fp16_flag = fn_memorymanager.is_fp16();
  bool csr_flag  = fn_memorymanager.is_csr();

  // Clear dir value, to avoid confusion
  // fn_memorymanager.set_dir(tid, arcp2p_unused);

  AT_ASSERT(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it invovles the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  // Copy between CPU and GPU
  cuda::OptionalCUDAGuard device_guard;
  cudaMemcpyKind kind;
  if (dst_device.is_cuda() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = cudaMemcpyHostToDevice;
    fn_memorymanager.event_arr_h2d[tid] = true;
  } else if (dst_device.is_cpu() && src_device.is_cuda()) {
    device_guard.set_device(src_device);
    kind = cudaMemcpyDeviceToHost;
    fn_memorymanager.event_arr_d2h[tid] = true;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  uint64_t p2p_addr = 0, p2p_size = 0;

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  cudaStream_t stream = at::native::fn_memorymanager.fn_stream;

  fn_memorymanager.set_elem(tid, iter.element_size(0));

  if (true == ssd_flag) {
/*
    if (!fn_memorymanager.mapping) {
      // [TODO] this should be called only for Tesla option enabled
      void* deviceAddr = fn_memorymanager.get_device_addr();
      uint64_t deviceSz = fn_memorymanager.get_device_sz();
      fn_memorymanager.Arcp2pBarMapping((uint64_t)deviceAddr, deviceSz);
      fn_memorymanager.mapping = true;
    }
*/
  }

  size_t bit_elements, pos_elements, pos_elements_before;

  if (csr_flag) {
    bit_elements = (size_t)((iter.numel() + 1024 - 1) / 1024) * 32;
    pos_elements_before = (size_t)((iter.numel() + 32 - 1) / 32);
    int count = 0;
    while (pos_elements_before != 0) {
      pos_elements_before = pos_elements_before >> 1;  count++;
    }
    pos_elements = 1 << count;
  }

  if (kind == cudaMemcpyDeviceToHost) {
    if (iter.element_size(0) >= 4) {
      if (csr_flag && is_csr) {
        void *fp16, *bit, *pos;
        fn_memorymanager.p2p_malloc(&bit, sizeof(unsigned int) * bit_elements);
        fn_memorymanager.p2p_malloc(&pos, sizeof(unsigned int) * pos_elements);

        fn_memorymanager.set_bit_addr(tid, (uint64_t)bit);
        fn_memorymanager.set_pos_addr(tid, (uint64_t)pos);

        unsigned int *nz_pos;
        fn_memorymanager.p2p_malloc((void **)&nz_pos, pos_elements * sizeof(unsigned int));

        cudaMemsetAsync((void *)bit, 0, sizeof(unsigned int) * bit_elements, stream);
        cudaMemsetAsync((void *)pos, 0, sizeof(unsigned int) * pos_elements, stream);
        cudaMemsetAsync((void *)nz_pos, 0, sizeof(unsigned int) * pos_elements, stream);

        void *nz_src;
        if (iter.element_size(0) == 8) {
          fn_memorymanager.p2p_malloc((void **)&nz_src, iter.numel() * sizeof(double));
          cudaMemsetAsync((void *)nz_src, 0, sizeof(double) * iter.numel(), stream);
          thrust::device_ptr<double> dA_V((double *)src);
          thrust::device_ptr<double> dA_R((double *)nz_src);
          thrust::copy_if(dA_V, dA_V + iter.numel(), dA_R, is_not_zero_double());
        } else {
          fn_memorymanager.p2p_malloc((void **)&nz_src, iter.numel() * sizeof(float));
          cudaMemsetAsync((void *)nz_src, 0, sizeof(float) * iter.numel(), stream);
          thrust::device_ptr<float> dA_V((float *)src);
          thrust::device_ptr<float> dA_R((float *)nz_src);
          thrust::copy_if(dA_V, dA_V + iter.numel(), dA_R, is_not_zero());
        }

        zero_mask<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((float *)src, (unsigned int *)bit, nz_pos, iter.numel());

        pos_first<<<nblocks, nthreads, 0, stream>>>(nz_pos, pos_elements);
        pos_second<<<nblocks, nthreads, 0, stream>>>(nz_pos, (unsigned int*)pos, pos_elements);

        int resize = 0;

        cudaMemcpyAsync((void *)&resize, (void *)((size_t)pos + sizeof(unsigned int) * (pos_elements - 1)),
            sizeof(int), cudaMemcpyDeviceToHost, stream);

        fn_memorymanager.p2p_malloc(&fp16, sizeof(__half) * resize);
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);

        half_scale<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((float *)nz_src, (__half *)fp16, resize);

        fn_memorymanager.set_resize(tid, resize);
        fn_memorymanager.set_numel(tid, iter.numel());

        if (true == ssd_flag) {
/*
          p2p_addr = (uint64_t)fp16;
          p2p_size = (uint64_t)(resize * sizeof(__half));
*/
        } else {
          AT_CUDA_CHECK(cudaMemcpyAsync(dst, fp16, resize * sizeof(__half), kind, stream));

          fn_memorymanager.p2p_free(fp16, resize * sizeof(__half));
          fn_memorymanager.event_arr_d2h[tid] = false;
        }

        fn_memorymanager.p2p_free((void *)nz_pos, pos_elements * sizeof(unsigned int));
        fn_memorymanager.p2p_free((void *)nz_src, iter.numel() * sizeof(float));
      } else if (fp16_flag) {
        // this case include both cases
        // 1. csr_flag==true && is_csr==false (csr_flag==true always guarantee fp16_flag==true)
        // 2. csr_flag==false && fp16_flag==true

        // keep print message for debug purpose
        void *fp16;
        fn_memorymanager.p2p_malloc(&fp16, sizeof(__half) * iter.numel());
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);

        half_scale<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((float *)src, (__half *)fp16, iter.numel());

        fn_memorymanager.set_resize(tid, 0); // [TODO] slight hack code, we will distinguish CSR / FP16 by resize value
        fn_memorymanager.set_numel(tid, iter.numel());

        if (true == ssd_flag) {
/*
          p2p_addr = (uint64_t)fp16;
          p2p_size = (uint64_t)(iter.numel() * sizeof(__half));
*/
        } else {

          AT_CUDA_CHECK(cudaMemcpyAsync(dst, fp16, sizeof(__half) * iter.numel(), kind, stream));

          fn_memorymanager.p2p_free(fp16, iter.numel() * sizeof(__half));
          fn_memorymanager.event_arr_d2h[tid] = false;
        }
      } else { // false == csr_flag && false == fp16_flag
        if (true == ssd_flag) {
/*
          // TODO Need to malloc src ptr to BAR attached region
          void *fp16;
          fn_memorymanager.p2p_malloc(&fp16, nbytes);
          fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);
          fn_memorymanager.set_numel(tid, (size_t)nbytes);
          fn_memorymanager.set_resize(tid, -1); // [TODO] slight hack code, we will distinguish CSR / FP16 by resize value
          AT_CUDA_CHECK(cudaMemcpyAsync(fp16, src, nbytes, cudaMemcpyDeviceToDevice, stream));

          p2p_addr = (uint64_t)fp16;
          p2p_size = (uint64_t)nbytes;
*/
        } else {

          fn_memorymanager.set_resize(tid, -1); // [TODO] slight hack code, we will distinguish CSR / FP16 by resize value
          fn_memorymanager.set_numel(tid, iter.numel());
          fn_memorymanager.set_fp16_addr(tid, (uint64_t)NULL);

          AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));

          fn_memorymanager.event_arr_d2h[tid] = false;
        }
      }
    } else { // Non double or float
      if (true == ssd_flag) {
/*
        // TODO Need to malloc src ptr to BAR attached region
        void *fp16;
        fn_memorymanager.p2p_malloc(&fp16, nbytes);
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);
        fn_memorymanager.set_resize(tid, -1); // [TODO] slight hack code, we will distinguish CSR / FP16 by resize value
        fn_memorymanager.set_numel(tid, (size_t)nbytes);
        AT_CUDA_CHECK(cudaMemcpyAsync(fp16, src, nbytes, cudaMemcpyDeviceToDevice, stream));

        p2p_addr = (uint64_t)fp16;
        p2p_size = (uint64_t)nbytes;
*/
      } else {
        fn_memorymanager.set_resize(tid, -1); // [TODO] slight hack code, we will distinguish CSR / FP16 by resize value
        fn_memorymanager.set_numel(tid, iter.numel());
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)NULL);

        AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));

        fn_memorymanager.event_arr_d2h[tid] = false;
      }
    }
  }

  if (kind == cudaMemcpyHostToDevice) {
    if (iter.element_size(0) >= 4) {
      if (csr_flag && is_csr) {
        void* bit = fn_memorymanager.get_bit_addr(tid);
        void* pos = fn_memorymanager.get_pos_addr(tid);

        int resize = fn_memorymanager.get_resize(tid);

        void* fp16;
        fn_memorymanager.p2p_malloc(&fp16, sizeof(__half) * resize);
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);

        if (ssd_flag) {
/*
          p2p_addr = (uint64_t)fp16;
          p2p_size = (uint64_t)(resize * sizeof(__half));
          // [JS] all backend job will be called at Arcp2pCompletion
*/
        } else {
          float *nz_dst;
          fn_memorymanager.p2p_malloc((void **)&nz_dst, resize * sizeof(float));
          cudaMemsetAsync((void *)nz_dst, 0, resize * sizeof(float), stream);

          AT_CUDA_CHECK(cudaMemcpyAsync(fp16, src, resize * sizeof(__half), kind, stream));

          float_scale<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((__half *)fp16, nz_dst, resize);
//          float_scale<<<(resize + nTPB - 1) / nTPB, nTPB, 0, stream>>>((__half *)fp16, nz_dst, resize);

          if (iter.element_size(0) == 8) {
            zero_insert_double<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((unsigned int*)bit, (unsigned int*)pos, nz_dst, (double *)dst, iter.numel());
          } else {
            zero_insert_float<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((unsigned int*)bit, (unsigned int*)pos, nz_dst, (float *)dst, iter.numel());
          }

          fn_memorymanager.p2p_free((void *)nz_dst, resize * sizeof(float));
        }
      } else if (fp16_flag) {
        // keep print message for debug purpose
        void* fp16;
        fn_memorymanager.p2p_malloc(&fp16, sizeof(__half) * iter.numel());
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);
        fn_memorymanager.set_numel(tid, iter.numel());
        fn_memorymanager.set_resize(tid, 0);

        if (ssd_flag) {
/*
          p2p_addr = (uint64_t)fp16;
          p2p_size = (uint64_t)(iter.numel() * sizeof(__half));
*/
        } else {
          AT_CUDA_CHECK(cudaMemcpyAsync(fp16, src, iter.numel() * sizeof(__half), kind, stream));
          if (iter.element_size(0) == 8) {
            double_scale<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((__half* )fp16, (double*)dst, iter.numel());
          } else {
            float_scale<<<(iter.numel() + nTPB - 1) / nTPB, nTPB, 0, stream>>>((__half* )fp16, (float*)dst, iter.numel());
          }

          if (at::globalContext().FNGlobal.isOnDemand()) {
            cudaStreamSynchronize(stream);
            fn_memorymanager.event_arr_h2d[tid] = false;
          }
        }
      } else {
        if (true == ssd_flag) {
/*
          void* fp16;
          fn_memorymanager.p2p_malloc(&fp16, nbytes);
          fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);

          p2p_addr = (uint64_t)fp16;
          p2p_size = (uint64_t)nbytes;
*/
        } else {

          fn_memorymanager.set_fp16_addr(tid, (uint64_t)NULL);

          AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));

          if (at::globalContext().FNGlobal.isOnDemand()) {
            cudaStreamSynchronize(stream);
            fn_memorymanager.event_arr_h2d[tid] = false;
          }
        }
      }
    } else {
      if (true == ssd_flag) {
/*
        void* fp16;
        fn_memorymanager.p2p_malloc(&fp16, nbytes);
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)fp16);

        p2p_addr = (uint64_t)fp16;
        p2p_size = (uint64_t)nbytes;
*/
      } else {
        fn_memorymanager.set_fp16_addr(tid, (uint64_t)NULL);

        AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
        if (at::globalContext().FNGlobal.isOnDemand()) {
          cudaStreamSynchronize(stream);
          fn_memorymanager.event_arr_h2d[tid] = false;
        }
      }
    }
  }

  if (ssd_flag) {
/*
    uint64_t *p_offs = fn_memorymanager.get_offset_ptr(tid);
    arcp2p_cpl *p_cpl = (arcp2p_cpl *)fn_memorymanager.get_cpl_addr(tid, dir);

    if (arcp2p_gputossd == dir) {
      c10::Storage *stor = nullptr;

      if (false == fp16_flag) {
        stor = new c10::Storage;
        *stor = iter.tensor(1).storage();
      }

      arcp2p_info *info = nullptr;

      info = new arcp2p_info;
      info->tid = (uint64_t)tid;
      info->ptr = fn_memorymanager.get_fp16_addr(tid);

      fn_memorymanager.Arcp2pSubmission(p2p_addr, p2p_size, p_offs, p_cpl, dir, stor, info, stream);
      fn_memorymanager.Arcp2pCompletion(false);
    } else if (arcp2p_ssdtogpu == dir) {
      arcp2p_info *info = nullptr;

      info = new arcp2p_info;
      info->tid = (uint64_t)tid;
      info->numel = (uint64_t)iter.numel();
      info->ntpb = nTPB;
      info->dst = iter.data_ptr(0);
      info->src = iter.data_ptr(1);
      info->ptr = fn_memorymanager.get_fp16_addr(tid);

      fn_memorymanager.Arcp2pSubmission(p2p_addr, p2p_size, p_offs, p_cpl, dir, nullptr, info, stream);
      fn_memorymanager.Arcp2pCompletion(false);
    }
*/
  }
}

REGISTER_DISPATCH(copy_stub, &copy_kernel_cuda);
REGISTER_DISPATCH(FN_copy_stub, &FN_copy_kernel_cuda);

} // namespace native
} // namespace at
