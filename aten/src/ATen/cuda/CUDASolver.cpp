#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDASolver.h>
#include <c10/cuda/CUDACachingAllocator.h>

#ifdef CUDART_VERSION

namespace at {
namespace cuda {
namespace solver {

template <>
void getrf<double>(
    cusolverDnHandle_t handle, int m, int n, double* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrf(
      handle, m, n, dA, ldda, static_cast<double*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<float>(
    cusolverDnHandle_t handle, int m, int n, float* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrf(
      handle, m, n, dA, ldda, static_cast<float*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<double>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(
      handle, m, n, reinterpret_cast<cuDoubleComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex) * lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrf(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(dA),
      ldda,
      static_cast<cuDoubleComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrf<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<float>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(
      handle, m, n, reinterpret_cast<cuComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuComplex) * lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrf(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex*>(dA),
      ldda,
      static_cast<cuComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrs<double>(
    cusolverDnHandle_t handle, int n, int nrhs, double* dA, int lda, int* ipiv, double* ret, int ldb, int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrs(
    handle, CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<float>(
    cusolverDnHandle_t handle, int n, int nrhs, float* dA, int lda, int* ipiv, float* ret, int ldb, int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrs(
    handle, CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<double>* dA,
    int lda,
    int* ipiv,
    c10::complex<double>* ret,
    int ldb,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrs(
      handle,
      CUBLAS_OP_N,
      n,
      nrhs,
      reinterpret_cast<cuDoubleComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<cuDoubleComplex*>(ret),
      ldb,
      info));
}

template <>
void getrs<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<float>* dA,
    int lda,
    int* ipiv,
    c10::complex<float>* ret,
    int ldb,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrs(
      handle,
      CUBLAS_OP_N,
      n,
      nrhs,
      reinterpret_cast<cuComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<cuComplex*>(ret),
      ldb,
      info));
}


template<>
void gesvdj<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<float*>(dataPtr.get()),
    lwork, info, params));
}

template<>
void gesvdj<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<double*>(dataPtr.get()),
    lwork, info, params));
}

template<>
void gesvdj<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    static_cast<cuComplex*>(dataPtr.get()),
    lwork, info, params));
}

template<>
void gesvdj<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    static_cast<cuDoubleComplex*>(dataPtr.get()),
    lwork, info, params));
}


template<>
void gesvdjBatched<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<float*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<double*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    static_cast<cuComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    static_cast<cuDoubleComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

} // namespace solver
} // namespace cuda
} // namespace at

#endif // CUDART_VERSION
