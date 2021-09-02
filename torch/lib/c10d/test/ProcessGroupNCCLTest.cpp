#include <iostream>

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/test/CUDATest.hpp>
#include <c10d/test/TestUtils.hpp>

#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/csrc/autograd/profiler.h>
#include <gtest/gtest.h>

using namespace c10d::test;

using at::cuda::CUDAStream;
using c10d::ProcessGroup;

class NCCLTestBase {
 public:
  NCCLTestBase(const std::string& path) : path_(path) {}

  NCCLTestBase(NCCLTestBase&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  ::c10d::ProcessGroupNCCL& getProcessGroup() {
    return *pg_;
  }

  void initialize(int rank, int size) {
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);

    pg_ = std::unique_ptr<::c10d::ProcessGroupNCCL>(
        new ::c10d::ProcessGroupNCCL(store, rank, size));
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupNCCL> pg_;
};

class NCCLTest : public NCCLTestBase {
 public:
  NCCLTest(const std::string& path, int worldSize)
      : NCCLTestBase(path),
        numDevices_(cudaNumDevices()),
        state_(::at::globalContext().lazyInitCUDA()),
        worldSize_(worldSize) {
    // Each device has a single tensor to perf the NCCL op
    tensors_.resize(numDevices_);
    inputs_.resize(numDevices_);
    outputs_.resize(numDevices_);
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; ++i) {
      deviceGuard.set_index(i);
      tensors_[i] = at::empty({3, 3}, at::kCUDA);
      inputs_[i].resize(worldSize_ * numDevices_);
      outputs_[i].resize(worldSize_ * numDevices_);
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        inputs_[i][j] = at::empty({3, 3}, at::kCUDA);
        outputs_[i][j] = at::empty({3, 3}, at::kCUDA);
      }
    }

    // Allocate a stream per device.
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    streams_.reserve(numDevices_);
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      streams_.push_back(at::cuda::getStreamFromPool());
    }
  }

  void wait(
      c10::intrusive_ptr<ProcessGroup::Work>& work,
      std::chrono::milliseconds timeout = kNoTimeout) {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    work->wait(timeout);
  }

  std::vector<at::Tensor> getTensors() {
    std::vector<at::Tensor> outputs(numDevices_);

    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (auto i = 0; i < numDevices_; i++) {
      cudaStreamSynchronize(streams_[i].stream());
      outputs[i] = tensors_[i].cpu();
    }

    return outputs;
  }

  std::vector<std::vector<at::Tensor>> getInputTensors() {
    return getTensorLists(inputs_);
  }
  std::vector<std::vector<at::Tensor>> getOutputTensors() {
    return getTensorLists(outputs_);
  }

  int numDevices() const {
    return numDevices_;
  }

 private:
  std::vector<std::vector<at::Tensor>> getTensorLists(
      std::vector<std::vector<at::Tensor>>& tensor_lists) {
    std::vector<std::vector<at::Tensor>> outputs(numDevices_);
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i] = std::vector<at::Tensor>(worldSize_ * numDevices_);
    }

    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (auto i = 0; i < numDevices_; ++i) {
      cudaStreamSynchronize(streams_[i].stream());
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        outputs[i][j] = tensor_lists[i][j].cpu();
      }
    }
    return outputs;
  }

 protected:
  // Launches sleep on every CUDA device
  void launchDeviceSleep() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }
  }

  // Launches value initialization for every tensor
  void valueInitialization() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      tensors_[i].fill_(pg_->getRank() * numDevices_ + i);
    }
  }

  const int numDevices_;
  THCState* state_;
  int worldSize_;
  std::vector<at::Tensor> tensors_;
  std::vector<std::vector<at::Tensor>> inputs_;
  std::vector<std::vector<at::Tensor>> outputs_;
  std::vector<CUDAStream> streams_;
};

class AllreduceNCCLTest : public NCCLTest {
 public:
  AllreduceNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  c10::intrusive_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    using namespace torch::autograd::profiler;
    // Make sure enabling profile does not make any issue. Note, in single
    // process multi-device mode we do not expect any events be populated for
    // collective operations, since profiling for that mode is not supported.
    enableProfilerLegacy({ProfilerState::CPU});
    auto results = pg_->allreduce(tensors_);
    disableProfilerLegacy();
    return results;
  }
};

class BroadcastNCCLTest : public NCCLTest {
 public:
  BroadcastNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  c10::intrusive_ptr<c10d::ProcessGroup::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->broadcast(tensors_, options);
  }
};

class ReduceNCCLTest : public NCCLTest {
 public:
  ReduceNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  c10::intrusive_ptr<c10d::ProcessGroup::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    ::c10d::ReduceOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->reduce(tensors_, options);
  }
};

class AllgatherNCCLTest : public NCCLTest {
 public:
  AllgatherNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  c10::intrusive_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    return pg_->allgather(outputs_, tensors_);
  }
};

struct ReduceScatterNCCLTest : NCCLTest {
  ReduceScatterNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  c10::intrusive_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    at::cuda::OptionalCUDAGuard deviceGuard;
    launchDeviceSleep();

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        inputs_[i][j].fill_(
            pg_->getRank() * numDevices_ * worldSize_ + i * worldSize_ + j);
      }
    }

    return pg_->reduce_scatter(tensors_, inputs_);
  }
};

void testAllreduce(const std::string& path, int rank, int size) {
  auto test = AllreduceNCCLTest(path, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  // Validation
  const int totalNumGPUs = test.numDevices() * size;
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  auto tensors = test.getTensors();
  for (size_t j = 0; j < tensors.size(); j++) {
    auto& tensor = tensors[j];
    auto data = tensor.data_ptr<float>();
    for (auto k = 0; k < tensor.numel(); k++) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce ouputs do not match expected outputs";
    }
  }
}

void testBroadcast(const std::string& path, int rank, int size) {
  auto test = BroadcastNCCLTest(path, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // try every permutation of root rank and root tensor
  for (auto rootRank = 0; rootRank < size; rootRank++) {
    for (auto rootTensor = 0; rootTensor < numDevices; rootTensor++) {
      auto work = test.run(rootRank, rootTensor);

      // wait for work to complete
      test.wait(work);

      // Check results
      const auto expected = (rootRank * numDevices + rootTensor);
      auto tensors = test.getTensors();
      for (size_t j = 0; j < tensors.size(); j++) {
        auto& tensor = tensors[j];
        auto data = tensor.data_ptr<float>();
        for (auto k = 0; k < tensor.numel(); k++) {
          EXPECT_EQ(data[k], expected)
              << "Broadcast outputs do not match expected outputs";
        }
      }
    }
  }
}

void testReduce(const std::string& path, int rank, int size) {
  auto test = ReduceNCCLTest(path, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // try every permutation of root rank and root tensor
  for (auto rootRank = 0; rootRank < size; rootRank++) {
    for (auto rootTensor = 0; rootTensor < numDevices; rootTensor++) {
      auto work = test.run(rootRank, rootTensor);

      // wait for work to complete
      test.wait(work);

      // Validation
      const int totalNumGPUs = numDevices * size;
      const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
      auto tensors = test.getTensors();
      if (rank == rootRank) {
        auto& tensor = tensors[rootTensor];
        auto data = tensor.data_ptr<float>();
        for (auto k = 0; k < tensor.numel(); k++) {
          EXPECT_EQ(data[k], expected)
              << "Reduce outputs do not match expected outputs";
        }
      }
    }
  }
}

void testAllgather(const std::string& path, int rank, int size) {
  auto test = AllgatherNCCLTest(path, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  // Validation
  auto tensors = test.getOutputTensors();
  // device index
  for (size_t i = 0; i < tensors.size(); ++i) {
    // rank index
    for (size_t j = 0; j < tensors[i].size(); ++j) {
      const auto expected = j;
      auto& tensor = tensors[i][j];
      auto data = tensor.data_ptr<float>();
      for (auto k = 0; k < tensor.numel(); k++) {
        EXPECT_EQ(data[k], expected)
            << "Allgather outputs do not match expected outputs";
      }
    }
  }
}

void testReduceScatter(const std::string& path, int rank, int size) {
  auto test = ReduceScatterNCCLTest(path, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  const auto participants = test.numDevices() * size;
  const auto base = (participants * (participants - 1)) / 2;

  // Validation
  auto tensors = test.getTensors();
  // device index
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto modifier = participants * (rank * participants + i);
    const auto expected = base + modifier;
    auto& tensor = tensors[i];
    auto data = tensor.data_ptr<float>();
    for (auto j = 0; j < tensor.numel(); j++) {
      EXPECT_EQ(data[j], expected) << "ReduceScatter outputs do not match expected outputs!";
    }
  }
}

class ProcessGroupNCCLTest: public ::testing::Test {
 protected:
  void SetUp() override {
    // Use WORLD_SIZE and RANK environmental variables to do multi-node
    // distributed testing
    auto sizeEnv = std::getenv("WORLD_SIZE");
    auto rankEnv = std::getenv("RANK");

    if (sizeEnv && rankEnv) {
      size_ = std::stoi(std::string(sizeEnv));
      rank_ = std::stoi(std::string(rankEnv));
    }
    LOG(INFO) << "Multi-node world size: " << size_ << " rank: " << rank_;
  }

  void TearDown() override {
    // Reset NCCL_BLOCKING_WAIT environment variable after each run.
    ASSERT_TRUE(setenv(c10d::NCCL_BLOCKING_WAIT, "0", 1) == 0);
  }

  bool skipTest() {
    // Skip tests if CUDA is not available.
    if (!at::cuda::is_available()) {
      LOG(INFO) << "CUDA not available, skipping test";
      return true;
    }
    return false;
  }

  int size_{1};
  int rank_{0};
};

TEST_F(ProcessGroupNCCLTest, testAllreduce) {
  if (skipTest()) {
    return;
  }
  {
    TemporaryFile file;
    testAllreduce(file.path, rank_, size_);
  }
}

TEST_F(ProcessGroupNCCLTest, testBroadcast) {
  if (skipTest()) {
    return;
  }
  {
    TemporaryFile file;
    testBroadcast(file.path, rank_, size_);
  }
}

TEST_F(ProcessGroupNCCLTest, testReduce) {
  if (skipTest()) {
    return;
  }
  {
    TemporaryFile file;
    testReduce(file.path, rank_, size_);
  }
}

TEST_F(ProcessGroupNCCLTest, testAllgather) {
  if (skipTest()) {
    return;
  }
  {
    TemporaryFile file;
    testAllgather(file.path, rank_, size_);
  }
}

TEST_F(ProcessGroupNCCLTest, testReduceScatter) {
  if (skipTest()) {
    return;
  }
  {
    TemporaryFile file;
    testReduceScatter(file.path, rank_, size_);
  }
}

TEST_F(ProcessGroupNCCLTest, testBackendName) {
  if (skipTest()) {
    return;
  }
  {
    TemporaryFile file;
    auto test = NCCLTestBase(file.path);
    test.initialize(rank_, size_);
    EXPECT_EQ(
      test.getProcessGroup().getBackendName(), std::string(c10d::NCCL_BACKEND_NAME));
  }
}
