
import os

import numpy as np
import torch
import torch.distributed as c10d
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import (
    DDPCommHookType,
    register_ddp_comm_hook,
)
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.common_utils import run_tests


def gpus_for_rank(world_size):
    visible_devices = list(range(torch.cuda.device_count()))
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()
        torch.manual_seed(0)
        self.p = nn.Parameter(torch.randn(40, 20))

    def forward(self, x):
        return self.p * x


class TestDdpCommHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        return self.t0(x ** (1 + rank))


class DistributedDataParallelCommHookTest(MultiProcessTestCase):
    def setUp(self):
        super(DistributedDataParallelCommHookTest, self).setUp()
        self._fork_processes()

    def tearDown(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _local_model(self):
        local_model = TestDdpCommHook().cpu()

        return local_model

    def _get_grads(self, process_group, hook_type=None):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            TestDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        # Register DDP Communication Hook if defined
        if hook_type is not None:
            register_ddp_comm_hook(
                comm_hook_type=hook_type, model=gpu_model, state=process_group
            )

        return self._run_and_get_grads(gpu_model)

    def _run_and_get_grads(self, model):
        torch.manual_seed(2020)
        input = torch.randn(40, 20)
        # Run forward
        output = model(input, self.rank)

        # Run backward
        output.mean().backward()

        return [p.grad.data.cpu().numpy() for p in model.parameters()]

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm
    def test_ddp_comm_hook_allreduce_hook(self):
        """
        This unit test verifies the ``allreduce`` hook registered case gives same result
        with no hook registered case.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.ALLREDUCE)

        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=0)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm
    def test_ddp_comm_hook_fp16compress_hook(self):
        """
        This unit test verifies the ``fp16 compress`` hook registered case
        gives close result with no hook registered case.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.FP16_COMPRESS)

        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm
    def test_ddp_comm_hook_quantize_per_tensor_hook(self):
        """
        This unit test verifies the ``quantize per tensor`` hook registered case
        gives close result with no hook registered case.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.QUANTIZE_PER_TENSOR)

        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm
    def test_ddp_comm_hook_quantize_per_channel_hook(self):
        """
        This unit test verifies the ``quantize per channel`` hook registered case
        gives close result with no hook registered case.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(
            process_group, DDPCommHookType.QUANTIZE_PER_CHANNEL
        )

        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
