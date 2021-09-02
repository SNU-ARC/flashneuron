#!/usr/bin/env python3

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    GENERIC_TESTS,
    TENSORPIPE_TESTS,
    MultiProcess,
    generate_tests,
)


globals().update(
    generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_TESTS + TENSORPIPE_TESTS,
        MultiProcess.SPAWN,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
