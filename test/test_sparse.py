import torch

# TODO: remove this global setting
# Sparse tests use double as the default dtype
torch.set_default_dtype(torch.double)

import itertools
import functools
import operator
import random
from collections import defaultdict
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfRocm, do_test_dtypes, \
    do_test_empty_full, load_tests, TEST_NUMPY, TEST_SCIPY, IS_WINDOWS, gradcheck
from torch.testing._internal.common_cuda import TEST_CUDA, _get_torch_cuda_version
from numbers import Number
from typing import Dict, Any
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops)
from torch.testing._internal.common_methods_invocations import \
    (sparse_unary_ufuncs)

if TEST_SCIPY:
    import scipy.sparse

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# batched grad doesn't support sparse
gradcheck = functools.partial(gradcheck, check_batched_grad=False)

def cpu_only(inner):
    @functools.wraps(inner)
    def outer(self, *args, **kwargs):
        if self.is_cuda:
            raise unittest.SkipTest("Test is CPU-only")
        inner(self, *args, **kwargs)
    return outer


def cuda_only(inner):
    @functools.wraps(inner)
    def outer(self, *args, **kwargs):
        if not self.is_cuda:
            raise unittest.SkipTest("Test is GPU-only")
        inner(self, *args, **kwargs)
    return outer

class TestSparse(TestCase):

    def setUp(self):
        # These parameters control the various ways we can run the test.
        # We will subclass and override this method to implement CUDA
        # tests
        self.is_cuda = False
        self.is_uncoalesced = False
        self.device = 'cpu'
        self.exact_dtype = True
        self.value_dtype = torch.float64
        self.index_tensor = lambda *args: torch.tensor(*args, dtype=torch.int64, device=self.device)
        self.value_empty = lambda *args: torch.empty(*args, dtype=self.value_dtype, device=self.device)
        self.value_tensor = lambda *args: torch.tensor(*args, dtype=self.value_dtype, device=self.device)

        def sparse_empty_factory(*args, **kwargs):
            kwargs['dtype'] = kwargs.get('dtype', self.value_dtype)
            kwargs['layout'] = kwargs.get('layout', torch.sparse_coo)
            kwargs['device'] = kwargs.get('device', self.device)
            return torch.empty(*args, **kwargs)
        self.sparse_empty = sparse_empty_factory

        def sparse_tensor_factory(*args, **kwargs):
            kwargs['dtype'] = kwargs.get('dtype', self.value_dtype)
            kwargs['device'] = kwargs.get('device', self.device)
            return torch.sparse_coo_tensor(*args, **kwargs)
        self.sparse_tensor = sparse_tensor_factory
        self.legacy_sparse_tensor = torch.sparse.DoubleTensor
        super(TestSparse, self).setUp()

    def _gen_sparse(self, sparse_dim, nnz, with_size):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, self.is_uncoalesced, self.device)

        if self.is_uncoalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    def assert_uncoalesced(self, x):
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        existing_indices = set()
        for i in range(x._nnz()):
            index = str(x._indices()[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    def randn(self, *args, **kwargs):
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return self.value_empty(*args, **kwargs).normal_()

    def test_print(self):
        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]

        printed = []
        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            printed.append("# shape: {}".format(torch.Size(shape)))
            printed.append("# nnz: {}".format(nnz))
            printed.append("# sparse_dim: {}".format(sparse_dim))
            printed.append("# indices shape: {}".format(indices_shape))
            printed.append("# values shape: {}".format(values_shape))

            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=self.device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if self.is_uncoalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=self.value_dtype,
                                  device=self.device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape)

            dtypes = [torch.int32]
            if values.dtype == torch.double:
                dtypes.append(torch.float)
            else:
                dtypes.append(torch.double)
            for dtype in dtypes:
                printed.append("########## {} ##########".format(dtype))
                x = sp_tensor.detach().to(dtype)
                printed.append("# sparse tensor")
                printed.append(str(x))
                if x.dtype.is_floating_point:
                    printed.append("# after requires_grad_")
                    printed.append(str(x.requires_grad_()))
                    printed.append("# after addition")
                    printed.append(str(x + x))
                printed.append("# _indices")
                printed.append(str(x._indices()))
                printed.append("# _values")
                printed.append(str(x._values()))
            printed.append('')
        self.assertExpected('\n'.join(printed))

    def test_basic(self):
        def test_shape(sparse_dims, nnz, with_size):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size)
            self.assertEqual(i, x._indices())
            self.assertEqual(v, x._values())
            self.assertEqual(x.ndimension(), len(with_size))
            self.assertEqual(x.coalesce()._nnz(), nnz)
            self.assertEqual(list(x.size()), with_size)

            # Test .indices() and .values()
            if self.is_uncoalesced:
                with self.assertRaisesRegex(RuntimeError, "Cannot get indices on an uncoalesced tensor"):
                    x.indices()
                with self.assertRaisesRegex(RuntimeError, "Cannot get values on an uncoalesced tensor"):
                    x.values()
            else:
                self.assertEqual(x.indices(), x._indices())
                self.assertEqual(x.values(), x._values())

        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # Make sure that coalesce handles duplicate indices correctly
        i = self.index_tensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]])
        v = self.value_tensor([[idx**2, idx] for idx in range(i.size(1))])
        x = self.sparse_tensor(i, v, torch.Size([10, 2]))
        self.assertEqual(x.coalesce()._nnz(), 9)

        # Make sure we can access empty indices / values
        x = self.legacy_sparse_tensor()
        self.assertEqual(x._indices().numel(), 0)
        self.assertEqual(x._values().numel(), 0)

    def test_coalesce(self):

        def _test_coalesce(x):
            tc = t.coalesce()
            self.assertEqual(tc.to_dense(), t.to_dense())
            self.assertTrue(tc.is_coalesced())
            # Our code below doesn't work when nnz is 0, because
            # then it's a 0D tensor, not a 2D tensor.
            if t._nnz() == 0:
                self.assertEqual(t._indices(), tc._indices())
                self.assertEqual(t._values(), tc._values())
                return tc

            value_map: Dict[Any, Any] = {}
            for idx, val in zip(t._indices().t(), t._values()):
                idx_tup = tuple(idx.tolist())
                if idx_tup in value_map:
                    value_map[idx_tup] += val
                else:
                    value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

            new_indices = sorted(list(value_map.keys()))
            _new_values = [value_map[idx] for idx in new_indices]
            if t._values().ndimension() < 2:
                new_values = t._values().new(_new_values)
            else:
                new_values = torch.stack(_new_values)

            new_indices = t._indices().new(new_indices).t()
            tg = t.new(new_indices, new_values, t.size())

            self.assertEqual(tc._indices(), tg._indices())
            self.assertEqual(tc._values(), tg._values())

            if t.is_coalesced():
                self.assertEqual(tc._indices(), t._indices())
                self.assertEqual(tc._values(), t._values())

        for empty_i, empty_v, empty_nnz in itertools.product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5

            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size + dense_size)
            _test_coalesce(t)  # this tests correctness

    def test_ctor_size_checks(self):
        indices = self.index_tensor([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        values = self.value_tensor([2, 1, 3, 4])

        # indices inconsistent with size
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 1, 1])))

        # values inconsistent with size
        values = self.value_tensor([
            [2, 1, 2, 1],
            [1, 0, 5, 2],
        ])
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 4, 2, 1])))

    def test_to_dense(self):
        def test_tensor(x, res):
            x.to_dense()  # Tests triple to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            # We dont have to_dense for half types, so we don't request
            # exact_dtype if res.type is torch.float16.
            dense_x = x.to_dense()
            safe_dense_x = self.safeToDense(x)
            if (res.dtype == torch.float16):
                exact_dtype = False
            else:
                exact_dtype = True
                dense_x = dense_x.to(res.dtype)
                safe_dense_x = safe_dense_x.to(res.dtype)
            self.assertEqual(res, dense_x, exact_dtype=exact_dtype)
            self.assertEqual(res, safe_dense_x, exact_dtype=exact_dtype)

            def fn(x):
                return x.to_dense()
            x.requires_grad_(True)
            gradcheck(fn, (x,), check_sparse_nnz=True)

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        # we don't have to_dense for half types on CPU because it is implemented
        # with a slower add_ operation
        for dtype in [torch.float16, torch.float32, torch.float64] if self.device != 'cpu' else [torch.float32, torch.float64]:
            v = self.value_tensor([2, 1, 3, 4]).to(dtype=dtype)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
            res = self.value_tensor([
                [[2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 4]],
            ]).to(dtype=dtype)

            test_tensor(x, res)

            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ])
            v = self.value_empty(4, 0).to(dtype=dtype)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
            res = self.value_empty(3, 4, 5, 0).to(dtype=dtype)
            test_tensor(x, res)

    # half tensors on cpu don't implement to_dense, so need to convert to float
    def _to_dense_half_safe(self, tensor):
        if(tensor.dtype == torch.half and tensor.device.type == 'cpu'):
            return tensor.to(torch.float).to_dense().to(torch.half)
        else:
            return tensor.to_dense()

    @skipIfRocm
    def test_to_sparse(self):
        shape = [10, 5, 19, 8]
        max_nnz = 1
        for dim, dim_sz in enumerate(shape, 1):
            max_nnz *= dim_sz
            rnnz = torch.randint(2, max_nnz, (1,)).item()
            for nnz in [0, 1, rnnz]:
                for dtype in [torch.float16, torch.float64, torch.int]:
                    expected, _, _ = self._gen_sparse(dim, nnz, shape)
                    expected = expected.to(dtype)

                    d = self._to_dense_half_safe(expected)
                    result = d.to_sparse(dim)
                    self.assertEqual(d, self._to_dense_half_safe(result))  # == not implemented for sparse tensors yet
                    self.assertEqual(expected.size(), result.size())
                    self.assertEqual(dim, result.sparse_dim())

        sp, _, _ = self._gen_sparse(2, 10, [3, 3, 3])
        self.assertRaises(RuntimeError, lambda: sp.to_sparse())

    def test_sparse_bool(self):
        a = self.value_tensor([True, False]).to(torch.bool)
        b = a.to_sparse().to_dense()
        self.assertEqual(a, b)

    def test_scalar(self):
        # tensor with value
        a = self.sparse_tensor(self.index_tensor([]).unsqueeze(1), 12.3, [])
        self.assertEqual(1, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(self.value_tensor(12.3), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

        # tensor with multiple values
        a = self.sparse_tensor(self.index_tensor([]).unsqueeze(1).expand(0, 2), [12.3, 12.3], [])
        self.assertEqual(2, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(self.value_tensor(12.3 * 2), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

        # tensor without value
        a = self.sparse_empty(())
        self.assertEqual(0, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(self.value_tensor(0), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

    def test_shared(self):
        i = self.index_tensor([[2]])
        v = self.value_tensor([5])
        x = self.sparse_tensor(i, v, torch.Size([3]))
        v[0] = 6
        self.assertEqual(self.value_tensor([0, 0, 6]), self.safeToDense(x))
        i[0][0] = 0
        self.assertEqual(self.value_tensor([6, 0, 0]), self.safeToDense(x))

        i = self.index_tensor([[2]])
        v = self.value_empty(1, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        i[0][0] = 0
        self.assertEqual(self.value_empty(3, 0), self.safeToDense(x))

    def test_to_dense_hybrid(self):
        def test_tensor(x, res):
            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())
            self.assertEqual(res, self.safeToDense(x))

            def fn(x):
                return x.to_dense()
            x.requires_grad_(True)
            gradcheck(fn, (x,), check_sparse_nnz=True)

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ])
        v = self.value_tensor([[2, 3], [1, 2], [3, 4], [4, 5]])
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2]))
        res = self.value_tensor([
            [[2, 3],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[1, 2],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[3, 4],
             [0, 0],
             [0, 0],
             [4, 5]],
        ])
        test_tensor(x, res)

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ])
        v = self.value_empty(4, 2, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2, 0]))
        res = self.value_empty(3, 4, 2, 0)
        test_tensor(x, res)

    def test_contig(self):
        def test_tensor(x, exp_i, exp_v):
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = self.value_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x = self.sparse_tensor(i, v, torch.Size([100, 100]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = self.value_tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7])
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.value_tensor([3, 2, 4, 1])
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.value_tensor([2, 1, 3, 4])
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.value_empty(4, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.value_empty(4, 0)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.value_tensor([3, 2, 4, 1])
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.value_tensor([6, 4])
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.value_empty(4, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.value_empty(2, 0)
        test_tensor(x, exp_i, exp_v)

    def test_contig_hybrid(self):
        def test_tensor(x, exp_i, exp_v):
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = self.value_tensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ])
        x = self.sparse_tensor(i, v, torch.Size([100, 100, 2]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = self.value_tensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ])
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.value_tensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]])
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.value_tensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]])
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.value_empty(4, 3, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.value_empty(4, 3, 0)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.value_tensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]])
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.value_tensor([[6, 4, 5], [4, 3, 4]])
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.value_empty(4, 3, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.value_empty(2, 3, 0)
        test_tensor(x, exp_i, exp_v)

    def test_clone(self):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size)[0]
            if self.is_uncoalesced:
                self.assertFalse(x.is_coalesced())
                y = x.clone()
                self.assertFalse(y.is_coalesced())
            x = x.coalesce()
            self.assertTrue(x.is_coalesced())
            y = x.clone()
            self.assertTrue(y.is_coalesced())

        test_shape(4, 20, 5)
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

    def test_Sparse_to_Sparse_copy_(self):
        # This is for testing torch.copy_(SparseTensor, SparseTensor)
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes)

        # test copy
        x2_dense = x2.to_dense()
        x1.copy_(x2)
        self.assertEqual(x2_dense, x1.to_dense())

        # test type conversion (when x1.copy_(x2), x1.dtype should stay the same)
        x1 = x1.to(torch.float32)

        x2 = x2.to(torch.float16)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        x2 = x2.to(torch.float64)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        # test no broadcast
        self.assertRaises(RuntimeError, lambda: x1.copy_(x2.narrow_copy(0, 0, 1)))

        # test raise error on copy_() between dense and sparse Tensors
        self.assertRaises(RuntimeError, lambda: x1.copy_(torch.randn(5, 5)))

        # test autograd
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes)
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone()
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to_dense())
        self.assertEqual(None, x1.grad)

    @unittest.skipIf(torch.cuda.device_count() < 2, "no multi-GPU")
    def test_Sparse_to_Sparse_copy_multi_gpu(self):
        # This is for testing torch.copy_(SparseTensor, SparseTensor) across GPU devices
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes)
        x1 = x1.to('cuda:0')

        def test_cross_device(x1, x2):
            x1_device = x1.device
            x1.copy_(x2)
            self.assertEqual(x2.to('cuda:0').to_dense(), x1.to_dense())
            self.assertEqual(x1_device, x1.device)

        test_cross_device(x1, x2.to('cuda:1'))  # test across gpu devices
        test_cross_device(x1, x2.to('cpu'))  # test between cpu and gpu

        # test autograd
        x2 = x2.to('cuda:1')
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone().to('cuda:0')
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to('cuda:0').to_dense())
        self.assertEqual(None, x1.grad)

    @cuda_only
    def test_cuda_empty(self):
        def test_tensor(x):
            y = x.cuda(0)
            self.assertEqual(x.sparse_dim(), y.sparse_dim())
            self.assertEqual(x.dense_dim(), y.dense_dim())
            x = y.cpu()
            self.assertEqual(y.sparse_dim(), x.sparse_dim())
            self.assertEqual(y.dense_dim(), x.dense_dim())

        x = torch.sparse.FloatTensor(2, 3, 4)
        test_tensor(x)

        x = torch.sparse.HalfTensor(2, 3, 4)
        test_tensor(x)

        x = torch.cuda.sparse.HalfTensor(2, 3, 4)
        test_tensor(x)

        x = torch.sparse.FloatTensor(2, 3, 4, 0)
        test_tensor(x)

    def test_transpose(self):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size)[0]
            y = self.safeToDense(x)

            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

                x = x.transpose(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

        test_shape(4, 6, 3)
        test_shape(4, 3, [7, 7, 7, 3, 3, 3, 0])
        test_shape(4, 0, [0, 0, 7, 3, 3, 3, 0])

    @cpu_only
    def test_coalesce_transpose_mm(self):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [dj, di])
            y = torch.randn(dj, dk)

            x_coalesced = x.coalesce()
            self.assertTrue(x_coalesced.is_coalesced())

            x_coalesced_t = x_coalesced.t()
            # Transpose is `colasced`-preserving if the indices tensor is empty.
            self.assertEqual(x_coalesced_t.is_coalesced(), di * nnz == 0)

            res = torch.mm(x_coalesced_t, y)
            expected = torch.mm(self.safeToDense(x_coalesced_t), y)
            self.assertEqual(res, expected)

        test_shape(10, 20, 30, 20)
        test_shape(0, 20, 30, 0)
        test_shape(10, 0, 30, 0)
        test_shape(10, 20, 0, 0)
        test_shape(10, 20, 0, 20)

    def test_t_empty(self):
        def test_in_place(x):
            shape_original = x.shape
            x.t_()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), x.size())
            self.assertEqual(0, x._indices().numel())
            self.assertEqual(0, x._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        def test_not_in_place(x):
            shape_original = x.shape
            y = x.t()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), y.size())
            self.assertEqual(0, y._indices().numel())
            self.assertEqual(0, y._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        x = self.sparse_empty(2, 3)
        test_in_place(x)
        test_not_in_place(x)

        x = self.sparse_empty(2, 0)
        test_in_place(x)
        test_not_in_place(x)

    def test_add_zeros(self):
        def test_shape(sparse_dims, nnz, sizes):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
            zeros = torch.zeros(sizes, layout=torch.sparse_coo).to(x.device)
            r1 = zeros + x
            r2 = x + zeros
            self.assertEqual(r1, x)
            self.assertEqual(r2, x)

        test_shape(1, 20, [1])
        test_shape(4, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 0])

    def test_add_sub_nnz(self):
        # nnz should not grow unbounded (gh-34964)
        x = torch.randn(10, device=self.device).to_sparse()
        x.add_(x)
        x.add_(x)
        self.assertLessEqual(x._nnz(), 10)

        x.sub_(2 * x)
        x.sub_(2 * x)
        self.assertLessEqual(x._nnz(), 10)

    def test_cat(self):
        # shapes: list of tuples (sparse_dims, nnz, sizes)
        def test_shapes(shapes, dim, fail_message=None):
            inputs = [self._gen_sparse(shape[0], shape[1], shape[2])[0]
                      for shape in shapes]
            if fail_message:
                with self.assertRaisesRegex(RuntimeError, fail_message):
                    torch.cat(inputs, dim)
            else:
                result = torch.cat(inputs, dim)
                dense_result = torch.cat([t.to_dense() for t in inputs], dim)
                self.assertEqual(dense_result, result.to_dense())

        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], 1)

        # mismatched sizes
        test_shapes([(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4])], 0,
                    "All tensors must have the same shape: \\[2, 3, 4].*\\[2, 1, 4]")
        # hybrid sparse/dense
        test_shapes(
            [(2, 10, [2, 3, 4]), (2, 10, [2, 1, 4]), (2, 10, [2, 4, 4])], 1)
        # cat along dense dim
        test_shapes([(2, 10, [2, 3, 4]), (2, 10, [2, 3, 7])], 2)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 1)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 2)
        # mismatched dimensions
        test_shapes([(2, 10, [2, 3, 4]), (3, 10, [2, 3, 4])], 0,
                    "All tensors must have the same.*2, 1, but tensor at position 1 has 3, 0.")
        # wrapped dimension
        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], -2)

        # sparse with dense
        sp = self._gen_sparse(3, 10, [2, 3, 4])[0]
        dn = sp.to_dense()
        with self.assertRaisesRegex(RuntimeError,
                                    "Concatenating sparse tensors, but a dense tensor was found at position 1."):
            torch.cat((sp, dn))

    def test_unsqueeze(self):
        def test_shape(sparse_dims, nnz, sizes, unsqueeze_dim, fail_message=None):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.unsqueeze(x, unsqueeze_dim)
            else:
                result = torch.unsqueeze(x, unsqueeze_dim)
                dense_result = torch.unsqueeze(x.to_dense(), unsqueeze_dim)
                self.assertEqual(dense_result, result.to_dense())

        # basic case
        test_shape(3, 10, [5, 7, 11], 0)

        # hybrid sparse/dense, unsqueeze along sparse dim
        test_shape(3, 10, [5, 7, 11, 13, 17], 0)
        test_shape(3, 10, [5, 7, 11, 13, 17], 3)

        # unsqueeze along dense dimensions
        test_shape(3, 10, [5, 7, 11, 13, 17], 4)
        test_shape(3, 10, [5, 7, 11, 13, 17], 5)

        # wrapped dimensions
        test_shape(3, 10, [5, 7, 11, 13, 17], -1)
        test_shape(3, 10, [5, 7, 11, 13, 17], -6)

        # bounds
        test_shape(3, 10, [5, 7, 11, 13, 17], -7, "Dimension out of range")
        test_shape(3, 10, [5, 7, 11, 13, 17], 6, "Dimension out of range")

    def test_select(self):
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.select(x, select_dim, select_index)
            else:
                result = torch.select(x, select_dim, select_index)
                if result.is_sparse:
                    result = result.to_dense()
                dense_result = torch.select(x.to_dense(), select_dim, select_index)
                self.assertEqual(dense_result, result)


        sizes = [5, 7, 11, 13, 17]
        # hybrid sparse/dense, select sparse dim, result is dense
        for i in range(sizes[0]):
            test_shape(1, 10, sizes, 0, i)
        test_shape(1, 10, sizes, 0, sizes[0] + 1, r'select[(][)][:] index \d out of range.*')

        # hybrid sparse/dense, select sparse dim, result is sparse
        for d in range(3):
            for i in range(sizes[d]):
                test_shape(3, 10, sizes, d, i)

        # hybrid sparse/dense, select dense dim, result is sparse
        for d in range(1, 3):
            for i in range(sizes[d]):
                test_shape(1, 10, sizes, d, i)


    def test_index_select(self):
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            if isinstance(select_index, int):
                select_index = [select_index]
            if isinstance(select_index, list):
                select_index = torch.tensor(select_index, device=self.device, dtype=torch.long)
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.index_select(x, select_dim, select_index)
            else:
                result = torch.index_select(x, select_dim, select_index)
                if result.is_sparse:
                    result = result.to_dense()
                dense_result = torch.index_select(x.to_dense(), select_dim, select_index)
                self.assertEqual(dense_result, result)

        sizes = [5, 7, 11, 13, 17]
        for d in range(len(sizes)):
            for index in [0, sizes[d] - 1, [0, sizes[d] // 2, sizes[d] - 1]]:
                test_shape(1, 10, sizes, d, index)
                test_shape(len(sizes) // 2, 10, sizes, d, index)
                test_shape(len(sizes), 10, sizes, d, index)

    @cpu_only
    def test_mm(self):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [di, dj])
            t = torch.randn(di, dk)
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(t, self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 200, 20)
        test_shape(64, 10000, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 100, 0)
        test_shape(10, 100, 0, 0)
        test_shape(10, 100, 0, 20)

    @unittest.skipIf(
        IS_WINDOWS and TEST_CUDA,
        "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1"
    )
    @unittest.skipIf(
        TEST_CUDA and _get_torch_cuda_version() < [10, 1],
        "bmm sparse-dense requires CUDA 10.1 or greater"
    )
    def test_bmm(self):
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            a_list = []
            b_list = []
            for mat_idx in range(num_mats):
                a_mat = self._gen_sparse(2, nnz, [dim_i, dim_j])[0]
                b_mat = torch.randn([dim_j, dim_k])
                if self.is_cuda:
                    a_mat = a_mat.cuda()
                    b_mat = b_mat.cuda()
                a_list.append(a_mat)
                b_list.append(b_mat)

            a = torch.stack(a_list)
            b = torch.stack(b_list)
            ab = a.bmm(b)

            # Compare each matrix against result from mm()
            for mat_idx in range(num_mats):
                a_mat = a_list[mat_idx]
                b_mat = b_list[mat_idx]
                ab_mat_bmm = ab[mat_idx]
                ab_mat_mm = a_mat.mm(b_mat)
                self.assertEqual(ab_mat_bmm, ab_mat_mm)

        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

        a = torch.rand([10, 23, 32])
        a[3] = torch.zeros(23, 32)
        a[6] = torch.zeros(23, 32)
        a = a.to_sparse()
        b = torch.rand([10, 32, 10])
        b[4] = torch.zeros(32, 10)
        b[6] = torch.zeros(32, 10)
        if self.is_cuda:
            a = a.cuda()
            b = b.cuda()
        ab = a.bmm(b)
        for mat_idx in range(ab.size(0)):
            ab_mat = ab[mat_idx]
            ab_mat_check = a[mat_idx].mm(b[mat_idx])
            self.assertEqual(ab_mat, ab_mat_check)

        ab_traspose_check = b.transpose(1, 2).to_sparse().bmm(
            a.transpose(1, 2).to_dense()
        ).transpose(1, 2)
        self.assertEqual(ab, ab_traspose_check)

    @cuda_only
    @unittest.skipIf(
        IS_WINDOWS,
        "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1"
    )
    @unittest.skipIf(
        _get_torch_cuda_version() < [10, 1],
        "bmm sparse-dense requires CUDA 10.1 or greater"
    )
    def test_bmm_deterministic(self):
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            a_list = []
            b_list = []
            for mat_idx in range(num_mats):
                a_list.append(self._gen_sparse(2, nnz, [dim_i, dim_j])[0])
                b_list.append(torch.randn([dim_j, dim_k]))

            a = torch.stack(a_list).cuda()
            b = torch.stack(b_list).cuda()
            ab_nondeterministic = torch._bmm(a, b, deterministic=False)
            ab_deterministic = torch._bmm(a, b, deterministic=True)
            diff_abs = (ab_deterministic - ab_nondeterministic).abs()
            diff_rel = diff_abs / ab_deterministic.abs()
            diff_rel[torch.isnan(diff_rel)] = 0

            # deterministic and non-deterministic results should either be
            # equal or within a small relative difference
            equal_abs_or_rel = diff_abs.eq(0).logical_or(diff_rel.lt(0.001))
            self.assertTrue(equal_abs_or_rel.all())

        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

    @cuda_only
    @unittest.skipIf(
        not IS_WINDOWS or _get_torch_cuda_version() >= [11, 0],
        "this test ensures bmm sparse-dense CUDA gives an error when run on Windows with CUDA < 11.0"
    )
    def test_bmm_windows_error(self):
        a = torch.rand(2, 2, 2).to_sparse().cuda()
        b = torch.rand(2, 2, 2).cuda()
        with self.assertRaisesRegex(
                RuntimeError,
                "bmm sparse-dense CUDA is not supported on Windows with cuda before 11.0"):
            ab = a.bmm(b)

    @cuda_only
    @skipIfRocm
    @unittest.skipIf(
        _get_torch_cuda_version() >= [10, 1],
        "this test ensures bmm gives error if CUDA version is less than 10.1"
    )
    def test_bmm_cuda_version_error(self):
        a = torch.rand(2, 2, 2).to_sparse().cuda()
        b = torch.rand(2, 2, 2).cuda()
        with self.assertRaisesRegex(
                RuntimeError,
                "bmm sparse-dense requires CUDA 10.1 or greater"):
            ab = a.bmm(b)

    @cpu_only
    def test_saddmm(self):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            t = self._gen_sparse(2, nnz, [di, dk])[0]
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.smm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

    @cpu_only
    def test_sspaddmm(self):

        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            t = self._gen_sparse(2, nnz, [di, dk])[0]
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = t.sspaddmm(x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = t.sspaddmm(x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

        # Test code from issue https://github.com/pytorch/pytorch/issues/45113
        batch_size, input_size, hidden_size = 5, 3, 7

        # Create coalesced sparse tensor as in the issue
        weight = torch.randn(hidden_size, input_size).to_sparse()
        self.assertTrue(weight.is_coalesced())
        self.assertFalse(weight._indices().is_contiguous())
        # Create un/coalesced sparse tensor
        bias = torch.randn((hidden_size, 1)).to_sparse()
        bias = torch.cat([bias] * batch_size, dim=1)

        if not self.is_uncoalesced:
            bias = bias.coalesce()

        x = torch.randn(input_size, batch_size)
        res = bias.sspaddmm(weight, x)

        true_result = (bias.to_dense() + torch.matmul(weight.to_dense(), x)).to_sparse()
        self.assertEqual(self.safeToDense(res), self.safeToDense(true_result))

    def test_sparse_addmm(self):
        def test_shape(m, n, p, nnz, broadcast):
            if broadcast:
                D1 = torch.randn((), device=self.device).requires_grad_(True)
            else:
                D1 = torch.randn(n, p, device=self.device).requires_grad_(True)
            D2 = torch.randn(m, p, device=self.device).requires_grad_(True)
            S = self._gen_sparse(2, nnz, [n, m])[0]
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            self.assertEqual(torch.sparse.addmm(D1, S, D2), torch.addmm(D1, S_dense, D2))

            def fn(S, D1, D2):
                return torch.sparse.addmm(D1, S, D2)
            gradcheck(fn, (S, D1, D2), check_sparse_nnz=True)

        test_shape(7, 8, 9, 20, False)
        test_shape(7, 8, 9, 20, True)

    def test_sparse_mm(self):
        def test_shape(d1, d2, d3, nnz, transposed):
            if transposed:
                D = torch.randn(d3, d2,
                                device=self.device).t_().requires_grad_(True)
            else:
                D = torch.randn(d2, d3, device=self.device).requires_grad_(True)
            S = self._gen_sparse(2, nnz, [d1, d2])[0]
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

            def fn(S, D):
                return torch.sparse.mm(S, D)
            gradcheck(fn, (S, D), check_sparse_nnz=True)

        test_shape(7, 8, 9, 20, False)
        test_shape(7, 8, 9, 20, True)

    def test_dsmm(self):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            y = self.randn(dj, dk)

            res = torch.dsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    def test_hsmm(self):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            y = self.randn(dj, dk)

            res = torch.hsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    def _test_spadd_shape(self, nnz, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x, _, _ = self._gen_sparse(len(shape_i), nnz, shape)
        y = self.randn(*shape)
        r = random.random()

        res = torch.add(y, x, alpha=r)
        expected = y + r * self.safeToDense(x)

        self.assertEqual(res, expected)

        # Non contiguous dense tensor
        s = list(shape)
        s[0] = shape[-1]
        s[-1] = shape[0]
        y = self.randn(*s)
        y.transpose_(0, len(s) - 1)
        r = random.random()

        res = torch.add(y, x, alpha=r)
        expected = y + r * self.safeToDense(x)

        self.assertEqual(res, expected)

        x, i, v = self._gen_sparse(len(shape_i), nnz, shape)
        nnz = i.size(1)

        # Non contiguous sparse indices tensor
        x_ = self.sparse_tensor(i[:, ::2], v[:int(nnz / 2)], x.shape)
        res = torch.add(y, x_, alpha=r)
        expected = y + r * self.safeToDense(x_)
        self.assertEqual(res, expected)

        # Non contiguous sparse values tensor
        x_ = self.sparse_tensor(i[:, :int(nnz / 2)], v[::2], x.shape)
        res = torch.add(y, x_, alpha=r)
        expected = y + r * self.safeToDense(x_)
        self.assertEqual(res, expected)

        # Non contiguous sparse indices and values tensors
        x_ = self.sparse_tensor(i[:, 1::2], v[1::2], x.shape)
        res = torch.add(y, x_, alpha=r)
        expected = y + r * self.safeToDense(x_)
        self.assertEqual(res, expected)

    def test_spadd(self):
        self._test_spadd_shape(10, [5, 6])
        self._test_spadd_shape(10, [10, 10, 10])
        self._test_spadd_shape(10, [50, 30, 20])
        self._test_spadd_shape(10, [5, 5, 5, 5, 5, 5])
        self._test_spadd_shape(0, [0, 30, 20])
        self._test_spadd_shape(0, [50, 0, 20])
        self._test_spadd_shape(0, [50, 30, 0])

    def test_spadd_hybrid(self):
        self._test_spadd_shape(10, [5, 6], [2, 3])
        self._test_spadd_shape(10, [10, 10, 10], [3])
        self._test_spadd_shape(10, [50, 30, 20], [2])
        self._test_spadd_shape(10, [5, 5, 5, 5, 5, 5], [2])
        self._test_spadd_shape(0, [0, 30, 20], [2, 0])
        self._test_spadd_shape(0, [50, 0, 20], [2, 0])
        self._test_spadd_shape(0, [50, 30, 0], [2, 0])
        self._test_spadd_shape(10, [50, 30, 20], [2, 0])

    @cuda_only
    def test_sparse_add_out_bfloat16(self):
        # fp32
        x, _, _ = self._gen_sparse(3, 5, 10)
        y, _, _ = self._gen_sparse(3, 5, 10)
        x = x.float().cuda()
        y = y.float().cuda()
        res_fp32 = torch.add(x, y)

        # bfloat16
        x = x.bfloat16()
        y = y.bfloat16()
        res_bf16 = torch.add(x, y)
        res_bf16 = res_bf16.float()  # to compare with reference
        self.assertEqual(res_fp32, res_bf16, atol=1e-2, rtol=0)

    def test_norm(self):
        def test_shape(sparse_dims, nnz, with_size):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, with_size)
            y = x.coalesce()
            self.assertEqual(x.norm(), y._values().norm())

        test_shape(3, 10, 100)
        test_shape(4, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(4, 0, [0, 0, 100, 5, 5, 5, 0])

        # Unsupported arguments should error
        kwarg_error_pairs = [
            ({'keepdim': True},
             RuntimeError, r'norm_sparse currently does not support keepdim=True'),
            ({'dim': 0},
             RuntimeError, r'norm_sparse currently only supports full reductions'),
            ({'dtype': torch.double, 'p': 'fro'},
             ValueError, r'dtype argument is not supported in frobenius norm'),
            ({'dtype': torch.double, 'p': 0},
             RuntimeError, r"norm_sparse currently does not support 'dtype' argument")
        ]
        x = self._gen_sparse(3, 10, 100)[0]
        for kwargs, err, msg in kwarg_error_pairs:
            with self.assertRaisesRegex(err, msg):
                x.norm(**kwargs)


    def test_sparse_sum(self):

        def run_tests(S, td=None):
            D = S.coalesce().to_dense().detach().requires_grad_(True)
            mask = (D == 0)
            if td is None:
                S_sum = torch.sparse.sum(S)
                D_sum = D.sum()
                self.assertEqual(S_sum, D_sum)

                def fn(S):
                    res = torch.sparse.sum(S)
                    if res.is_sparse:
                        res = res.to_dense()
                    return res
                gradcheck(fn, (S,), check_sparse_nnz=True)

            else:
                S_sum = torch.sparse.sum(S, td)
                D_sum = D.sum(td)
                self.assertEqual(S_sum.to_dense() if S_sum.is_sparse else S_sum, D_sum)

                def fn(S):
                    res = torch.sparse.sum(S, td)
                    if res.is_sparse:
                        res = res.to_dense()
                    return res
                gradcheck(fn, (S,), check_sparse_nnz=True)

        nnz = 10
        sparse_dims = 2
        with_size = [5, 5, 1, 4]  # use a dense dim = 1 to test for squeeze
        test_dims = []
        for i in range(1, 5):
            test_dims += itertools.combinations(range(len(with_size)), i)

        # https://github.com/pytorch/pytorch/issues/16501
        x = torch.tensor([[1., 0., 0., 1.],
                          [0., 1., 0., 0.],
                          [0., 1., 1., 0.],
                          [0., 1., 0., 2.]]).to_sparse()
        self.assertEqual(torch.sparse.sum(x, dim=0), torch.sparse.sum(x, dim=-2))
        self.assertEqual(torch.sum(x.to_dense(), dim=0), torch.sparse.sum(x, dim=0).to_dense())

        # not support SparseTensor.sum()
        S = self._gen_sparse(sparse_dims, nnz, with_size)[0]
        self.assertRaises(RuntimeError, lambda: S.sum())

        # dim out of range
        self.assertRaises(IndexError, lambda: torch.sparse.sum(S, 5))

        # dim 0 appears multiple times in the list of dims
        self.assertRaises(RuntimeError, lambda: torch.sparse.sum(S, [0, 0]))

        # sum an empty tensor
        empty_S = torch.sparse_coo_tensor(size=with_size)
        self.assertRaises(RuntimeError, lambda: torch.sparse.sum(empty_S, [0]))
        self.assertEqual(torch.sparse.sum(empty_S), torch.tensor(0, dtype=torch.float64))
        empty_S.requires_grad_(True)
        empty_S_sum = torch.sparse.sum(empty_S)
        empty_S_sum.backward()
        self.assertEqual(empty_S.grad.to_dense(), empty_S.clone().detach().to_dense())

        # test values().sum()
        S = self._gen_sparse(sparse_dims, nnz, with_size)[0]
        run_tests(S.requires_grad_(True))

        for test_dim in test_dims:
            S = self._gen_sparse(sparse_dims, nnz, with_size)[0]
            run_tests(S.requires_grad_(True), test_dim)

    def _test_basic_ops_shape(self, nnz_x1, nnz_x2, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 - x2
        y2 = x1.clone()
        y2.sub_(x2)
        expected = self.safeToDense(x1) - self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * x2
        y2 = x1.clone()
        y2.mul_(x2)
        expected = self.safeToDense(x1) * self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * 37.5
        y2 = x1.clone()
        y2.mul_(37.5)
        expected = self.safeToDense(x1) * 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 / 37.5
        y2 = x1.clone()
        y2.div_(37.5)
        expected = self.safeToDense(x1) / 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 // 37.5
        y2 = x1.clone()
        y2.floor_divide_(37.5)
        expected = self.safeToDense(x1) // 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # TODO: add back inplace support
        y1 = x1 ** 2
        y2 = x1.clone()
        y2 = y2.pow(2)
        expected = self.safeToDense(x1) ** 2
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y = x1.clone()
        y.zero_()
        expected = torch.zeros(x1.size())
        self.assertEqual(self.safeToDense(y), expected)

        self.assertEqual(x1.is_coalesced(), not self.is_uncoalesced)
        y = x1.coalesce()
        z = x1.coalesce()
        self.assertEqual(x1.is_coalesced(), not self.is_uncoalesced)
        self.assertTrue(y.is_coalesced())
        self.assertEqual(x1, y)
        y._values().add_(1)
        if not x1.is_coalesced():
            # check that coalesce is out of place if the original tensor is not
            # coalesced.
            self.assertEqual(z._values() + 1, y._values())
        else:
            # check that coalesce is in-place if the original tensor is
            # coalesced.
            self.assertEqual(z._values(), y._values())

    def test_basic_ops(self):
        self._test_basic_ops_shape(9, 12, [5, 6])
        self._test_basic_ops_shape(9, 12, [10, 10, 10])
        self._test_basic_ops_shape(9, 12, [50, 30, 20])
        self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5])
        self._test_basic_ops_shape(0, 12, [10, 10, 10])
        self._test_basic_ops_shape(9, 0, [10, 10, 10])
        self._test_basic_ops_shape(0, 0, [10, 10, 10])
        self._test_basic_ops_shape(0, 0, [10, 10, 0])

    def test_basic_ops_hybrid(self):
        self._test_basic_ops_shape(9, 12, [5, 6], [2, 3])
        self._test_basic_ops_shape(9, 12, [10, 10, 10], [3])
        self._test_basic_ops_shape(9, 12, [50, 30, 20], [2])
        self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [2])
        self._test_basic_ops_shape(0, 12, [10, 10, 10], [2])
        self._test_basic_ops_shape(9, 0, [10, 10, 10], [2])
        self._test_basic_ops_shape(0, 0, [10, 10, 10], [2])
        self._test_basic_ops_shape(9, 12, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(0, 12, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(9, 0, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(0, 0, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(0, 0, [10, 10, 0], [2, 0])

    def test_add_dense_sparse_mismatch(self):
        def test_shape(dense_size, sparse_dims_shape, dense_dims_shape, sparse_size):
            x = torch.zeros(dense_size, dtype=self.value_dtype, device=self.device)
            sparse_y = self.sparse_tensor(torch.zeros(sparse_dims_shape, dtype=torch.int64, device=self.device),
                                          torch.randn(dense_dims_shape, dtype=self.value_dtype, device=self.device),
                                          torch.Size(sparse_size))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "add: expected 'self' and 'other' to have same size"):
                x + sparse_y

        test_shape([3, 4], [1, 4], [4, 4, 4], [3, 4, 4])
        test_shape([3, 4, 0], [1, 4], [4, 4, 4, 0], [3, 4, 4, 0])

    def test_add_noncontiguous(self):
        indices = self.index_tensor([[1, 2], [0, 2]])
        values = self.value_tensor([1.]).expand(2, 3, 4, 5)
        x = self.sparse_tensor(indices, values)
        assert not x._values().is_contiguous()
        y = x + x
        expected = self.safeToDense(x) + self.safeToDense(x)
        self.assertEqual(self.safeToDense(y), expected)

    def _test_sparse_mask_shape(self, nnz_x1, nnz_x2, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

    def _test_sparse_mask_fixed(self):
        i = self.index_tensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.value_tensor([1, 2, 3, 4])
        x = self.sparse_tensor(i, v, torch.Size([5, 4])).coalesce()
        dense = self.value_tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ])
        exp_v = self.value_tensor([7, 14, 3, 20])
        res = dense.sparse_mask(x)
        expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4]))
        self.assertEqual(res, expected)

        i = self.index_tensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.value_empty(4, 0)
        x = self.sparse_tensor(i, v, torch.Size([5, 4, 0])).coalesce()
        dense = self.value_empty(5, 4, 0)
        exp_v = self.value_empty(4, 0)
        res = dense.sparse_mask(x)
        expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 0]))
        self.assertEqual(res, expected)

    def test_sparse_mask(self):
        self._test_sparse_mask_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6])
        self._test_sparse_mask_shape(9, 12, [10, 10, 10])
        self._test_sparse_mask_shape(9, 12, [50, 30, 20])
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5])
        self._test_sparse_mask_shape(0, 12, [10, 10, 10])
        self._test_sparse_mask_shape(9, 0, [10, 10, 10])
        self._test_sparse_mask_shape(0, 0, [10, 10, 10])
        self._test_sparse_mask_shape(0, 0, [10, 10, 0])

    def _test_sparse_mask_hybrid_fixed(self):
        i = self.index_tensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.value_tensor([[1, 2], [2, 3], [3, 4], [4, 5]])
        # TODO: This is also testing that, if coalesce is a no-op,
        # the indices don't get permuted. I don't know if we actually
        # want to give this invariant.
        x = self.sparse_tensor(i, v, torch.Size([5, 4, 2])).coalesce()
        dense = self.value_tensor([
            [[1, 3], [2, 2], [3, 3], [4, 2]],
            [[5, 7], [6, 7], [7, 9], [8, 9]],
            [[9, 2], [10, 4], [11, 1], [12, 3]],
            [[13, 5], [14, 1], [15, 1], [16, 6]],
            [[17, 7], [18, 2], [19, 7], [20, 1]],
        ])
        res = dense.sparse_mask(x)
        exp_v = self.value_tensor([[7, 9], [14, 1], [3, 3], [20, 1]])
        expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 2]))
        self.assertEqual(res, expected)

        i = self.index_tensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.value_empty(4, 2, 0)
        x = self.sparse_tensor(i, v, torch.Size([5, 4, 2, 0])).coalesce()
        dense = self.value_empty(5, 4, 2, 0)
        res = dense.sparse_mask(x)
        exp_v = self.value_empty(4, 2, 0)
        expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 2, 0]))
        self.assertEqual(res, expected)

    def test_sparse_mask_hybrid(self):
        self._test_sparse_mask_hybrid_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6], [2, 3])
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [3])
        self._test_sparse_mask_shape(9, 12, [50, 30, 20], [2])
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5], [2])
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2])
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2])
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2])
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(0, 0, [10, 10, 0], [2, 0])

    def _test_zeros(self, nnzs, shape, out_shape_i, out_shape_v=None):
        out_shape = out_shape_i + (out_shape_v or [])
        for nnz in nnzs:
            out, _, _ = self._gen_sparse(len(out_shape_i), nnz, out_shape)
            torch.zeros(*shape, out=out)
            self.assertEqual(tuple(out.size()), tuple(shape))
            self.assertTrue(out._indices().numel() == out._values().numel() == 0)
            self.assertEqual(out._nnz(), 0)
            self.assertEqual(out.sparse_dim(), len(shape))
            self.assertEqual(out.dense_dim(), 0)

    def test_zeros(self):
        def test_shape(i_shapes, v_shapes, shape, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    self._test_zeros(nnzs, shape, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 4], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 0], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 0], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 0], [9, 12])

    def _test_zeros_like(self, nnzs, template_shape_i, template_shape_v=None):
        template_shape_v = template_shape_v or []
        template_shape = template_shape_i + template_shape_v
        for nnz in nnzs:
            t, _, _ = self._gen_sparse(len(template_shape_i), nnz, template_shape)
            res = torch.zeros_like(t)
            self.assertEqual(tuple(res.size()), tuple(template_shape))
            self.assertTrue(res._indices().numel() == res._values().numel() == 0)
            self.assertEqual(res._nnz(), 0)
            self.assertEqual(res.sparse_dim(), len(template_shape_i))
            self.assertEqual(res.dense_dim(), len(template_shape_v))

    def test_zeros_like(self):
        def test_shape(i_shapes, v_shapes, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    self._test_zeros_like(nnzs, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6])
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.zeros_like(x, memory_format=mem_format)

            result = torch.zeros_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        with self.assertRaisesRegex(
            RuntimeError, r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA)' backend"
        ):
            dense_tensor = sparse_tensor.to_dense()
            result = torch.zeros_like(dense_tensor, layout=torch.sparse_coo)

    def _assert_sparse_invars(self, t):
        # SparseTensor has the following invariants:
        # - sparse_dim + dense_dim = len(SparseTensor.shape)
        # - SparseTensor._indices().shape = (sparse_dim, nnz)
        # - SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
        self.assertEqual(t.sparse_dim() + t.dense_dim(), len(t.shape))
        self.assertEqual(tuple(t._indices().shape), (t.sparse_dim(), t._nnz()))
        self.assertEqual(tuple(t._values().shape), (t._nnz(), ) + t.shape[t.sparse_dim():])

    def _test_empty_like(self, sparse_tensor):

        result = torch.empty_like(sparse_tensor)
        self.assertTrue(result.is_sparse)
        self._assert_sparse_invars(result)
        self.assertEqual(result.shape, sparse_tensor.shape)
        self.assertEqual(result.dtype, sparse_tensor.dtype)
        self.assertEqual(result.device, sparse_tensor.device)
        self.assertEqual(result.sparse_dim(), sparse_tensor.sparse_dim())
        self.assertEqual(result.dense_dim(), sparse_tensor.dense_dim())

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6])
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.empty_like(x, memory_format=mem_format)

            result = torch.empty_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        with self.assertRaisesRegex(
            RuntimeError, r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA)' backend"
        ):
            dense_tensor = sparse_tensor.to_dense()
            result = torch.empty_like(dense_tensor, layout=torch.sparse_coo)

    def test_empty_like(self):
        # tests https://github.com/pytorch/pytorch/issues/43699

        if not self.is_uncoalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                device=self.device
            ).coalesce()
            self._test_empty_like(input_coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                device=self.device
            ).coalesce()
            self._test_empty_like(input_coalesced)

        if self.is_uncoalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                device=self.device
            )
            self._test_empty_like(input_uncoalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                device=self.device
            )
            self._test_empty_like(input_uncoalesced)

    def _test_narrow(self, input, narrow_args):
        expected = input.to_dense().narrow(*narrow_args)
        self.assertEqual(expected, input.narrow_copy(*narrow_args).to_dense())

    def _all_narrow_combs(self, shape):
        for dim, dim_sz in enumerate(shape):
            for start in range(dim_sz):
                for length in range(dim_sz - start):
                    yield [dim, start, length]

    def test_narrow(self):
        shape = [3, 3, 4, 2]
        input, _, _ = self._gen_sparse(4, 19, shape)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(input, narrow_args)

        self.assertRaises(RuntimeError, lambda: input.narrow_copy(-1, 0, 3))  # dim < 0
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(10, 0, 3))  # dim > input.dim()
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, shape[0] + 1, 3))  # start > size of dim
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, 2, shape[0]))  # start+length > size of dim

        with_dense, _, _ = self._gen_sparse(2, 7, shape)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(with_dense, narrow_args)

        self.assertRaises(RuntimeError, lambda: with_dense.narrow_copy(10, 0, 3))  # dim > sparseDim + denseDim

    def _test_log1p_tensor(self, sparse_tensor):
        def is_integral(dtype):
            return dtype in torch.testing.get_all_int_dtypes()

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.log1p()
        is_integral_dtype = is_integral(sparse_tensor.dtype)
        self.assertEqual(expected_output, sparse_tensor.log1p().to_dense())
        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "log1p: result type cannot be Integral, got:"):
                sparse_tensor.coalesce().log1p_()
        else:
            self.assertEqual(expected_output, sparse_tensor.coalesce().log1p_().to_dense())

        if self.is_uncoalesced and not is_integral_dtype:
            # test in-place op on uncoalesced input
            with self.assertRaisesRegex(RuntimeError, "in-place on uncoalesced tensors is not supported"):
                sparse_tensor.log1p_()
        elif self.is_uncoalesced and is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "log1p: result type cannot be Integral, got"):
                sparse_tensor.log1p_()

        if not is_integral_dtype:
            sparse_tensor.requires_grad_()
            self.assertTrue(sparse_tensor.requires_grad)

            # test autograd
            x = sparse_tensor.clone()
            y = sparse_tensor.log1p()
            with self.assertRaisesRegex(RuntimeError, "log1p of a sparse tensor is made to be non-differentiable"):
                y.backward(x)
        else:
            with self.assertRaisesRegex(RuntimeError, "only Tensors of floating point dtype can require gradients"):
                sparse_tensor.requires_grad_()

    def test_log1p(self):
        for dtype in torch.testing.get_all_dtypes(include_bool=False, include_half=False,
                                                  include_bfloat16=False, include_complex=False):
            if not self.is_uncoalesced:
                input_coalesced = torch.sparse_coo_tensor(
                    indices=torch.tensor([[0], [1], [2]]).transpose(1, 0),
                    values=torch.tensor([3.0, 4.0, 5.0]),
                    size=[3, ],
                    device=self.device,
                    dtype=dtype
                ).coalesce()
                self._test_log1p_tensor(input_coalesced)

                # hybrid sparse input
                input_coalesced = torch.sparse_coo_tensor(
                    indices=torch.tensor([[1, 3], [2, 4]]),
                    values=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
                    size=[4, 5, 2],
                    device=self.device,
                    dtype=dtype
                ).coalesce()
                self._test_log1p_tensor(input_coalesced)

            if self.is_uncoalesced:
                # test uncoalesced input
                input_uncoalesced = torch.sparse_coo_tensor(
                    indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                    values=torch.tensor([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
                    size=[3, ],
                    device=self.device,
                    dtype=dtype
                )
                self._test_log1p_tensor(input_uncoalesced)

                # test on empty sparse tensor
                input_uncoalesced = torch.sparse_coo_tensor(
                    indices=torch.zeros([2, 0]),
                    values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                    size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                    device=self.device,
                    dtype=dtype
                )
                self._test_log1p_tensor(input_uncoalesced)

    def _test_neg_negative(self, sparse_tensor):
        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.neg()

        ops = (
            torch.neg, torch.Tensor.neg, torch.Tensor.neg_,
            torch.negative, torch.Tensor.negative, torch.Tensor.negative_,
            operator.neg
        )
        for op in ops:
            sparse_tensor_copy = sparse_tensor.clone()
            self.assertEqual(expected_output, op(sparse_tensor_copy).to_dense())

            if op in (torch.neg, torch.negative):
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                op(sparse_tensor, out=sparse_tensor_out)
                self.assertEqual(expected_output, sparse_tensor_out.to_dense())

    def test_neg_negative(self):

        if not self.is_uncoalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                device=self.device
            ).coalesce()
            self._test_neg_negative(input_coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                device=self.device
            ).coalesce()
            self._test_neg_negative(input_coalesced)

        if self.is_uncoalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                device=self.device
            )
            self._test_neg_negative(input_uncoalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                device=self.device
            )
            self._test_neg_negative(input_uncoalesced)

    def _test_asin_arcsin(self, sparse_tensor):
        def is_integral(dtype):
            return dtype in torch.testing.get_all_int_dtypes()
        is_integral_dtype = is_integral(sparse_tensor.dtype)

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.asin()

        ops = (
            torch.asin, torch.Tensor.asin,
            torch.arcsin, torch.Tensor.arcsin,
        )
        for op in ops:
            self.assertEqual(expected_output, op(sparse_tensor).to_dense())
            if op in (torch.asin, torch.arcsin):
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                if not is_integral_dtype:
                    op(sparse_tensor, out=sparse_tensor_out)
                    self.assertEqual(expected_output, sparse_tensor_out.to_dense())
                else:
                    with self.assertRaisesRegex(RuntimeError, "asin: result type cannot be Integral"):
                        op(sparse_tensor, out=sparse_tensor_out)

        for op in (torch.Tensor.asin_, torch.Tensor.arcsin_):
            if is_integral_dtype:
                # test coalesce on integral dtype tensor
                with self.assertRaisesRegex(RuntimeError, "asin: result type cannot be Integral"):
                    op(sparse_tensor.clone().coalesce()).to_dense()
            else:
                self.assertEqual(expected_output, op(sparse_tensor.clone().coalesce()).to_dense())

            if self.is_uncoalesced and not is_integral_dtype:
                # test in-place op on uncoalesced input
                with self.assertRaisesRegex(RuntimeError, "in-place on uncoalesced tensors is not supported"):
                    op(sparse_tensor)
            elif self.is_uncoalesced:
                # test in-place op on integral dtype tensor
                with self.assertRaisesRegex(RuntimeError, "asin: result type cannot be Integral"):
                    op(sparse_tensor)

    def test_asin_arcsin(self):
        for dtype in torch.testing.get_all_dtypes(include_bool=False, include_half=False,
                                                  include_bfloat16=False, include_complex=False):
            if not self.is_uncoalesced:
                input_coalesced = torch.sparse_coo_tensor(
                    indices=torch.tensor([[0, 1, 2, 3]]),
                    values=torch.tensor([0.5, -0.5, 0.7, -0.7]),
                    size=[4, ],
                    dtype=dtype,
                    device=self.device
                ).coalesce()
                self._test_asin_arcsin(input_coalesced)

                # hybrid sparse input
                input_coalesced = torch.sparse_coo_tensor(
                    indices=torch.tensor([[1, 3], [2, 4]]),
                    values=torch.tensor([[-0.1, 0.24], [-0.44, 0.1]]),
                    size=[4, 5, 2],
                    dtype=dtype,
                    device=self.device
                ).coalesce()
                self._test_asin_arcsin(input_coalesced)

            if self.is_uncoalesced:
                # test uncoalesced input
                input_uncoalesced = torch.sparse_coo_tensor(
                    indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                    values=torch.tensor([0.3, -0.3, -0.4, 0.3, -0.5, 0.15]),
                    size=[3, ],
                    dtype=dtype,
                    device=self.device
                )
                self._test_asin_arcsin(input_uncoalesced)

                # test on empty sparse tensor
                input_uncoalesced = torch.sparse_coo_tensor(
                    indices=torch.zeros([2, 0]),
                    values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                    size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                    dtype=dtype,
                    device=self.device
                )
                self._test_asin_arcsin(input_uncoalesced)

    def test_mv(self):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [di, dj])
            t = torch.randn(dk, device=self.device)

            res = x.matmul(t)
            expected = self.safeToDense(x).matmul(t)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 1000, 20)
        test_shape(64, 10000, 10000, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 0, 0)
        test_shape(10, 100, 100, 0)
        test_shape(10, 100, 100, 20)

        with self.assertRaisesRegex(RuntimeError, r"mv: expected self\.size\(-1\) == vec\.size\(-1\)"):
            test_shape(10, 100, 10, 20)

        with self.assertRaisesRegex(RuntimeError, "mv: two tensor dim should be 2 and 1"):
            x, _, _ = self._gen_sparse(2, 20, [10, 100])
            y, _, _ = self._gen_sparse(2, 20, [10, 100])
            res = x.mv(y)

    def test_sparse_add_coalesce(self):
        i = self.index_tensor([[1, 2, 1]])
        v = self.value_tensor([3, 4, 5])
        x = self.sparse_tensor(i, v, torch.Size([3]))
        y = self.sparse_tensor(i, v, torch.Size([3]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

        i = self.index_tensor([[1, 2, 1]])
        v = self.value_empty(3, 0)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        y = self.sparse_tensor(i, v, torch.Size([3, 0]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

    @cuda_only
    def test_storage_not_null(self):
        x = torch.cuda.sparse.FloatTensor(2)
        self.assertNotEqual(x.get_device(), -1)

        x = torch.cuda.sparse.FloatTensor(2, 0)
        self.assertNotEqual(x.get_device(), -1)

    @cuda_only
    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_same_gpu(self):
        def check_device(x, device_id):
            self.assertEqual(x.get_device(), device_id)
            self.assertEqual(x._values().get_device(), device_id)
            self.assertEqual(x._indices().get_device(), device_id)

        i = self.index_tensor([[2]]).cuda(1)
        v = self.value_tensor([5]).cuda(1)
        x = self.sparse_tensor(i, v, torch.Size([3]), device=1)
        check_device(x, 1)

        i = self.index_tensor([[2]]).cuda(1)
        v = self.value_empty(1, 0).cuda(1)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]), device=1)
        check_device(x, 1)

        x = self.sparse_empty(3, device=1)
        check_device(x, 1)

        x = self.sparse_empty(3, 0, device=1)
        check_device(x, 1)

        i = self.index_tensor([[2]]).cuda(1)
        v = self.value_tensor([5]).cuda(0)
        # NB: non-legacy constructor allows this and moves indices
        self.assertRaises(RuntimeError, lambda: self.legacy_sparse_tensor(i, v, torch.Size([3])))

        i = self.index_tensor([[2]]).cuda(1)
        v = self.value_empty(1, 0).cuda(0)
        # NB: non-legacy constructor allows this and moves indices
        self.assertRaises(RuntimeError, lambda: self.legacy_sparse_tensor(i, v, torch.Size([3, 0])))

    def _test_new_device(self, size, device):
        with torch.cuda.device(device):
            x = torch.cuda.sparse.DoubleTensor(*size)
        self.assertEqual(x.get_device(), device)
        x1 = x.new()
        x2 = x.new(2, 3)
        self.assertEqual(x1.get_device(), device)
        self.assertEqual(x2.get_device(), device)

    @cuda_only
    def test_new_device_single_gpu(self):
        self._test_new_device((), 0)
        self._test_new_device((30, 20), 0)
        self._test_new_device((30, 20, 10), 0)
        self._test_new_device((30, 20, 10, 0), 0)

    @cuda_only
    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_new_device_multi_gpu(self):
        self._test_new_device((), 1)
        self._test_new_device((30, 20), 1)
        self._test_new_device((30, 20, 10), 1)
        self._test_new_device((30, 20, 10, 0), 1)

    def test_new(self):
        def test_shape(sparse_dims, nnz, with_size):
            x, indices, values = self._gen_sparse(sparse_dims, nnz, with_size)
            if not x.is_cuda:
                # CUDA sparse tensors currently requires the size to be
                # specified if nDimV > 0
                out = x.new(indices, values).coalesce()
                x_c = x.coalesce()
                self.assertEqual((out.indices(), out.values()), (x_c.indices(), x_c.values()))
            self.assertEqual(x.new(indices, values, x.size()), x)

        test_shape(3, 10, 100)
        test_shape(3, 0, [100, 100, 0])

    @cpu_only  # not really, but we only really want to run this once
    def test_factory(self):
        for test_empty_tensor in [True, False]:
            if test_empty_tensor:
                default_size = torch.Size([1, 3, 0])
                size = torch.Size([3, 3, 0])
            else:
                default_size = torch.Size([1, 3])
                size = torch.Size([3, 3])
            for include_size in [True, False]:
                for use_tensor_idx in [True, False]:
                    for use_tensor_val in [True, False]:
                        for use_cuda in ([False] if not torch.cuda.is_available() else [True, False]):
                            for dtype in [torch.float64, torch.float16]:
                                # have to include size with cuda sparse tensors
                                include_size = include_size or use_cuda
                                long_dtype = torch.int64
                                device = torch.device('cpu') if not use_cuda else \
                                    torch.device(torch.cuda.device_count() - 1)
                                indices = torch.tensor(([0], [2]), dtype=long_dtype) if use_tensor_idx else ([0], [2])
                                if test_empty_tensor:
                                    values = self.value_empty(1, 0).to(dtype)
                                else:
                                    if use_tensor_val:
                                        values = torch.tensor([1.], dtype=dtype)
                                    else:
                                        values = 1.
                                if include_size:
                                    sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=dtype,
                                                                            device=device, requires_grad=True)
                                else:
                                    sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=dtype,
                                                                            device=device, requires_grad=True)
                                self.assertEqual(indices, sparse_tensor._indices())
                                self.assertEqual(values, sparse_tensor._values())
                                self.assertEqual(size if include_size else default_size, sparse_tensor.size())
                                self.assertEqual(dtype, sparse_tensor.dtype)
                                if use_cuda:
                                    self.assertEqual(device, sparse_tensor._values().device)
                                self.assertEqual(True, sparse_tensor.requires_grad)

    def test_factory_size_check(self):
        indices = self.index_tensor([[1, 2],
                                    [0, 2]])
        values = self.value_tensor([.5, .5])
        sizes = torch.Size([2, 3])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, "found negative index"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]])
        values = self.value_empty(2, 1, 0)
        sizes = torch.Size([2, 3, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]])
        values = self.value_empty(2, 2, 2)
        sizes = torch.Size([0, 0, 2, 2])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]])
        values = self.value_tensor([[1, 1, 1], [1, 1, 1]])
        sizes = torch.Size([3, 3, 2])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]])
        values = self.value_empty(2, 1, 0)
        sizes = torch.Size([3, 3, 2, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

    def test_factory_default(self):
        tensor = self.legacy_sparse_tensor()
        expected_indices = self.index_tensor([[]])
        expected_size = torch.Size([0])
        self.assertEqual(tensor._indices(), expected_indices)
        self.assertEqual(tensor.shape, expected_size)

    def test_factory_empty_indices(self):
        device = 'cuda' if self.is_cuda else 'cpu'
        tensor = self.legacy_sparse_tensor()
        expected_indices = torch.empty((1, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 0]), device=device)
        expected_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0]), device=device)
        expected_indices = torch.empty((3, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0, 0]), device=device)
        expected_indices = torch.empty((4, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

    def test_factory_nnz(self):
        indices = self.index_tensor([[0]])  # (sparse_dim, nnz): (1, 1)
        values = self.value_tensor([[1, 1], [1, 1]])  # (nnz, ...): (2, 2)
        sizes = torch.Size([2, 2])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[0]])  # (sparse_dim, nnz): (1, 1)
        values = self.value_empty(2, 0)  # (nnz, ...): (2, 0)
        sizes = torch.Size([2, 0])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes)

    def test_factory_nnz_zero(self):
        def test_shape(i_shape, v_shape, size, expected_size):
            device = 'cuda' if self.is_cuda else 'cpu'
            if size:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), torch.Size(size), device=device)
            else:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), device=device)
            expected_indices = torch.empty(i_shape, device=device, dtype=torch.int64)
            expected_values = torch.empty(v_shape, device=device, dtype=torch.float64)
            expected_size = torch.Size(expected_size)
            self.assertEqual(t._indices(), expected_indices)
            self.assertEqual(t._values(), expected_values)
            self.assertEqual(t.size(), expected_size)

        test_shape([1, 0], [0, 2, 4, 0], None, [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], None, [0, 0, 0, 2, 4, 0])
        test_shape([1, 0], [0, 2, 4, 0], [0, 2, 4, 0], [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [0, 0, 0, 2, 4, 0], [0, 0, 0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [1, 2, 3, 2, 4, 0], [1, 2, 3, 2, 4, 0])

    def test_factory_dense_dim(self):
        indices = self.index_tensor([[0]])
        values = self.value_tensor([[[1, 1, 1], [1, 1, 1]]])
        sizes = torch.Size([1, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[0]])
        values = self.value_empty(1, 2, 3, 0)
        sizes = torch.Size([1, 3, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

    @cpu_only
    def test_factory_type_inference(self):
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=torch.float16))
        self.assertEqual(torch.float16, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=torch.float32))
        self.assertEqual(torch.float32, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=torch.float64))
        self.assertEqual(torch.float64, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1]))
        self.assertEqual(torch.int64, t.dtype)

        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.HalfTensor(1, 0))
        self.assertEqual(torch.float16, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.FloatTensor(1, 0))
        self.assertEqual(torch.float32, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.DoubleTensor(1, 0))
        self.assertEqual(torch.float64, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.LongTensor(1, 0))
        self.assertEqual(torch.int64, t.dtype)

    @cuda_only
    def test_factory_device_type_inference(self):
        # both indices/values are CUDA

        cpu_cuda = ('cpu', 'cuda')
        cpu_cuda_none = cpu_cuda + (None,)
        for indices_device, values_device, device in itertools.product(cpu_cuda,
                                                                       cpu_cuda,
                                                                       cpu_cuda_none):
            indices = torch.tensor(([0], [2]), device=indices_device)
            values = torch.tensor([1.], device=values_device)
            empty_values = self.value_empty(1, 0).to(values_device)
            shape = (1, 3)
            empty_shape = (1, 3, 0)
            if device is None and indices_device != values_device:
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, values, shape, device=device)
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
            else:
                t = torch.sparse_coo_tensor(indices, values, shape, device=device)
                t_empty = torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
                should_be_cuda = (device == 'cuda' or (device is None and values_device == 'cuda'))
                self.assertEqual(should_be_cuda, t.is_cuda)
                self.assertEqual(t.is_cuda, t_empty.is_cuda)

    @cpu_only
    def test_factory_copy(self):
        def test_tensor(indices, values, indices_equal, values_equal):
            sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=torch.float64)
            if indices_equal:
                self.assertEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            else:
                self.assertNotEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            if values_equal:
                self.assertEqual(values.data_ptr(), sparse_tensor._values().data_ptr())
            else:
                self.assertNotEqual(values.data_ptr(), sparse_tensor._values().data_ptr())

        # both correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, True, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, True, True)

        # only indices correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float16)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, True, True)  # An empty tensor's data_ptr is always equal to 0

        # only values correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, False, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, False, True)

        # neither correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, False, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, False, True)  # An empty tensor's data_ptr is always equal to 0

    @cpu_only  # just run once, we test both cpu and cuda
    def test_constructor_device_legacy(self):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])

        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(i, v, device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(i, v, size, device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))

        x = torch.sparse_coo_tensor(i, v, size, device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))

        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(i, v, device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(i, v, size, device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))

            x = torch.sparse_coo_tensor(i, v, size, device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))

    def test_legacy_constructor(self):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])

        self.assertRaises(TypeError, lambda: torch.sparse.FloatTensor(v.storage()))
        self.assertRaises(TypeError, lambda: torch.sparse.FloatTensor(v))
        self.assertEqual(torch.sparse_coo, torch.sparse.FloatTensor(torch.Size([2, 3])).layout)
        self.assertRaises(TypeError, lambda: torch.sparse.FloatTensor([6]))

    def test_legacy_new(self):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])
        s = torch.sparse_coo_tensor(i, v, size)

        self.assertEqual(torch.sparse_coo, s.new(device='cpu').layout)
        self.assertRaises(TypeError, lambda: s.new(v.storage()))
        self.assertRaises(TypeError, lambda: s.new(v))
        self.assertEqual(torch.sparse_coo, s.new(torch.Size([2, 3])).layout)
        self.assertRaises(TypeError, lambda: s.new([6]))

    @cpu_only  # not really, but we only really want to run this once
    def test_dtypes(self):
        all_sparse_dtypes = torch.testing.get_all_dtypes()
        do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        if torch.cuda.is_available():
            do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))

    @cpu_only  # not really, but we only really want to run this once
    def test_empty_full(self):
        all_sparse_dtypes = torch.testing.get_all_dtypes()
        do_test_empty_full(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        if torch.cuda.device_count() > 0:
            do_test_empty_full(self, all_sparse_dtypes, torch.sparse_coo, None)
            do_test_empty_full(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))

    def test_is_sparse(self):
        x = torch.randn(3, 3)
        self.assertFalse(x.is_sparse)

        x = torch.randn(3, 3, 0)
        self.assertFalse(x.is_sparse)

        x = self.legacy_sparse_tensor()
        self.assertTrue(x.is_sparse)

        x = self.sparse_empty(1, 0)
        self.assertTrue(x.is_sparse)

    def test_resize_as(self):
        def do_test(t):
            y = t.new().resize_as_(t).zero_()
            self.assertEqual(y.shape, t.shape)
            # Check that y can be added to t. Currently, this requires that
            # sparse_dim and dense_dim match.
            self.assertEqual(t, t + y)

        do_test(self.legacy_sparse_tensor())
        do_test(self.sparse_empty(3, 0))
        do_test(self.sparse_empty(3, 3))

    def _test_resize_shape(self, x_i, x_v, x_size, y_i, y_v, y_size):
        x_v_numel = torch.zeros(x_v).numel()
        y_v_numel = torch.zeros(y_v).numel()
        x = torch.sparse_coo_tensor(torch.zeros(x_i),
                                    torch.arange(x_v_numel).resize_(x_v).to(torch.float),
                                    torch.Size(x_size))
        x_dense = x.to_dense()
        y = torch.sparse_coo_tensor(torch.zeros(y_i),
                                    torch.ones(y_v).to(torch.float),
                                    torch.Size(y_size))
        y_dense = y.to_dense()
        x.resize_as_(y)
        x_dense.resize_as_(y_dense)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.sparse_dim(), y.sparse_dim())
        self.assertEqual(x.dense_dim(), y.dense_dim())
        self.assertEqual(x.shape, x_dense.shape)
        self.assertEqual(y.shape, y_dense.shape)
        # Here we make sure that the original data are preserved after resizing
        self.assertEqual(x.to_dense().view(-1)[0:x_v_numel].view(x_v),
                         x_dense.view(-1)[0:x_v_numel].view(x_v))

    def test_resize(self):
        # 1. Expand the size of some dense dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 4], [2, 2, 4])

        self._test_resize_shape([1, 1], [1, 2, 0], [2, 2, 0],
                                [1, 1], [1, 2, 4], [2, 2, 4])

        # 2. Expand the size of some sparse dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 3], [4, 2, 3])

        # 3. Change the shapes of both sparse and dense dimensions when nnz is zero [Supported]
        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 5], [1, 1, 2, 4, 5])

        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 0], [1, 1, 2, 4, 0])

        # 4. Add dims to dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 4], [2, 2, 3, 4])

        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 0], [2, 2, 3, 0])

        # 5. Remove dims from dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2], [2, 2])

        # 6. Change the number of sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [2, 1], [1, 2, 3], [1, 2, 2, 3])

        # 7. Shrink the size of some sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3], [1, 2, 3])

        # 8. Shrink the size of some dense dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 2], [2, 2, 2])

        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 0], [2, 2, 0])

    def test_is_nonzero(self):
        self.assertTrue(torch.sparse_coo_tensor(([0],), 1., (1,)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0., (1,)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0], [0]), 0., (1, 1)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (0., 0.), (1,)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (-1., 1.), (1,)).is_nonzero())
        self.assertTrue(torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, []).is_nonzero())  # scalar sparse tensor
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.sparse_coo_tensor(([0, 1],), self.value_empty(2, 0), (4, 0)).is_nonzero()

    def test_allow_tensor_metadata_change(self):
        def do_test(t):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "raw_resize_ is not allowed on a Tensor created from .data or .detach()"):
                t.transpose_(0, 1)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "resize_ is not allowed on a Tensor created from .data or .detach()"):
                t.resize_as_(self.sparse_empty(3, 3))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "resize_and_clear_ is not allowed on a Tensor created from .data or .detach()"):
                t.mul_(t)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "set_coalesced is not allowed on a Tensor created from .data or .detach()"):
                t._coalesced_(True)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "set_indices_and_values_unsafe is not allowed on a Tensor created from .data or .detach()"):
                a = self.sparse_tensor(torch.tensor([[0, 1, 1], [2, 0, 2]]), torch.tensor([3., 4., 5.])).data
                a.add_(a)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "resize_and_clear_ is not allowed on a Tensor created from .data or .detach()"):
                a.zero_()
            with self.assertRaisesRegex(
                    RuntimeError,
                    "resize_ is not allowed on a Tensor created from .data or .detach()"):
                a.copy_(self.sparse_empty(3, 3))

        do_test(self.sparse_empty(3, 0).data)
        do_test(self.sparse_empty(3, 0).detach())

    def test_change_tensor_metadata(self):
        i = self.index_tensor([[0], [1]])
        v = self.value_tensor([[3, 4, 5]])
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.resize_(2, 3)
        v.resize_(4, 5)
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]])
        v = self.value_tensor([[3, 4, 5]])
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.resize_as_(self.index_tensor([0, 1]))
        v.resize_as_(self.value_tensor([3, 4, 5]))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]])
        v = self.value_tensor([[3, 4, 5]])
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.as_strided_((2, 1), (1, 1))
        v.as_strided_((1, 3), (1, 1))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]])
        v = self.value_tensor([[3, 4, 5]])
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.set_(self.index_tensor([0, 1]))
        v.set_(self.value_tensor([3, 4, 5]))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]])
        v = self.value_tensor([[3, 4, 5]])
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.transpose_(0, 1)
        v.transpose_(0, 1)
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

    def test_pickle(self):
        import pickle

        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]

        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=self.device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if self.is_uncoalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=self.value_dtype,
                                  device=self.device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape)
            serialized = pickle.dumps(sp_tensor)
            sp_tensor_loaded = pickle.loads(serialized)
            self.assertEqual(sp_tensor, sp_tensor_loaded)

    def test_any(self):
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([False, False]))
        t_any = torch.tensor(False)
        self.assertEqual(torch.any(t), t_any)
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([True, False]))
        t_any = torch.tensor(True)
        self.assertEqual(torch.any(t), t_any)

    def test_isnan(self):
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([1, 4]))
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([False, False]))
        self.assertEqual(torch.isnan(t).int(), t_nan.int())
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([1, float("nan")]))
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([False, True]))
        self.assertEqual(torch.isnan(t).int(), t_nan.int())

    def test_div_by_sparse_error(self):
        self.assertRaisesRegex(RuntimeError, 'Sparse division requires',
                               lambda: torch.tensor(1., device=self.device).to_sparse()
                               / torch.tensor(1., device=self.device).to_sparse())

    def test_floor_divide_by_sparse_error(self):
        self.assertRaisesRegex(RuntimeError, 'Sparse floor division requires',
                               lambda: torch.tensor(1., device=self.device).to_sparse()
                               // torch.tensor(1., device=self.device).to_sparse())

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_sparse_to_numpy(self):
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([1, 4]))
        self.assertRaises(TypeError, lambda: t.numpy())

    def test_softmax(self):
        import torch.nn.functional as F

        def to_dense(sparse, fill_value=None):
            """
            Return dense tensor from a sparse tensor using given fill value.
            """
            if fill_value is None or fill_value == 0:
                return sparse.to_dense()
            sparse = sparse.coalesce()
            dense = torch.full(sparse.shape, fill_value, dtype=sparse.dtype, device=sparse.device)
            for idx, value in zip(sparse._indices().t(), sparse._values()):
                dense[tuple(idx)] = value
            return dense

        def softmax_to_dense(sparse, dim):
            """Dense softmax of a sparse tensor. Useful only for testing softmax
            correctness.

            When computing softmax of a sparse tensor, the value of
            unspecified items is negative infinity rather than zero so
            that

              softmax(sparse.to_dense(fill_value=-inf), dim) == softmax(sparse, dim).to_dense()

            holds for non-empty lines. One empty lines, the softmax
            values are defined as 0 in order to preserve the sparsity
            of result.

            Note that in PyTorch, ``to_dense`` method does not
            implement the ``fill_value`` keyword argument.
            """
            dtype = sparse.dtype
            device = sparse.device
            dense = to_dense(sparse, fill_value=-float('inf'))
            r = F.softmax(dense, dim)
            # softmax on empty lines results nan, replace with zeros to match the definition
            r[r != r] = 0
            return r

        def sparse_softmax(sparse, dim):
            """Pure Python softmax of a sparse tensor. Assuming -inf for
            unspecified sparse tensor data. This is a prototype of
            sparse softmax algorithm in Python.
            """
            dtype = sparse.dtype
            device = sparse.device

            # softmax is non-linear operation, so sparse tensors must
            # be coalesced.
            sparse = sparse.coalesce()
            inf = float('inf')
            indices = sparse._indices()
            values = sparse._values()

            if dim < sparse.sparse_dim():
                nnz = sparse._nnz()

                # compute pool indices
                size = sparse.size()
                strides = torch.ones((sparse.sparse_dim(), 1), dtype=indices.dtype, device=indices.device)
                for i in reversed(range(sparse.sparse_dim() - 1)):
                    strides[i, 0] = strides[i + 1, 0] * size[i + 1]
                strides[dim, 0] = 0

                pool = (indices * strides).sum(dim=0)
                i2p = {}
                for i in range(nnz):
                    c = int(pool[i])
                    if c not in i2p:
                        i2p[c] = len(i2p)
                    pool[i] = i2p[c]

                # compute max
                dense_size = tuple(size[sparse.sparse_dim():])
                mx = torch.empty((pool.max() + 1,) + dense_size, dtype=dtype, device=device)
                mx[:] = -inf
                for n in range(nnz):
                    p = pool[n]
                    mx[p] = torch.max(mx[p], values[n])

                # apply exp to (v - mx) and sum the results
                exp_values = torch.empty_like(values)
                exp_sums = torch.zeros_like(mx)
                for n in range(nnz):
                    p = pool[n]
                    v = exp_values[n] = (values[n] - mx[p]).exp()
                    exp_sums[p] = exp_sums[p] + v

                # normalize with the sum of exponents
                for n in range(nnz):
                    p = pool[n]
                    exp_values[n] = exp_values[n] / exp_sums[p]

                return torch.sparse_coo_tensor(indices,
                                               exp_values,
                                               sparse.size(),
                                               dtype=dtype, device=device)

            elif dim < sparse.sparse_dim() + sparse.dense_dim():
                return torch.sparse_coo_tensor(indices,
                                               F.softmax(values, dim - sparse.sparse_dim() + 1),
                                               sparse.size(),
                                               dtype=dtype, device=device)
            else:
                raise ValueError(
                    '`dim(=%s)` must be smaller than `sparse_dim(=%s) + dense_dim(=%s)`'
                    % (dim, sparse.sparse_dim(), sparse.dense_dim()))

        def softmax_jacobian_analytic(x, dim):
            """Return Jacobian of softmax using analytic formula

               D_jS_i = S_i * (1[i==j] - S_j).

            where S = softmax(x, dim), x is dense tensor, i,j in
            range(x.shape[dim]).
            """
            y = F.softmax(x, dim)
            y[y != y] = 0  # replace nan-s with zeros
            J = torch.zeros((x.shape[dim],) + tuple(x.shape), dtype=x.dtype, device=x.device)
            si = [slice(None)] * len(y.shape)
            sj = [slice(None)] * len(y.shape)
            s = [slice(None)] * len(J.shape)
            for i in range(y.shape[dim]):
                si[dim] = i
                s[dim + 1] = i
                yi = y[tuple(si)]
                for j in range(y.shape[dim]):
                    sj[dim] = j
                    s[0] = j
                    if i == j:
                        J[tuple(s)] = yi * (1 - yi)
                    else:
                        yj = y[tuple(sj)]
                        J[tuple(s)] = - yi * yj
                    sj[dim] = slice(None)
                si[dim] = slice(None)
                s[dim + 1] = slice(None)
            return J

        def softmax_jacobian_autograd(x, dim, log=False):
            """Return Jacobian of softmax using PyTorch autograd feature.

            x can be dense or sparse tensor.
            """
            import itertools

            if x.is_sparse:
                x = x.coalesce()

            dtype = x.dtype
            device = x.device
            shape = tuple(x.shape)
            J = torch.zeros((shape[dim],) + shape, dtype=dtype, device=device)
            for i in range(shape[dim]):
                if x.is_sparse:
                    sparse_dim = x.sparse_dim()
                    dense_dim = x.dense_dim()
                    if dim < sparse_dim:
                        ranges = []
                        for j, sz in enumerate(shape[:sparse_dim]):
                            if dim == j:
                                ranges.append([i])
                            else:
                                ranges.append(list(range(sz)))
                        indices = torch.tensor(list(itertools.product(*ranges)), dtype=torch.long, device=device).t()
                        values = torch.ones((indices.shape[1],) + shape[sparse_dim:], dtype=dtype, device=device)
                    else:
                        ranges = []
                        for j, sz in enumerate(shape[:sparse_dim]):
                            ranges.append(list(range(sz)))
                        indices = torch.tensor(list(itertools.product(*ranges)), dtype=torch.long, device=device).t()
                        values = torch.zeros((indices.shape[1],) + shape[sparse_dim:], dtype=dtype, device=device)
                        sv = [slice(None)] * (dense_dim + 1)
                        sv[dim - sparse_dim + 1] = i
                        values[tuple(sv)] = 1
                    v = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)
                else:
                    v = torch.zeros_like(x)
                    sv = [slice(None)] * len(v.shape)
                    sv[dim] = i
                    v[tuple(sv)] = 1
                x_ = x.clone()
                x_.requires_grad_(True)

                if log:
                    if x_.is_sparse:
                        y = torch.sparse.log_softmax(x_, dim)
                    else:
                        y = F.log_softmax(x_, dim)
                else:
                    if x_.is_sparse:
                        y = torch.sparse.softmax(x_, dim)
                    else:
                        y = F.softmax(x_, dim)
                        # replace nan-s with zeros
                        y.data[y != y] = 0
                y.backward(v)
                g = x_.grad
                if not g.is_sparse:
                    # replace nan-s with zeros
                    g.data[g != g] = 0
                J[i] = g.to_dense() if g.is_sparse else g
            return J

        def test_op(sparse_dims, nnz, with_size):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims

            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size)

            def sparse_log(x):
                return torch.sparse_coo_tensor(x._indices(), x._values().log(),
                                               x.size(), dtype=x.dtype, device=x.device)

            for dim in range(x.sparse_dim() + x.dense_dim()):
                # Check sparse softmax definition

                # check Python sparse softmax
                y = sparse_softmax(x, dim)
                r1 = softmax_to_dense(x, dim)
                r2 = y.to_dense()
                self.assertEqual(r1, r2)

                # check C++ sparse softmax
                y1 = torch.sparse.softmax(x, dim)
                self.assertEqual(y, y1)

                # check C++ sparse log_softmax
                ly1 = torch.sparse.log_softmax(x, dim)
                self.assertEqual(ly1, sparse_log(y1))

                # Check autograd support on sparse softmax

                # check softmax Jacobian definition for dense input
                x1 = to_dense(x, fill_value=float('-inf'))
                J = softmax_jacobian_analytic(x1, dim)
                assert J.shape[0] == x.shape[dim]
                assert J.shape[dim + 1] == x.shape[dim]

                # check softmax Jacobian from autograd, dense input
                J2 = softmax_jacobian_autograd(x1, dim)
                self.assertEqual(J, J2)

                # check softmax Jacobian from autograd, sparse input
                J3 = softmax_jacobian_autograd(x, dim)
                self.assertEqual(J, J3)

                '''
                y = softmax(x, dim)
                z = log(y) = log_softmax(x, dim)
                Dy/Dx = J
                Dz/Dx = Dz/Dy Dy/Dx = 1/y * J
                => J = J_log * y
                '''
                # log_softmax Jacobian from autograd, dense input
                J2_log = softmax_jacobian_autograd(x1, dim, log=True)

                # log_softmax Jacobian from autograd, sparse input
                J3_log = softmax_jacobian_autograd(x, dim, log=True)

                J = J.transpose(0, dim + 1)
                J2_log = J2_log.transpose(0, dim + 1)
                J3_log = J3_log.transpose(0, dim + 1)
                self.assertEqual(J, J2_log * r1)
                self.assertEqual(J, J3_log * r1)

                if dim == 0:
                    # check dtype argument
                    other_dtype = torch.float32
                    y2 = torch.sparse.softmax(x, dim, dtype=other_dtype)
                    self.assertEqual(y2.dtype, other_dtype)
                    self.assertEqual(y2, y1.type(other_dtype))

                    ly2 = torch.sparse.log_softmax(x, dim, dtype=other_dtype)
                    self.assertEqual(ly2.dtype, other_dtype)
                    self.assertEqual(ly2, ly1.type(other_dtype))

        test_op(1, 10, [3])
        test_op(1, 10, [2, 3])
        test_op(1, 10, [3, 2])
        test_op(2, 10, [2, 3, 4])
        test_op(2, 10, [3, 4])
        test_op(2, 5, [5, 4])
        test_op(2, 10, [3, 4, 2])
        test_op(3, 10, [3, 4, 2])
        test_op(3, 100, [3, 4, 2])
        test_op(3, 100, [3, 4, 2, 3])
        test_op(3, 100, [3, 4, 2, 3, 5, 2])
        test_op(4, 100, [3, 4, 2, 3, 5, 2])

    def test_sparse_matmul(self):
        """
        This function test `torch.sparse.mm` when both the mat1 and mat2 are sparse tensors. 
        """

        def _indices2csr(indices, dim):
            nnz = len(indices)
            r = [0] * (dim + 1)
            last_i = 0
            for i in indices:
                if i != last_i:
                    for _i in range(last_i, i + 1):
                        r[_i + 1] = r[last_i + 1]
                    last_i = i
                r[last_i + 1] += 1
            for _i in range(last_i, dim):
                r[_i + 1] = r[last_i + 1]
            assert r[-1] == nnz
            return r

        def sparse_mm(a, b, method='scipy'):
            a = a.to('cpu')
            b = b.to('cpu')
            if method == 'scipy':
                indices_1 = a._indices().numpy()
                values_1 = a._values().numpy()
                indices_2 = b._indices().numpy()
                values_2 = b._values().numpy()

                mat1 = scipy.sparse.coo_matrix((values_1, (indices_1[0], indices_1[1])), shape=a.shape)
                mat2 = scipy.sparse.coo_matrix((values_2, (indices_2[0], indices_2[1])), shape=b.shape)
                result = mat1.dot(mat2).tocoo()
                return torch.sparse_coo_tensor([result.row, result.col], result.data, result.shape)
            else:
                assert a.shape[1] == b.shape[0]
                n, p = a.shape
                p, m = b.shape
                indices_a = a._indices()
                values_a = a._values()
                indices_b = b._indices()
                values_b = b._values()
                nnz1 = len(indices_a[0])
                nnz2 = len(indices_b[0])

                if a.is_coalesced() and b.is_coalesced():
                    r2 = _indices2csr(indices_b[0], b.shape[0])
                    d = defaultdict(values_b.numpy().dtype.type)
                    for n1 in range(nnz1):
                        for n2 in range(r2[indices_a[1][n1]], r2[indices_a[1][n1] + 1]):
                            d[indices_a[0][n1].item(), indices_b[1][n2].item()] += values_a[n1] * values_b[n2]

                else:
                    d = defaultdict(values_b.numpy().dtype.type)
                    for n1 in range(nnz1):
                        for n2 in range(nnz2):
                            if indices_b[0][n2] == indices_a[1][n1]:
                                d[indices_a[0][n1].item(), indices_b[1][n2].item()] += values_a[n1] * values_b[n2]
                i3 = []
                j3 = []
                values = []
                for i, j in sorted(d):
                    i3.append(i)
                    j3.append(j)
                    values.append(d[i, j])
                return torch.sparse_coo_tensor(torch.tensor([i3, j3]), torch.tensor(values), (n, m))

        def grad_with_custom_sparsity_pattern_test_helper(sparse_dims, nnz, shape_a, shape_b):
            def test_grad_dense(a_s, b_s, g_s):
                a = a_s.to_dense().detach()
                b = b_s.to_dense().detach()
                g = g_s.to_dense().detach()

                a.requires_grad_(True)
                b.requires_grad_(True)
                c = a @ b
                c.backward(g)
                return a.grad.sparse_mask(a_s.coalesce()), b.grad.sparse_mask(b_s.coalesce())

            a, _, _ = self._gen_sparse(sparse_dims, nnz, shape_a)
            b, _, _ = self._gen_sparse(sparse_dims, nnz, shape_b)
            a.requires_grad_(True)
            b.requires_grad_(True)

            c = torch.sparse.mm(a, b)
            c2 = c.to_dense().detach()
            c2 = torch.rand_like(c2)
            g = c2.sparse_mask(c.coalesce())

            c.backward(g)

            a_grad, b_grad = test_grad_dense(a, b, g)
            self.assertEqual(a.grad, a_grad)
            self.assertEqual(b.grad, b_grad)

        def test_sparse_matmul(sparse_dims, nnz, shape_a, shape_b):
            a, i_a, v_a = self._gen_sparse(sparse_dims, nnz, shape_a)
            b, i_b, v_b = self._gen_sparse(sparse_dims, nnz, shape_b)

            # python implementation
            r1 = sparse_mm(a, b, 'scipy' if TEST_SCIPY else 'direct')

            self.assertEqual(r1.to_dense(), torch.mm(a.to_dense(), b.to_dense()))

            # cpp implementation
            r2 = torch.sparse.mm(a, b)
            self.assertEqual(r1, r2)

            a.requires_grad_(True)
            b.requires_grad_(True)

            # check autograd support on sparse matmul
            def fn(D1, D2):
                return torch.sparse.mm(D1, D2).to_dense()

            # For cuda, `nondet_tol` is set with `1e-5` 
            # This is because cuSparse sometimes returns approximate zero values like `~e-323`
            # TODO: Check this cuSparse issue. 
            # This happens when you do chain multiplication `torch.sparse.mm` operations 
            gradcheck(fn, (a, b), check_sparse_nnz=True, nondet_tol=1e-5)
            grad_with_custom_sparsity_pattern_test_helper(sparse_dims, nnz, shape_a, shape_b)

        def test_error_cases():
            def fn(sparse_dims, nnz, shape_a, shape_b):
                a, i_a, v_a = self._gen_sparse(sparse_dims, nnz, shape_a)
                b, i_b, v_b = self._gen_sparse(sparse_dims, nnz, shape_b)
                r2 = torch.sparse.mm(a, b)

            # This is not a matrix
            self.assertRaises(RuntimeError, lambda: fn(3, 4, [2, 2, 2], [2, 2, 2]))

            # Shapes does not 
            self.assertRaisesRegex(RuntimeError, 
                                   r"mat1 and mat2 shapes cannot be multiplied \(2x3 and 4x2\)",
                                   lambda: fn(2, 10, [2, 3], [4, 2]))

            def different_dtypes():
                a, i_a, v_a = self._gen_sparse(2, 10, [2, 2])
                b, i_b, v_b = self._gen_sparse(2, 10, [2, 2])
                r2 = torch.sparse.mm(a.to(torch.float64), a.to(torch.float32))

            self.assertRaisesRegex(RuntimeError, 'mat1 dtype Double does not match mat2 dtype Float', different_dtypes)

        for n in range(2, 5):
            for m in range(2, 8):
                for p in range(2, 8):
                    test_sparse_matmul(2, 10, [n, m], [m, p])

        test_sparse_matmul(2, 0, [0, 0], [0, 0])
        test_sparse_matmul(2, 0, [0, 10], [10, 0])
        test_error_cases()

    def test_assign(self):
        def assign_to(a):
            a, i_a, v_a = self._gen_sparse(2, 5, [2, 3])
            a[0] = 100

        self.assertRaises(TypeError, assign_to)


class TestUncoalescedSparse(TestSparse):
    def setUp(self):
        super(TestUncoalescedSparse, self).setUp()
        self.is_uncoalesced = True


@unittest.skipIf(not TEST_CUDA, 'CUDA not available')
class TestCudaSparse(TestSparse):
    def setUp(self):
        super(TestCudaSparse, self).setUp()
        self.is_cuda = True
        self.device = 'cuda'
        self.legacy_sparse_tensor = torch.cuda.sparse.DoubleTensor


@unittest.skipIf(not TEST_CUDA, 'CUDA not available')
class TestCudaUncoalescedSparse(TestCudaSparse):
    def setUp(self):
        super(TestCudaUncoalescedSparse, self).setUp()
        self.is_uncoalesced = True


class TestSparseOneOff(TestCase):
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_from_cpu(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "backend of indices \\(CUDA\\) must match backend of values \\(CPU\\)"):
            torch.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                     torch.randn(4, 4, 4),
                                     [3, 4, 4])

        with self.assertRaisesRegex(
                RuntimeError,
                "backend of indices \\(CUDA\\) must match backend of values \\(CPU\\)"):
            torch.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                     torch.randn(4, 4, 4, 0),
                                     [3, 4, 4, 0])

        with self.assertRaisesRegex(
                RuntimeError,
                "backend of indices \\(CUDA\\) must match backend of values \\(CPU\\)"):
            torch.sparse.FloatTensor(torch.LongTensor(1, 0).cuda(),
                                     torch.randn(0, 4, 4, 0),
                                     [0, 4, 4, 0])

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_sparse_cpu_dense_add(self):
        x = torch.zeros(3, 4, 4)
        sparse_y = torch.cuda.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                                 torch.randn(4, 4, 4).cuda(),
                                                 [3, 4, 4])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

        x = torch.zeros(3, 4, 4, 0)
        sparse_y = torch.cuda.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                                 torch.randn(4, 4, 4, 0).cuda(),
                                                 [3, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

        x = torch.zeros(0, 4, 4, 0)
        sparse_y = torch.cuda.sparse.FloatTensor(torch.LongTensor(1, 0).cuda(),
                                                 torch.randn(0, 4, 4, 0).cuda(),
                                                 [0, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

class TestSparseUnaryUfuncs(TestCase):
    exact_dtype = True

    @ops(sparse_unary_ufuncs)
    def test_sparse_consistency(self, device, dtype, op):
        unsupportedTypes = [torch.bfloat16, torch.cfloat, torch.cdouble]
        if dtype in unsupportedTypes:
            self.skipTest('Skipped! Unsupported dtypes for Sparse')

        samples = op.sample_inputs(device, dtype)

        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        sample = samples[0]

        if len(sample.input) > 1:
            self.skipTest("Skipped! Testing unary ops, one input is expected")
        sample = sample.input[0]

        expected = op(sample)
        assert torch.is_tensor(expected)
        output = op(sample.to_sparse())
        assert torch.is_tensor(output)
        self.assertEqual(output.to_dense(), expected)

instantiate_device_type_tests(TestSparseUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
