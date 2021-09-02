import torch
import numpy as np

import random
from torch._six import nan
from itertools import product

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, make_tensor)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyOnCPUAndCUDA,
     skipCUDAIfRocm, onlyCUDA, dtypesIfCUDA)

# TODO: remove this
SIZE = 100

class TestSortAndSelect(TestCase):

    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = 4
        if order == 'descending':
            def check_order(a, b):
                # `a != a` because we put NaNs
                # at the end of ascending sorted lists,
                # and the beginning of descending ones.
                return a != a or a >= b
        elif order == 'ascending':
            def check_order(a, b):
                # see above
                return b != b or a <= b
        else:
            error('unknown order "{}", must be "ascending" or "descending"'.format(order))

        are_ordered = True
        for j, k in product(range(SIZE), range(1, SIZE)):
            self.assertTrue(check_order(mxx[j][k - 1], mxx[j][k]),
                            'torch.sort ({}) values unordered for {}'.format(order, task))

        seen = set()
        indicesCorrect = True
        size = x.size(x.dim() - 1)
        for k in range(size):
            seen.clear()
            for j in range(size):
                self.assertEqual(x[k][ixx[k][j]], mxx[k][j],
                                 msg='torch.sort ({}) indices wrong for {}'.format(order, task))
                seen.add(ixx[k][j])
            self.assertEqual(len(seen), size)

    def test_sort(self, device):
        SIZE = 4
        x = torch.rand(SIZE, SIZE, device=device)
        res1val, res1ind = torch.sort(x)

        # Test use of result tensor
        res2val = torch.tensor((), device=device)
        res2ind = torch.tensor((), device=device, dtype=torch.long)
        torch.sort(x, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
        self.assertEqual(torch.argsort(x), res1ind)
        self.assertEqual(x.argsort(), res1ind)

        # Test sorting of random numbers
        self.assertIsOrdered('ascending', x, res2val, res2ind, 'random')

        # Test simple sort
        self.assertEqual(
            torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
            torch.tensor((10, 20, 30, 40, 50), device=device),
            atol=0, rtol=0
        )

        # Test that we still have proper sorting with duplicate keys
        x = torch.floor(torch.rand(SIZE, SIZE, device=device) * 10)
        torch.sort(x, out=(res2val, res2ind))
        self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with duplicate keys')

        # DESCENDING SORT
        x = torch.rand(SIZE, SIZE, device=device)
        res1val, res1ind = torch.sort(x, x.dim() - 1, True)

        # Test use of result tensor
        res2val = torch.tensor((), device=device)
        res2ind = torch.tensor((), device=device, dtype=torch.long)
        torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
        self.assertEqual(torch.argsort(x, x.dim() - 1, True), res1ind)
        self.assertEqual(x.argsort(x.dim() - 1, True), res1ind)

        # Test sorting of random numbers
        self.assertIsOrdered('descending', x, res2val, res2ind, 'random')

        # Test simple sort task
        self.assertEqual(
            torch.sort(torch.tensor((10, 20, 30, 40, 50), device=device), 0, True)[0],
            torch.tensor((50, 40, 30, 20, 10), device=device),
            atol=0, rtol=0
        )

        # Test that we still have proper sorting with duplicate keys
        self.assertIsOrdered('descending', x, res2val, res2ind, 'random with duplicate keys')

        # Test sorting with NaNs
        x = torch.rand(SIZE, SIZE, device=device)
        x[1][2] = float('NaN')
        x[3][0] = float('NaN')
        torch.sort(x, out=(res2val, res2ind))
        self.assertIsOrdered('ascending', x, res2val, res2ind,
                             'random with NaNs')
        torch.sort(x, out=(res2val, res2ind), descending=True)
        self.assertIsOrdered('descending', x, res2val, res2ind,
                             'random with NaNs')

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False)))
    def test_msort(self, device, dtype):
        def test(shape):
            tensor = make_tensor(shape, device, dtype, low=-9, high=9)
            if tensor.size() != torch.Size([]):
                expected = torch.from_numpy(np.msort(tensor.cpu().numpy()))
            else:
                expected = tensor  # numpy.msort() does not support empty shapes tensor

            result = torch.msort(tensor)
            self.assertEqual(result, expected)

            out = torch.empty_like(result)
            torch.msort(tensor, out=out)
            self.assertEqual(out, expected)

        shapes = (
            [],
            [0, ],
            [20, ],
            [1, 20],
            [30, 30],
            [10, 20, 30]
        )
        for shape in shapes:
            test(shape)

    def test_topk(self, device):
        def topKViaSort(t, k, dim, dir):
            sorted, indices = t.sort(dim, dir)
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # Values should be exactly equivalent
            self.assertEqual(res1, res2, atol=0, rtol=0)

            # Indices might differ based on the implementation, since there is
            # no guarantee of the relative order of selection
            if not ind1.eq(ind2).all():
                # To verify that the indices represent equivalent elements,
                # gather from the input using the topk indices and compare against
                # the sort indices
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, atol=0, rtol=0)

        def compare(t, k, dim, dir):
            topKVal, topKInd = t.topk(k, dim, dir, True)
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        t = torch.rand(random.randint(1, SIZE),
                       random.randint(1, SIZE),
                       random.randint(1, SIZE), device=device)

        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

    def test_topk_arguments(self, device):
        q = torch.randn(10, 2, 10, device=device)
        # Make sure True isn't mistakenly taken as the 2nd dimension (interpreted as 1)
        self.assertRaises(TypeError, lambda: q.topk(4, True))

    @skipCUDAIfRocm
    def test_unique_dim(self, device):
        self.assertFalse(hasattr(torch, 'unique_dim'))

        def run_test(device, dtype):
            x = torch.tensor([[[1., 1.],
                               [0., 1.],
                               [2., 1.],
                               [0., 1.]],
                              [[1., 1.],
                               [0., 1.],
                               [2., 1.],
                               [0., 1.]]],
                             dtype=dtype,
                             device=device)
            x_empty = torch.empty(5, 0, dtype=dtype, device=device)
            x_ill_formed_empty = torch.empty(5, 0, 0, dtype=dtype, device=device)
            x_ill_formed_empty_another = torch.empty(5, 0, 5, dtype=dtype, device=device)
            expected_unique_dim0 = torch.tensor([[[1., 1.],
                                                  [0., 1.],
                                                  [2., 1.],
                                                  [0., 1.]]],
                                                dtype=dtype,
                                                device=device)
            expected_inverse_dim0 = torch.tensor([0, 0])
            expected_counts_dim0 = torch.tensor([2])
            expected_unique_dim1 = torch.tensor([[[0., 1.],
                                                  [1., 1.],
                                                  [2., 1.]],
                                                 [[0., 1.],
                                                  [1., 1.],
                                                  [2., 1.]]],
                                                dtype=dtype,
                                                device=device)
            expected_unique_dim1_bool = torch.tensor([[[False, True], [True, True]],
                                                      [[False, True], [True, True]]],
                                                     dtype=torch.bool,
                                                     device=device)
            expected_inverse_dim1 = torch.tensor([1, 0, 2, 0])
            expected_inverse_dim1_bool = torch.tensor([1, 0, 1, 0])
            expected_counts_dim1 = torch.tensor([2, 1, 1])
            expected_counts_dim1_bool = torch.tensor([2, 2])
            expected_unique_dim2 = torch.tensor([[[1., 1.],
                                                  [0., 1.],
                                                  [2., 1.],
                                                  [0., 1.]],
                                                 [[1., 1.],
                                                  [0., 1.],
                                                  [2., 1.],
                                                  [0., 1.]]],
                                                dtype=dtype,
                                                device=device)
            expected_inverse_dim2 = torch.tensor([0, 1])
            expected_counts_dim2 = torch.tensor([1, 1])
            expected_unique_empty = torch.tensor([], dtype=dtype, device=device)
            expected_inverse_empty = torch.tensor([], dtype=torch.long, device=device)
            expected_counts_empty = torch.tensor([], dtype=torch.long, device=device)
            # dim0
            x_unique = torch.unique(x, dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)

            x_unique, x_inverse = torch.unique(
                x,
                return_inverse=True,
                dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            self.assertEqual(expected_inverse_dim0, x_inverse)

            x_unique, x_counts = torch.unique(
                x,
                return_inverse=False,
                return_counts=True,
                dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            self.assertEqual(expected_counts_dim0, x_counts)

            x_unique, x_inverse, x_counts = torch.unique(
                x,
                return_inverse=True,
                return_counts=True,
                dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            self.assertEqual(expected_inverse_dim0, x_inverse)
            self.assertEqual(expected_counts_dim0, x_counts)

            # dim1
            x_unique = torch.unique(x, dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)

            x_unique, x_inverse = torch.unique(
                x,
                return_inverse=True,
                dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
                self.assertEqual(expected_inverse_dim1_bool, x_inverse)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
                self.assertEqual(expected_inverse_dim1, x_inverse)

            x_unique, x_counts = torch.unique(
                x,
                return_inverse=False,
                return_counts=True,
                dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
                self.assertEqual(expected_counts_dim1_bool, x_counts)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
                self.assertEqual(expected_counts_dim1, x_counts)

            x_unique, x_inverse, x_counts = torch.unique(
                x,
                return_inverse=True,
                return_counts=True,
                dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
                self.assertEqual(expected_inverse_dim1_bool, x_inverse)
                self.assertEqual(expected_counts_dim1_bool, x_counts)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
                self.assertEqual(expected_inverse_dim1, x_inverse)
                self.assertEqual(expected_counts_dim1, x_counts)

            # dim2
            x_unique = torch.unique(x, dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)

            x_unique, x_inverse = torch.unique(
                x,
                return_inverse=True,
                dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            self.assertEqual(expected_inverse_dim2, x_inverse)

            x_unique, x_counts = torch.unique(
                x,
                return_inverse=False,
                return_counts=True,
                dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            self.assertEqual(expected_counts_dim2, x_counts)

            x_unique, x_inverse, x_counts = torch.unique(
                x,
                return_inverse=True,
                return_counts=True,
                dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            self.assertEqual(expected_inverse_dim2, x_inverse)
            self.assertEqual(expected_counts_dim2, x_counts)

            # test empty tensor
            x_unique, x_inverse, x_counts = torch.unique(
                x_empty,
                return_inverse=True,
                return_counts=True,
                dim=1)
            self.assertEqual(expected_unique_empty, x_unique)
            self.assertEqual(expected_inverse_empty, x_inverse)
            self.assertEqual(expected_counts_empty, x_counts)

            # test not a well formed tensor
            # Checking for runtime error, as this is the expected behaviour
            with self.assertRaises(RuntimeError):
                torch.unique(
                    x_ill_formed_empty,
                    return_inverse=True,
                    return_counts=True,
                    dim=1)

            # test along dim2
            with self.assertRaises(RuntimeError):
                torch.unique(
                    x_ill_formed_empty_another,
                    return_inverse=True,
                    return_counts=True,
                    dim=2)

            # test consecutive version
            y = torch.tensor(
                [[0, 1],
                 [0, 1],
                 [0, 1],
                 [1, 2],
                 [1, 2],
                 [3, 4],
                 [0, 1],
                 [0, 1],
                 [3, 4],
                 [1, 2]],
                dtype=dtype,
                device=device
            )
            expected_y_unique = torch.tensor(
                [[0, 1],
                 [1, 2],
                 [3, 4],
                 [0, 1],
                 [3, 4],
                 [1, 2]],
                dtype=dtype,
                device=device
            )
            expected_y_inverse = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 4, 5], dtype=torch.int64, device=device)
            expected_y_counts = torch.tensor([3, 2, 1, 2, 1, 1], dtype=torch.int64, device=device)
            expected_y_inverse_bool = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=device)
            expected_y_counts_bool = torch.tensor([3, 3, 2, 2], dtype=torch.int64, device=device)
            y_unique, y_inverse, y_counts = torch.unique_consecutive(y, return_inverse=True, return_counts=True, dim=0)
            if x.dtype == torch.bool:
                self.assertEqual(expected_y_inverse_bool, y_inverse)
                self.assertEqual(expected_y_counts_bool, y_counts)
            else:
                self.assertEqual(expected_y_inverse, y_inverse)
                self.assertEqual(expected_y_counts, y_counts)

        run_test(device, torch.float)
        run_test(device, torch.double)
        run_test(device, torch.long)
        run_test(device, torch.uint8)
        run_test(device, torch.bool)

    @onlyCUDA
    def test_topk_noncontiguous_gpu(self, device):
        t = torch.randn(20, device=device)[::2]
        top1, idx1 = t.topk(5)
        top2, idx2 = t.contiguous().topk(5)
        self.assertEqual(top1, top2)
        self.assertEqual(idx1, idx2)

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_topk_integral(self, device, dtype):
        a = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=(10,),
                          dtype=dtype, device=device)
        sort_topk = a.sort()[0][-5:].flip(0)
        topk = a.topk(5)
        self.assertEqual(sort_topk, topk[0])      # check values
        self.assertEqual(sort_topk, a[topk[1]])   # check indices

    @dtypesIfCUDA(*torch.testing.get_all_fp_dtypes())
    @dtypes(torch.float, torch.double)
    def test_topk_nonfinite(self, device, dtype):
        x = torch.tensor([float('nan'), float('inf'), 1e4, 0, -1e4, -float('inf')], device=device, dtype=dtype)
        val, idx = x.topk(4)
        expect = torch.tensor([float('nan'), float('inf'), 1e4, 0], device=device, dtype=dtype)
        self.assertEqual(val, expect)
        self.assertEqual(idx, [0, 1, 2, 3])

        val, idx = x.topk(4, largest=False)
        expect = torch.tensor([-float('inf'), -1e4, 0, 1e4], device=device, dtype=dtype)
        self.assertEqual(val, expect)
        self.assertEqual(idx, [5, 4, 3, 2])

    def test_topk_4d(self, device):
        x = torch.ones(2, 3072, 2, 2, device=device)
        x[:, 1, :, :] *= 2.
        x[:, 10, :, :] *= 1.5
        val, ind = torch.topk(x, k=2, dim=1)
        expected_ind = torch.ones(2, 2, 2, 2, dtype=torch.long, device=device)
        expected_ind[:, 1, :, :] = 10
        expected_val = torch.ones(2, 2, 2, 2, device=device)
        expected_val[:, 0, :, :] *= 2.
        expected_val[:, 1, :, :] *= 1.5
        self.assertEqual(val, expected_val, atol=0, rtol=0)
        self.assertEqual(ind, expected_ind, atol=0, rtol=0)

    def _test_unique_scalar_empty(self, dtype, device, f):
        # test scalar
        x = torch.tensor(0, dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([0], dtype=dtype, device=device)
        expected_inverse = torch.tensor(0, device=device)
        expected_counts = torch.tensor([1], device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

        # test zero sized tensor
        x = torch.zeros((0, 0, 3), dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([], dtype=dtype, device=device)
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device=device)
        expected_counts = torch.tensor([], dtype=torch.long, device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

    def _test_unique_with_expects(self, device, dtype, f, x, expected_unique, expected_inverse, expected_counts, additional_shape):
        def ensure_tuple(x):
            if isinstance(x, torch.Tensor):
                return (x,)
            return x

        for return_inverse in [True, False]:
            for return_counts in [True, False]:
                # test with expected
                ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                self.assertEqual(expected_unique, ret[0])
                if return_inverse:
                    self.assertEqual(expected_inverse, ret[1])
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    self.assertEqual(expected_counts, ret[count_index])

                # tests per-element unique on a higher rank tensor.
                y = x.view(additional_shape)
                y_unique, y_inverse, y_counts = f(y, return_inverse=True, return_counts=True)
                self.assertEqual(expected_unique, y_unique)
                self.assertEqual(expected_inverse.view(additional_shape), y_inverse)
                self.assertEqual(expected_counts, y_counts)

    @dtypes(*set(torch.testing.get_all_dtypes()) - {torch.bfloat16, torch.complex64, torch.complex128})
    def test_unique(self, device, dtype):
        if dtype is torch.half and self.device_type == 'cpu':
            return  # CPU does not have half support

        def ensure_tuple(x):
            if isinstance(x, torch.Tensor):
                return (x,)
            return x

        if dtype is torch.bool:
            x = torch.tensor([True, False, False, False, True, False, True, False], dtype=torch.bool, device=device)
            expected_unique = torch.tensor([False, True], dtype=torch.bool, device=device)
            expected_inverse = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.long, device=device)
            expected_counts = torch.tensor([5, 3], dtype=torch.long, device=device)
        else:
            x = torch.tensor([1, 2, 3, 2, 8, 5, 2, 3], dtype=dtype, device=device)
            expected_unique = torch.tensor([1, 2, 3, 5, 8], dtype=dtype, device=device)
            expected_inverse = torch.tensor([0, 1, 2, 1, 4, 3, 1, 2], device=device)
            expected_counts = torch.tensor([1, 3, 2, 1, 1], device=device)

        # test sorted unique
        fs = [
            lambda x, **kwargs: torch.unique(x, sorted=True, **kwargs),
            lambda x, **kwargs: x.unique(sorted=True, **kwargs),
        ]
        for f in fs:
            self._test_unique_with_expects(device, dtype, f, x, expected_unique, expected_inverse, expected_counts, (2, 2, 2))
            self._test_unique_scalar_empty(dtype, device, f)

        # test unsorted unique
        fs = [
            lambda x, **kwargs: torch.unique(x, sorted=False, **kwargs),
            lambda x, **kwargs: x.unique(sorted=False, **kwargs)
        ]
        for f in fs:
            self._test_unique_scalar_empty(dtype, device, f)
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                    self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                    x_list = x.tolist()
                    x_unique_list = ret[0].tolist()
                    self.assertEqual(expected_unique.tolist(), sorted(x_unique_list))
                    if return_inverse:
                        x_inverse_list = ret[1].tolist()
                        for i, j in enumerate(x_inverse_list):
                            self.assertEqual(x_list[i], x_unique_list[j])
                    if return_counts:
                        count_index = 1 + int(return_inverse)
                        x_counts_list = ret[count_index].tolist()
                        for i, j in zip(x_unique_list, x_counts_list):
                            count = 0
                            for k in x_list:
                                if k == i:
                                    count += 1
                            self.assertEqual(j, count)

    @dtypes(*set(torch.testing.get_all_dtypes()) - {torch.bfloat16, torch.complex64, torch.complex128})
    def test_unique_consecutive(self, device, dtype):
        if dtype is torch.half and self.device_type == 'cpu':
            return  # CPU does not have half support

        if dtype is torch.bool:
            x = torch.tensor([True, False, False, False, True, True, False, False, False], dtype=torch.bool, device=device)
            expected_unique = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
            expected_inverse = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=torch.long, device=device)
            expected_counts = torch.tensor([1, 3, 2, 3], dtype=torch.long, device=device)
        else:
            x = torch.tensor([1, 2, 2, 2, 5, 5, 2, 2, 3], dtype=dtype, device=device)
            expected_unique = torch.tensor([1, 2, 5, 2, 3], dtype=dtype, device=device)
            expected_inverse = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4], device=device)
            expected_counts = torch.tensor([1, 3, 2, 2, 1], device=device)

        for f in [torch.unique_consecutive, lambda x, **kwargs: x.unique_consecutive(**kwargs)]:
            self._test_unique_with_expects(device, dtype, f, x, expected_unique, expected_inverse, expected_counts, (3, 3))
            self._test_unique_scalar_empty(dtype, device, f)

    @dtypes(torch.double)
    def test_kthvalue(self, device, dtype):
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE, dtype=dtype, device=device)
        x0 = x.clone()

        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k, keepdim=False)
        res2val, res2ind = torch.sort(x)

        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)
        # test use of result tensors
        k = random.randint(1, SIZE)
        res1val = torch.tensor([], dtype=dtype, device=device)
        res1ind = torch.tensor([], dtype=torch.long, device=device)
        torch.kthvalue(x, k, keepdim=False, out=(res1val, res1ind))
        res2val, res2ind = torch.sort(x)
        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)

        # test non-default dim
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k, 0, keepdim=False)
        res2val, res2ind = torch.sort(x, 0)
        self.assertEqual(res1val, res2val[k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind[k - 1], atol=0, rtol=0)

        # non-contiguous
        y = x.narrow(1, 0, 1)
        y0 = y.contiguous()
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(y, k)
        res2val, res2ind = torch.kthvalue(y0, k)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # non-contiguous [Reference: https://github.com/pytorch/pytorch/issues/45721]
        non_contig_t = torch.tensor([0, -1, 1, -2, 2], dtype=dtype, device=device)[::2]
        expected_val, expected_ind = non_contig_t.contiguous().kthvalue(2)
        non_contig_cpu_t = non_contig_t.cpu()
        expected_val_cpu, expected_ind_cpu = non_contig_cpu_t.kthvalue(2)

        out_val, out_ind = non_contig_t.kthvalue(2)
        self.assertEqual(expected_val, out_val, atol=0, rtol=0)
        self.assertEqual(expected_ind, out_ind, atol=0, rtol=0)
        self.assertEqual(expected_val_cpu, out_val, atol=0, rtol=0)
        self.assertEqual(expected_ind_cpu, out_ind, atol=0, rtol=0)

        # check that the input wasn't modified
        self.assertEqual(x, x0, atol=0, rtol=0)

        # simple test case (with repetitions)
        y = torch.tensor((3., 5, 4, 1, 1, 5), dtype=dtype, device=device)
        self.assertEqual(torch.kthvalue(y, 3)[0], 3, atol=0, rtol=0)
        self.assertEqual(torch.kthvalue(y, 2)[0], 1, atol=0, rtol=0)

        # simple test case (with NaN)
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE, dtype=dtype, device=device)
        x[torch.arange(SIZE), :, torch.randint(50, (50,))] = nan
        ks = [random.randint(1, SIZE), 1, SIZE, SIZE - 1]
        res2val, res2ind = torch.sort(x)
        for k in ks:
            res1val, res1ind = torch.kthvalue(x, k, keepdim=False)
            self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
            self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)

    # test overlapping output
    @dtypes(torch.double)
    @onlyOnCPUAndCUDA   # Fails on XLA
    def test_kthvalue_overlap(self, device, dtype):
        S = 10
        k = 5
        a = torch.randn(S)
        indices = torch.empty((), device=device, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "unsupported operation:"):
            torch.kthvalue(a, k, out=(a, indices))

    @dtypes(torch.float)
    @onlyOnCPUAndCUDA   # Fails on XLA
    def test_kthvalue_scalar(self, device, dtype):
        # Test scalar input (test case from https://github.com/pytorch/pytorch/issues/30818)
        # Tests that passing a scalar tensor or 1D tensor with 1 element work either way
        res = torch.tensor(2, device=device, dtype=dtype).kthvalue(1)
        ref = torch.tensor([2], device=device, dtype=dtype).kthvalue(1)
        self.assertEqual(res[0], ref[0].squeeze())
        self.assertEqual(res[1], ref[1].squeeze())

instantiate_device_type_tests(TestSortAndSelect, globals())

if __name__ == '__main__':
    run_tests()
