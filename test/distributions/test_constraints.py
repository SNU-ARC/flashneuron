import pytest

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.testing._internal.common_cuda import TEST_CUDA


CONSTRAINTS = [
    (constraints.real,),
    (constraints.real_vector,),
    (constraints.positive,),
    (constraints.greater_than, [-10., -2, 0, 2, 10]),
    (constraints.greater_than, 0),
    (constraints.greater_than, 2),
    (constraints.greater_than, -2),
    (constraints.greater_than_eq, 0),
    (constraints.greater_than_eq, 2),
    (constraints.greater_than_eq, -2),
    (constraints.less_than, [-10., -2, 0, 2, 10]),
    (constraints.less_than, 0),
    (constraints.less_than, 2),
    (constraints.less_than, -2),
    (constraints.unit_interval,),
    (constraints.interval, [-4., -2, 0, 2, 4], [-3., 3, 1, 5, 5]),
    (constraints.interval, -2, -1),
    (constraints.interval, 1, 2),
    (constraints.half_open_interval, [-4., -2, 0, 2, 4], [-3., 3, 1, 5, 5]),
    (constraints.half_open_interval, -2, -1),
    (constraints.half_open_interval, 1, 2),
    (constraints.simplex,),
    (constraints.corr_cholesky,),
    (constraints.lower_cholesky,),
]


def build_constraint(constraint_fn, args, is_cuda=False):
    if not args:
        return constraint_fn
    t = torch.cuda.DoubleTensor if is_cuda else torch.DoubleTensor
    return constraint_fn(*(t(x) if isinstance(x, list) else x for x in args))


@pytest.mark.parametrize('constraint_fn, args', [(c[0], c[1:]) for c in CONSTRAINTS])
@pytest.mark.parametrize('is_cuda', [False,
                                     pytest.param(True, marks=pytest.mark.skipif(not TEST_CUDA,
                                                                                 reason='CUDA not found.'))])
def test_biject_to(constraint_fn, args, is_cuda):
    constraint = build_constraint(constraint_fn, args, is_cuda=is_cuda)
    try:
        t = biject_to(constraint)
    except NotImplementedError:
        pytest.skip('`biject_to` not implemented.')
    assert t.bijective, "biject_to({}) is not bijective".format(constraint)
    if constraint_fn is constraints.corr_cholesky:
        # (D * (D-1)) / 2 (where D = 4) = 6 (size of last dim)
        x = torch.randn(6, 6, dtype=torch.double)
    else:
        x = torch.randn(5, 5, dtype=torch.double)
    if is_cuda:
        x = x.cuda()
    y = t(x)
    assert constraint.check(y).all(), '\n'.join([
        "Failed to biject_to({})".format(constraint),
        "x = {}".format(x),
        "biject_to(...)(x) = {}".format(y),
    ])
    x2 = t.inv(y)
    assert torch.allclose(x, x2), "Error in biject_to({}) inverse".format(constraint)

    j = t.log_abs_det_jacobian(x, y)
    assert j.shape == x.shape[:x.dim() - t.domain.event_dim]


@pytest.mark.parametrize('constraint_fn, args', [(c[0], c[1:]) for c in CONSTRAINTS])
@pytest.mark.parametrize('is_cuda', [False,
                                     pytest.param(True, marks=pytest.mark.skipif(not TEST_CUDA,
                                                                                 reason='CUDA not found.'))])
def test_transform_to(constraint_fn, args, is_cuda):
    constraint = build_constraint(constraint_fn, args, is_cuda=is_cuda)
    t = transform_to(constraint)
    if constraint_fn is constraints.corr_cholesky:
        # (D * (D-1)) / 2 (where D = 4) = 6 (size of last dim)
        x = torch.randn(6, 6, dtype=torch.double)
    else:
        x = torch.randn(5, 5, dtype=torch.double)
    if is_cuda:
        x = x.cuda()
    y = t(x)
    assert constraint.check(y).all(), "Failed to transform_to({})".format(constraint)
    x2 = t.inv(y)
    y2 = t(x2)
    assert torch.allclose(y, y2), "Error in transform_to({}) pseudoinverse".format(constraint)


if __name__ == '__main__':
    pytest.main([__file__])
