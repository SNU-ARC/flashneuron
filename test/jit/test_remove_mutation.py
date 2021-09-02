import os
import sys

import torch
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestRemoveMutation(JitTestCase):
    def test_aten_inplace(self):
        def test_not_new_alias(x):
            y = x[0]
            y.add_(2)
            return y

        fn = torch.jit.script(test_not_new_alias)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(torch.ones([2, 2])), test_not_new_alias(torch.ones([2, 2])))

        def test_no_lowering():
            x = torch.tensor([2, 2])
            x[0] = 3
            return x

        # there is no functional equivalent of x[0] = ...
        fn = torch.jit.script(test_no_lowering)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::copy_").run(graph)
        self.assertEqual(fn(), test_no_lowering())

        def test_move_before_not_valid():
            y = torch.tensor([2, 2])
            z = y + 2
            y.add_(2)
            return y, z

        fn = torch.jit.script(test_move_before_not_valid)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(), test_move_before_not_valid())

        def test_successful():
            x = torch.tensor([2, 2])
            x.add_(1)
            x.add_(3)
            y = x + 4
            return x, y

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not("aten::add_").run(graph)
        self.assertEqual(test_successful(), fn())

        def test_intermediary_use():
            x = torch.tensor([2, 2])
            x.add_(1)
            y = x + 4
            x.add_(3)
            return x, y

        fn = torch.jit.script(test_intermediary_use)
        graph = fn.graph
        FileCheck().check_count("aten::add_", 2).run(graph)
        self.run_pass('remove_mutation', graph)
        # Unable to remove the second add_ because of the y = x + 4 use
        # In the future we could duplicating the value of x as a temporary and replacing
        # its intermediary use (so long as aliasing is safe)
        FileCheck().check_count("aten::add_", 1).run(graph)
        self.assertEqual(test_intermediary_use(), fn())

    def test_if_output(self):
        def foo(x, cond: bool):
            if cond:
                y = x + 5
            else:
                y = x + 2
            y.add_(4)
            return y

        out_eager = foo(torch.tensor(5), True)
        foo_script = torch.jit.script(foo)
        FileCheck().check("aten::add_").run(foo_script.graph)
        self.run_pass('remove_mutation', foo_script.graph)
        FileCheck().check_not("aten::add_").run(foo_script.graph)

        self.assertEqual(out_eager, foo_script(torch.tensor(5), True))

    def test_if_output_fail(self):
        @torch.jit.script
        def foo(cond: bool):
            li = []
            if cond:
                x = torch.tensor(1)
                li.append(x)
            else:
                x = torch.tensor(2)
            y = x.add_(2)
            return y, li

        self.run_pass('inline', foo.graph)
        self.run_pass('remove_mutation', foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)

        @torch.jit.script
        def foo(cond: bool, y):
            if cond:
                x = y
            else:
                x = torch.tensor(2)
            z = x.add_(2)
            return z

        self.run_pass('inline', foo.graph)
        self.run_pass('remove_mutation', foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)

    def test_special_mapped_op(self):
        def test_successful():
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.zero_()
            y.fill_(3)
            return x, y

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not("aten::zero_").check_not("aten::fill_").run(graph)
        self.assertEqual(test_successful(), fn())

        # full_like is not implemented for a tensor fill value

        def test_unsuccessful():
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.fill_(y)
            return x + x

        fn = torch.jit.script(test_unsuccessful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check('aten::fill_').run(graph)

    def test_lists_append(self):
        def successful_remove():
            return [i for i in range(5)]  # noqa: C416

        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        self.run_pass('loop_unrolling', graph)
        self.run_pass('remove_mutation', graph)
        self.run_pass('constant_propagation', graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(graph)
        self.assertEqual(successful_remove(), successful_remove())

        def intermediary_use():
            a = [1, 2]
            b = len(a)
            a.append(3)
            return a

        fn = torch.jit.script(intermediary_use)
        graph = fn.graph
        FileCheck().check("append").run(graph)
        self.run_pass('remove_mutation', graph)
        # it is possible to remove the append here but don't currently have the logic for it
        FileCheck().check_not("append").run(graph)
        self.assertEqual(intermediary_use(), fn())

    def test_common_pytorch_list_ops(self):
        for op in ["cat", "stack", "vstack", "hstack", "dstack"]:
            class OpMod(torch.nn.Module):
                def __init__(self, op):
                    super(OpMod, self).__init__()
                    self.op = torch_op

                def forward(self):
                    x = torch.tensor([1, 2, 3, 4])
                    x.add_(3)
                    y = [x, x]
                    return self.op(y) + 3

            torch_op = getattr(torch, op)
            mod = OpMod(torch_op)
            mod_script = torch.jit.script(mod)
            self.run_pass('remove_mutation', mod_script.forward.graph)
            FileCheck().check_not("aten::add_").run(mod_script.forward.graph)
            self.assertEqual(mod(), mod_script())

            # test that the output doesnt alias the input
            for inputs in [torch.rand(2, 2)], [torch.rand(2, 2) for _ in range(2)]:
                result = torch_op(inputs)
                sums = [ten.sum() for ten in result]

                for inp in inputs:
                    inp.fill_(10)

                self.assertEqual(sums, [ten.sum() for ten in result])


        @torch.jit.script
        def test_multiple_uses():
            x = torch.tensor([1, 2, 3, 4])
            x.add_(3)
            y = [x, x]
            return torch.cat(y), y

        self.run_pass('remove_mutation', mod_script.forward.graph)
        FileCheck().check("aten::add_").run(test_multiple_uses.graph)
