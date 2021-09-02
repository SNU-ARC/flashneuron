from tools.codegen.model import *

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
from tools.codegen import local

from typing import Union, Sequence, List, Optional

# This file describes the translation of JIT schema to the native functions API.
# This looks a lot like the C++ API (which makes historical sense, because the
# idea was you wrote native functions to implement functions in the C++ API),
# but over time we have evolved the C++ API without actually changing our
# native:: kernels.  The intention is to make native API and dispatcher API
# line up as closely as possible, since this results in the least overhead
# (no translation is needed from dispatcher API to native API).
#
# When a function is not use_c10_dispatcher: full, the dispatcher API actually
# coincides with the native:: API (e.g., we do as dumb as pass through as
# possible).

def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    if func.is_out_fn():
        name += '_out'
    if func.name.overload_name:
        name += f'_{func.name.overload_name}'
    return name

def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> CType:
    if str(t) == 'Tensor?':
        tensor_type: CType = BaseCType('Tensor', binds)
        if local.use_c10_dispatcher() is not UseC10Dispatcher.hacky_wrapper_for_legacy_signatures:
            tensor_type = OptionalCType(tensor_type)
        if mutable:
            return MutRefCType(tensor_type)
        else:
            return ConstRefCType(tensor_type)
    elif str(t) == 'Tensor?[]':
        return ConstRefCType(BaseCType("c10::List<c10::optional<Tensor>>", binds))
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds)

def returns_type(rs: Sequence[Return]) -> str:
    return cpp.returns_type(rs)

def argument_type(a: Argument, *, binds: ArgName) -> CType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)

def argument(a: Union[Argument, SelfArgument, TensorOptionsArguments], *, is_out: bool) -> List[Binding]:
    # Ideally, we NEVER default native functions.  However, there are a number
    # of functions that call native:: directly and rely on the defaulting
    # existing.  So for BC, we generate defaults for non-out variants (but not
    # for out variants, where it is impossible to generate an appropriate
    # default)
    should_default = not is_out or local.use_c10_dispatcher() is not UseC10Dispatcher.full
    if isinstance(a, Argument):
        default: Optional[str] = None
        if should_default and a.default is not None:
            default = cpp.default_expr(a.default, a.type)
        return [Binding(
            ctype=argument_type(a, binds=a.name),
            name=a.name,
            default=default,
            argument=a,
        )]
    elif isinstance(a, SelfArgument):
        # Erase SelfArgument from the distinction
        return argument(a.argument, is_out=is_out)
    elif isinstance(a, TensorOptionsArguments):
        if local.use_c10_dispatcher() == UseC10Dispatcher.hacky_wrapper_for_legacy_signatures:
            # TODO: expunge this logic entirely
            default = None
            if should_default:
                if all(x.default == "None" for x in a.all()):
                    default = '{}'
                elif a.dtype.default == "long":
                    default = 'at::kLong'  # TODO: this is wrong
            return [Binding(
                ctype=ConstRefCType(BaseCType('TensorOptions', 'options')),
                name='options',
                default=default,
                argument=a,
            )]
        else:
            assert local.use_c10_dispatcher() == UseC10Dispatcher.full
            default = None
            if should_default:
                default = '{}'
            # TODO: Not sure why the arguments assigned here are for
            # TensorOptionsArguments and not the constituent pieces.  It seems
            # to matter
            return [
                Binding(
                    ctype=OptionalCType(BaseCType('ScalarType', 'dtype')),
                    name='dtype',
                    default=default,
                    argument=a,
                ),
                Binding(
                    ctype=OptionalCType(BaseCType('Layout', 'layout')),
                    name='layout',
                    default=default,
                    argument=a,
                ),
                Binding(
                    ctype=OptionalCType(BaseCType('Device', 'device')),
                    name='device',
                    default=default,
                    argument=a,
                ),
                Binding(
                    ctype=OptionalCType(BaseCType('bool', 'pin_memory')),
                    name='pin_memory',
                    default=default,
                    argument=a,
                )]
    else:
        assert_never(a)

def arguments(func: FunctionSchema) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if local.use_c10_dispatcher() is UseC10Dispatcher.full:
        args.extend(func.arguments.non_out)
        args.extend(func.arguments.out)
    else:
        args.extend(func.arguments.out)
        args.extend(func.arguments.non_out)
    return [r for arg in args for r in argument(arg, is_out=func.is_out_fn())]
