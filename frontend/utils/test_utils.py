from typing import Dict, List, Optional, Set, Tuple, Union

import ir.AST as T

Dim = Union[int, Tuple[str, str]]


def validate_primfunc_ast(primfunc: T.PrimFunc) -> List[str]:
    errors: List[str] = []
    tensor_shapes: dict[str, List[int]] = {}
    for info in primfunc.input_tensors:
        tensor_shapes[info.name] = info.shape
    tensor_shapes[primfunc.output_tensor.name] = primfunc.output_tensor.shape
    for info in primfunc.allocated_tensors:
        tensor_shapes[info.name] = info.shape

    axis_usage: Dict[str, Dict[str, Set[int]]] = {}
    loop_sizes: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    current_tensor: List[Optional[str]] = [None]
    op_stack: List[str] = []

    def _add_error(msg: str) -> None:
        tensor = current_tensor[0]
        op_path = " > ".join(op_stack)
        if op_path:
            msg = f"{msg} [op {op_path}]"
        if tensor:
            errors.append(f"{msg} (tensor '{tensor}')")
        else:
            errors.append(msg)

    def _with_op(op_name: str, fn):
        op_stack.append(op_name)
        try:
            return fn()
        finally:
            op_stack.pop()

    def _record_axis_usage(index: T.Index, tensor_name: str) -> None:
        for dim, idx in enumerate(index.indices):
            if isinstance(idx, (T.Tile, T.TileOffset)):
                axis_usage.setdefault(idx.name, {}).setdefault(tensor_name, set()).add(dim)

    def _dim_token(kind: str, name: str) -> Dim:
        return (kind, name)

    def _format_dim(dim: Dim) -> str:
        if isinstance(dim, tuple):
            kind, name = dim
            if kind == "bcast":
                return "bcast"
            return f"{kind}_{name}"
        return str(dim)

    def _format_shape(shape: List[Dim]) -> str:
        return "[" + ", ".join(_format_dim(dim) for dim in shape) + "]"

    def _index_shape(tensor: str, index: T.Index) -> Optional[List[Dim]]:
        shape = tensor_shapes.get(tensor)
        if shape is None:
            return None
        out: List[Dim] = []
        for dim, idx in enumerate(index.indices):
            if isinstance(idx, T.FullTile):
                out.append(shape[dim])
            elif isinstance(idx, T.ConstTile):
                out.append(idx.interval)
            elif isinstance(idx, (T.Tile, T.TileOffset)):
                out.append(_dim_token("tile", idx.name))
            elif isinstance(idx, T.Elem):
                out.append(1)
            elif isinstance(idx, T.Const):
                out.append(1)
            else:
                return None
        return out

    def _dims_match(a: Dim, b: Dim) -> bool:
        if a == b:
            return True
        if isinstance(a, tuple):
            size = loop_sizes.get(a[1], (None, None))
            if a[0] in ("tile", "shifted_tile") and isinstance(b, int) and size[0] == b:
                return True
            if a[0] == "elem" and isinstance(b, int) and b == 1:
                return True
        if isinstance(b, tuple):
            size = loop_sizes.get(b[1], (None, None))
            if b[0] in ("tile", "shifted_tile") and isinstance(a, int) and size[0] == a:
                return True
            if b[0] == "elem" and isinstance(a, int) and a == 1:
                return True
        if isinstance(a, tuple) or isinstance(b, tuple):
            if isinstance(a, tuple) and a[0] == "bcast":
                return True
            if isinstance(b, tuple) and b[0] == "bcast":
                return True
            return False
        return False

    def _shapes_match(a: List[Dim], b: List[Dim]) -> bool:
        if not a or not b:
            return True
        if len(a) != len(b):
            return False
        return all(_dims_match(x, y) for x, y in zip(a, b))

    def _check_index(tensor: str, index: T.Index) -> None:
        shape = tensor_shapes.get(tensor)
        if shape is not None and len(index.indices) != len(shape):
            _add_error(
                f"Tensor '{tensor}' rank {len(shape)} vs index rank {len(index.indices)}"
            )
        for idx in index.indices:
            if not isinstance(idx, (T.Tile, T.FullTile, T.ConstTile, T.Elem, T.TileOffset)):
                _add_error(
                    f"Invalid index node for tensor '{tensor}': {type(idx)}"
                )
        if shape is not None:
            for dim, idx in enumerate(index.indices):
                if shape[dim] == 1 and not isinstance(idx, (T.FullTile, T.Const, T.ConstTile, T.TileOffset, T.Elem)):
                    _add_error(
                        f"Tensor '{tensor}' dim {dim} is 1 but index is {type(idx)}"
                    )

    def _infer_shape(node: T.ASTNode) -> Optional[List[Dim]]:
        if isinstance(node, (T.Const, T.VarRef, T.Arange)):
            return []
        if isinstance(node, T.Load):
            return _index_shape(node.tensor.name, node.index)
        if isinstance(node, T.Tensor):
            shape = tensor_shapes.get(node.name)
            return shape[:] if shape is not None else None
        if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.GenericBinary)):
            op_name = node.__class__.__name__.lower()
            def _eval():
                left = _infer_shape(node.left)
                right = _infer_shape(node.right)
                if left is None or right is None:
                    _add_error("Cannot infer shape for elementwise op")
                    return None
                if not left:
                    return right
                if not right:
                    return left
                if not _shapes_match(left, right):
                    _add_error(
                        f"Elementwise shape mismatch: {_format_shape(left)} vs {_format_shape(right)}"
                    )
                return left
            return _with_op(op_name, _eval)
        if isinstance(node, T.Matmul):
            def _eval():
                left = _infer_shape(node.left)
                right = _infer_shape(node.right)
                if left is None or right is None:
                    _add_error("Cannot infer shape for matmul")
                    return None
                if len(left) < 2 or len(right) < 2:
                    _add_error(
                        f"Matmul expects >=2D shapes, got {_format_shape(left)} and {_format_shape(right)}"
                    )
                    return None
                if len(left) != len(right):
                    _add_error(f"Matmul rank mismatch: {_format_shape(left)} vs {_format_shape(right)}")
                    return None
                if left[:-2] != right[:-2]:
                    _add_error(
                        f"Matmul batch mismatch: {_format_shape(left[:-2])} vs {_format_shape(right[:-2])}"
                    )
                if not _dims_match(left[-1], right[-2]):
                    _add_error(f"Matmul K mismatch: {left[-1]} vs {right[-2]}")
                return left[:-2] + [left[-2], right[-1]]
            return _with_op("matmul", _eval)
        if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
            op_name = node.__class__.__name__.lower()
            return _with_op(op_name, lambda: _infer_shape(node.val))
        if isinstance(node, T.Broadcast):
            def _eval():
                shape = _infer_shape(node.val)
                if shape is None:
                    _add_error("Cannot infer shape for broadcast")
                    return None
                axis = node.axis
                if axis < 0 or axis > len(shape):
                    _add_error(f"Broadcast axis out of range: {axis}")
                    return None
                return shape[:axis] + [_dim_token("bcast", str(axis))] + shape[axis:]
            return _with_op("broadcast", _eval)
        if isinstance(node, (T.Squeeze, T.Unsqueeze)):
            op_name = "squeeze" if isinstance(node, T.Squeeze) else "unsqueeze"
            def _eval():
                shape = _infer_shape(node.val)
                if shape is None:
                    _add_error("Cannot infer shape for squeeze/unsqueeze")
                    return None
                axis = node.axis
                if isinstance(node, T.Squeeze):
                    if axis < 0 or axis >= len(shape):
                        _add_error(f"Squeeze axis out of range: {axis}")
                        return None
                    dim = shape[axis]
                    if dim != 1:
                        _add_error(f"Squeeze axis {axis} dim is not 1: {dim}")
                    return shape[:axis] + shape[axis + 1 :]
                if axis < 0 or axis > len(shape):
                    _add_error(f"Unsqueeze axis out of range: {axis}")
                    return None
                return shape[:axis] + [1] + shape[axis:]
            return _with_op(op_name, _eval)
        if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin)):
            op_name = node.__class__.__name__.lower()
            def _eval():
                shape = _infer_shape(node.val)
                if shape is None:
                    _add_error("Cannot infer shape for reduce")
                    return None
                axis = node.axis
                if axis < 0 or axis >= len(shape):
                    _add_error(f"Reduce axis out of range: {axis}")
                    return None
                return shape[:axis] + shape[axis + 1 :]
            return _with_op(op_name, _eval)
        if isinstance(node, T.Concat):
            def _eval():
                left = _infer_shape(node.a)
                right = _infer_shape(node.b)
                if left is None or right is None:
                    _add_error("Cannot infer shape for concat")
                    return None
                if len(left) != len(right):
                    _add_error(f"Concat rank mismatch: {_format_shape(left)} vs {_format_shape(right)}")
                    return None
                axis = node.axis
                if axis < 0 or axis >= len(left):
                    _add_error(f"Concat axis out of range: {axis}")
                    return None
                for i, (la, rb) in enumerate(zip(left, right)):
                    if i == axis:
                        continue
                    if not _dims_match(la, rb):
                        _add_error(
                            f"Concat dim mismatch at axis {i}: {_format_dim(la)} vs {_format_dim(rb)}"
                        )
                if isinstance(left[axis], int) and isinstance(right[axis], int):
                    out_axis = left[axis] + right[axis]
                else:
                    out_axis = _dim_token("concat", str(axis))
                out = left[:]
                out[axis] = out_axis
                return out
            return _with_op("concat", _eval)
        if isinstance(node, T.Take):
            def _eval():
                data_shape = _infer_shape(node.data)
                idx_shape = _infer_shape(node.indices)
                if data_shape is None or idx_shape is None:
                    _add_error("Cannot infer shape for take")
                    return None
                axis = node.axis
                if axis < 0 or axis >= len(data_shape):
                    _add_error(f"Take axis out of range: {axis}")
                    return None
                if len(idx_shape) != 1:
                    _add_error(f"Take indices must be 1D, got shape {_format_shape(idx_shape)}")
                    return None
                if isinstance(node.index, T.Index):
                    return _index_shape(node.data.name, node.index)
                return None
            return _with_op("take", _eval)
        if isinstance(node, T.Permute3):
            def _eval():
                shape = _infer_shape(node.val)
                if shape is None:
                    _add_error("Cannot infer shape for permute3")
                    return None
                if len(shape) != 3:
                    _add_error(f"Permute3 expects 3D shape, got {_format_shape(shape)}")
                    return None
                return [shape[node.d0], shape[node.d1], shape[node.d2]]
            return _with_op("permute3", _eval)
        if isinstance(node, T.GenericCall):
            if node.func_name == "select" and len(node.args) == 3:
                def _eval():
                    cond_shape = _infer_shape(node.args[0])
                    true_shape = _infer_shape(node.args[1])
                    false_shape = _infer_shape(node.args[2])
                    if cond_shape is None or true_shape is None or false_shape is None:
                        _add_error("Cannot infer shape for select")
                        return None
                    if not _shapes_match(true_shape, false_shape):
                        _add_error(
                            f"Select shape mismatch: {_format_shape(true_shape)} vs {_format_shape(false_shape)}"
                        )
                        return true_shape
                    return true_shape
                return _with_op("select", _eval)
            if node.func_name in ("max", "min") and len(node.args) == 2:
                def _eval():
                    left = _infer_shape(node.args[0])
                    right = _infer_shape(node.args[1])
                    if left is None or right is None:
                        _add_error(f"Cannot infer shape for {node.func_name}")
                        return None
                    if not left:
                        return right
                    if not right:
                        return left
                    if not _shapes_match(left, right):
                        _add_error(
                            f"{node.func_name} shape mismatch: {_format_shape(left)} vs {_format_shape(right)}"
                        )
                    return left
                return _with_op(node.func_name, _eval)
            if node.func_name in ("pow", "power") and len(node.args) >= 1:
                def _eval():
                    base_shape = _infer_shape(node.args[0])
                    exp_shape = _infer_shape(node.args[1]) if len(node.args) >= 2 else []
                    if base_shape is None or exp_shape is None:
                        _add_error("Cannot infer shape for pow")
                        return None
                    if not base_shape:
                        return exp_shape
                    if not exp_shape:
                        return base_shape
                    if not _shapes_match(base_shape, exp_shape):
                        _add_error(
                            f"pow shape mismatch: {_format_shape(base_shape)} vs {_format_shape(exp_shape)}"
                        )
                    return base_shape
                return _with_op("pow", _eval)
            if node.func_name == "erf" and len(node.args) == 1:
                return _with_op("erf", lambda: _infer_shape(node.args[0]))
            if node.func_name == "abs" and len(node.args) == 1:
                return _with_op("abs", lambda: _infer_shape(node.args[0]))
            if node.func_name == "transpose" and node.args:
                def _eval():
                    shape = _infer_shape(node.args[0])
                    if shape is None:
                        _add_error("Cannot infer shape for transpose")
                        return None
                    if len(node.args) == 1:
                        if len(shape) == 2:
                            return [shape[1], shape[0]]
                        return shape
                    perm: List[int] = []
                    for arg in node.args[1:]:
                        if isinstance(arg, T.Const) and isinstance(arg.value, int):
                            perm.append(arg.value)
                        else:
                            _add_error("Transpose permutation must be constant ints")
                            return shape
                    if len(perm) != len(shape):
                        _add_error(
                            f"Transpose perm length mismatch: {perm} vs {_format_shape(shape)}"
                        )
                        return shape
                    return [shape[p] for p in perm]
                return _with_op("transpose", _eval)
            for arg in node.args:
                _infer_shape(arg)
            return None
        return None

    def _const_int(node: T.ASTNode) -> Optional[int]:
        if isinstance(node, T.Const) and isinstance(node.value, int):
            return node.value
        return None

    def visit(node: T.ASTNode) -> None:
        if isinstance(node, T.Loop):
            prev = loop_sizes.get(node.loop_var)
            tile_size: Optional[int] = None
            elem_size: Optional[int] = None
            start = _const_int(node.start)
            end = _const_int(node.end)
            if node.tile_name.isdigit():
                tile_size = int(node.tile_name)
                if start is not None and end is not None and tile_size != 0:
                    span = end - start
                    if span % tile_size == 0:
                        elem_size = span // tile_size
            else:
                if start is not None and end is not None:
                    tile_size = end - start
            if tile_size is not None:
                loop_sizes[node.loop_var] = (tile_size, elem_size)
            visit(node.body)
            if prev is None:
                loop_sizes.pop(node.loop_var, None)
            else:
                loop_sizes[node.loop_var] = prev
            return
        if isinstance(node, T.Block):
            for stmt in node.stmts:
                visit(stmt)
            return
        if isinstance(node, T.Seq):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, T.If):
            visit(node.then_branch)
            if node.else_branch:
                visit(node.else_branch)
            return
        if isinstance(node, T.Let):
            visit(node.value)
            visit(node.body)
            return
        if isinstance(node, T.Store):
            current_tensor[0] = node.tensor.name
            if node.tensor.name not in tensor_shapes:
                _add_error(f"Unknown tensor in store: {node.tensor.name}")
            _check_index(node.tensor.name, node.index)
            _record_axis_usage(node.index, node.tensor.name)
            store_shape = _index_shape(node.tensor.name, node.index)
            value_shape = _infer_shape(node.value)
            if store_shape is None or value_shape is None:
                _add_error(f"Cannot infer store/value shape for '{node.tensor.name}'")
            elif not _shapes_match(store_shape, value_shape):
                _add_error(
                    f"Store/value shape mismatch for '{node.tensor.name}': {store_shape} vs {value_shape}"
                )
            visit(node.value)
            current_tensor[0] = None
            return
        if isinstance(node, T.Load):
            current_tensor[0] = node.tensor.name
            if node.tensor.name not in tensor_shapes:
                _add_error(f"Unknown tensor in load: {node.tensor.name}")
            _check_index(node.tensor.name, node.index)
            _record_axis_usage(node.index, node.tensor.name)
            current_tensor[0] = None
            return
        if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul)):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, T.Take):
            visit(node.data)
            visit(node.indices)
            visit(node.index)
            return
        if isinstance(node, T.GenericBinary):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            visit(node.val)
            return
        if isinstance(node, T.Broadcast):
            visit(node.val)
            return
        if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin)):
            visit(node.val)
            return
        if isinstance(node, (T.Squeeze, T.Unsqueeze)):
            visit(node.val)
            return
        if isinstance(node, T.GenericCall):
            for arg in node.args:
                visit(arg)
            return
        if isinstance(node, T.Cast):
            visit(node.val)
            return
        if isinstance(node, T.Concat):
            visit(node.a)
            visit(node.b)
            return

    visit(primfunc.root_node)
    for axis, tensors in axis_usage.items():
        for tensor, dims in tensors.items():
            if len(dims) > 1:
                current_tensor[0] = tensor
                _add_error(
                    f"Loop var '{axis}' used for multiple axes of tensor '{tensor}': {sorted(dims)}"
                )
                current_tensor[0] = None
    return errors


def validate_main_func(main_func: T.MainFunc, context: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    known: set[str] = {t.name for t in main_func.input_tensors}
    known.update(t.name for t in main_func.intermediate_tensors)
    known.update(t.name for t in main_func.output_tensors)

    produced: set[str] = set(t.name for t in main_func.input_tensors)
    for call in main_func.calls:
        for t in call.input_tensors:
            if t.name not in produced:
                errors.append(f"Call {call.primfunc.name} uses unknown input '{t.name}'")
        produced.add(call.out_var_tensor.name)

        for err in validate_primfunc_ast(call.primfunc):
            errors.append(f"{call.primfunc.name}: {err}")

        if call.out_var_tensor.name not in known and call.out_var_tensor.name not in produced:
            errors.append(f"Call output '{call.out_var_tensor.name}' not in main tensors")

    if context:
        return [f"{context}: {err}" for err in errors]
    return errors


def validate_main_func_errors(main_func: T.MainFunc, context: Optional[str] = None) -> List[str]:
    errors = validate_main_func(main_func, context=context)
    if errors:
        print("\nValidation errors:")
        for err in errors:
            print(f"- {err}")
    return errors
