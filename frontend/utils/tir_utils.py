import torch
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def _patch_relax_block_builder():
    block_builder_cls = relax.BlockBuilder

    if getattr(block_builder_cls, "_trinity_patch_applied", False):
        return

    func_stacks = {}

    def _stack(self):
        return func_stacks.setdefault(id(self), [])

    def patched_init(self, mod=None):
        self.__init_handle_by_constructor__(
            relax.block_builder._ffi_api.BlockBuilderCreate, mod
        )
        _stack(self)

    def patched_current_func(self):
        stack = _stack(self)
        if stack:
            return stack[-1]
        raise RuntimeError(
            "Cannot access BlockBuilder._func when outside a bb.function() block"
        )

    def patched_enter_function_scope(self, func_scope):
        block_builder_cls._stack.append(self)
        _stack(self).append(func_scope)
        self.begin_scope(func_scope._params)
        self._begin_binding_block()

    def patched_exit_function_scope(self, exc_type, exc_val, exc_tb):
        current_func = patched_current_func(self)
        is_emit_func_output_called = current_func._is_emit_func_output_called
        _stack(self).pop()

        assert block_builder_cls._stack
        assert block_builder_cls._stack[-1] is self
        block_builder_cls._stack.pop()

        if exc_type is None and not is_emit_func_output_called:
            raise RuntimeError("emit_func_output must be called in a relax function.")

    block_builder_cls.__init__ = patched_init
    block_builder_cls._func = property(patched_current_func)
    block_builder_cls._enter_function_scope = patched_enter_function_scope
    block_builder_cls._exit_function_scope = patched_exit_function_scope
    block_builder_cls._trinity_patch_applied = True


def to_relax(model, example_input):
    """PyTorch 모델을 Relax IR로 변환"""
    _patch_relax_block_builder()
    model.eval()
    with torch.no_grad():
        # export 함수는 tuple of tensors를 받아야 함
        if not isinstance(example_input, tuple):
            example_input = (example_input,)
        exported = export(model, example_input)
    user_output_count = len(getattr(exported.graph_signature, "user_outputs", []) or [])
    return from_exported_program(exported, keep_params_as_input=True), user_output_count

def to_tir(relax_mod):
    """Relax IR을 TIR로 lowering"""
    return relax.transform.LegalizeOps()(relax_mod)
