import torch
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

def to_relax(model, example_input):
    """PyTorch 모델을 Relax IR로 변환"""
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
