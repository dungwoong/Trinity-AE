import torch
import torchvision
import tvm
from tvm import relay

# 1. 사전 학습된 PyTorch 모델 준비
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 2. 모델을 TorchScript로 변환 (Tracing 방식)
# TVM이 입력 크기를 알 수 있도록 가상의 입력(Dummy Input)이 필요합니다.
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# 3. TVM 프론트엔드를 사용하여 Relay IR로 변환
input_name = "input0"  # 입력 레이어의 이름 설정
shape_list = [(input_name, input_shape)]

# relay.frontend.from_pytorch가 핵심 함수입니다.
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# 4. 결과 출력 (Relay IR 확인)
print("######### TVM Relay IR (Abstract Syntax Tree) #########")
print(mod.astext(show_meta_data=False))