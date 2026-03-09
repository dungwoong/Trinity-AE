# Optimizer

## Custom model run

This test reads IR and shapes from the frontend export path and writes optimized
expressions to `optimizer/expressions`.

Expected input files:
- `frontend/outputs/trinity/{model_name}/ir.txt`
- `frontend/outputs/trinity/{model_name}/shapes.json`

Run the test:

```bash
TRINITY_MODEL_NAME={model_name} cargo test --test test_custom_model optimize_exported_mainfunc -- --nocapture
TRINITY_MODEL_NAME=Roco cargo test --test test_custom_model optimize_exported_mainfunc -- --nocapture  # example
```

Output:
- `optimizer/expressions/{model_name}_cost6_kern2.txt`
