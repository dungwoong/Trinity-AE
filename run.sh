#!/bin/bash

CMD='RUSTFLAGS="-A warnings" cargo test --test keyformer count -- --nocapture'

echo ">>> Running command: $CMD"
eval $CMD