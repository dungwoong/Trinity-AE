// new_demo: RMSNorm + Linear, SwiGLU and LoRA through Trinity's optimizer, one shape each, with the same pipeline as
// tests/ffn.rs (run_until_saturated with rules(), 10 iterations; list_expressions_with_target_cost_v3_part1 with
// cost 6 / 5 kernels; list_expressions_from_semi_with_cost; postprocess_v2). The input programs are the unfused ones,
// written in the style of tests/ffn.rs, except that M is tiled ((tile m) instead of (fulltile)): these are prefill-size
// shapes, not M = 16. Each candidate list is written to backend/new_demo/cases/<workload>_candidates.txt, one
// "<id>: <expression>" per line, as tests/ffn.rs writes expressions/*.txt.
//
//   cargo test --release --test new_demo -- --nocapture --test-threads=1
use egg::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use trinity::language::SHAPE_TRACKER;
use trinity::shape::ShapeTracker;
use trinity::*;

fn setup_shape_tracker(shapes: Vec<(&str, Vec<usize>)>) {
    SHAPE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        *tracker = ShapeTracker::new();
        for (name, dims) in shapes {
            tracker.add_tensor(name, dims);
        }
    });
}

fn run(name: &str, expr: &str) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let semi_path = manifest.join(format!("expressions/semi/new_demo_{}.json", name));
    let out_path = manifest.join(format!("../backend/new_demo/cases/{}_candidates.txt", name));

    let start = std::time::Instant::now();
    let runner = run_until_saturated(expr, rules(), 10);
    println!("[{}] saturation: {:?}, {} e-classes, {} e-nodes, stop: {:?}", name, start.elapsed(),
             runner.egraph.number_of_classes(), runner.egraph.total_number_of_nodes(), runner.stop_reason);

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 5) {
        Ok(count) => println!("[{}] saved {} semi-expressions", name, count),
        Err(e) => panic!("save error: {}", e),
    }
    let (expressions, tile_sets) =
        list_expressions_from_semi_with_cost(&runner, semi_path.to_str().unwrap(), usize::MAX)
            .expect("load error");
    println!("[{}] {} final expressions, tile sets {:?}, total {:?}", name, expressions.len(), tile_sets,
             start.elapsed());

    let file = File::create(&out_path).expect("failed to create output file");
    let mut writer = BufWriter::new(file);
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| format!("{}: {}", i, postprocess_v2(expr, &tile_sets)))
        .collect::<Vec<String>>()
        .iter()
        .for_each(|line| writeln!(writer, "{}", line).expect("write failed"));
    writer.flush().expect("flush failed");
    println!("[{}] wrote {}", name, out_path.display());
}

// RMSNorm + Linear, M=4096 N=1024 K=4096: S = rowsum(X^2); Xn = X / sqrt(S / K); O = Xn @ W.
#[test]
fn rmsnorm_linear() {
    setup_shape_tracker(vec![
        ("X", vec![4096, 4096]),
        ("S", vec![4096]),
        ("Xn", vec![4096, 4096]),
        ("W", vec![4096, 1024]),
        ("O", vec![4096, 1024]),
    ]);
    let expr = "
(seq
    (loop 0 4096 tile_m m
        (loop 0 4096 tile_k k
            (store (tensor S)
                (+
                    (* (load (tensor S) (index (tile m))) 1)
                    (rsum (sqr (load (input X) (index (tile m) (tile k)))) 1)
                )
                (index (tile m))
            )
        )
    )
(seq
    (loop 0 4096 tile_m m
        (loop 0 4096 tile_k k
            (store (tensor Xn)
                (/
                    (load (input X) (index (tile m) (tile k)))
                    (bcast (sqrt (/ (load (tensor S) (index (tile m))) 4096)) 1)
                )
                (index (tile m) (tile k))
            )
        )
    )
    (loop 0 4096 tile_m m
        (loop 0 1024 tile_n n
            (loop 0 4096 tile_k k
                (store (output O)
                    (+
                        (* (load (output O) (index (tile m) (tile n))) 1)
                        (@
                            (load (tensor Xn) (index (tile m) (tile k)))
                            (load (input W) (index (tile k) (tile n)))
                        )
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    )
))
    ";
    run("rmsnorm_linear", expr);
}

// SwiGLU, M=4096 N=14336 K=4096: G = X @ Wg; U = X @ Wu; O = (G * sigmoid(G)) * U.
#[test]
fn swiglu() {
    setup_shape_tracker(vec![
        ("X", vec![4096, 4096]),
        ("Wg", vec![4096, 14336]),
        ("Wu", vec![4096, 14336]),
        ("G", vec![4096, 14336]),
        ("U", vec![4096, 14336]),
        ("O", vec![4096, 14336]),
    ]);
    let expr = "
(seq
    (loop 0 4096 tile_m m
        (loop 0 14336 tile_n n
            (loop 0 4096 tile_k k
                (store (tensor G)
                    (+
                        (* (load (tensor G) (index (tile m) (tile n))) 1)
                        (@
                            (load (input X) (index (tile m) (tile k)))
                            (load (input Wg) (index (tile k) (tile n)))
                        )
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    )
(seq
    (loop 0 4096 tile_m m
        (loop 0 14336 tile_n n
            (loop 0 4096 tile_k k
                (store (tensor U)
                    (+
                        (* (load (tensor U) (index (tile m) (tile n))) 1)
                        (@
                            (load (input X) (index (tile m) (tile k)))
                            (load (input Wu) (index (tile k) (tile n)))
                        )
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    )
    (loop 0 4096 tile_m m
        (loop 0 14336 tile_n n
            (store (output O)
                (*
                    (*
                        (load (tensor G) (index (tile m) (tile n)))
                        (sigmoid (load (tensor G) (index (tile m) (tile n))))
                    )
                    (load (tensor U) (index (tile m) (tile n)))
                )
                (index (tile m) (tile n))
            )
        )
    )
))
    ";
    run("swiglu", expr);
}

// LoRA, rank 16, M=N=K=4096: T = X @ A; Y = X @ W; O = Y + T @ B.
#[test]
fn lora() {
    setup_shape_tracker(vec![
        ("X", vec![4096, 4096]),
        ("A", vec![4096, 16]),
        ("B", vec![16, 4096]),
        ("W", vec![4096, 4096]),
        ("T", vec![4096, 16]),
        ("Y", vec![4096, 4096]),
        ("O", vec![4096, 4096]),
    ]);
    let expr = "
(seq
    (loop 0 4096 tile_m m
        (loop 0 4096 tile_k k
            (store (tensor T)
                (+
                    (* (load (tensor T) (index (tile m) (fulltile))) 1)
                    (@
                        (load (input X) (index (tile m) (tile k)))
                        (load (input A) (index (tile k) (fulltile)))
                    )
                )
                (index (tile m) (fulltile))
            )
        )
    )
(seq
    (loop 0 4096 tile_m m
        (loop 0 4096 tile_n n
            (loop 0 4096 tile_k k
                (store (tensor Y)
                    (+
                        (* (load (tensor Y) (index (tile m) (tile n))) 1)
                        (@
                            (load (input X) (index (tile m) (tile k)))
                            (load (input W) (index (tile k) (tile n)))
                        )
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    )
    (loop 0 4096 tile_m m
        (loop 0 4096 tile_n n
            (store (output O)
                (+
                    (load (tensor Y) (index (tile m) (tile n)))
                    (@
                        (load (tensor T) (index (tile m) (fulltile)))
                        (load (input B) (index (fulltile) (tile n)))
                    )
                )
                (index (tile m) (tile n))
            )
        )
    )
))
    ";
    run("lora", expr);
}
