use egg::{test_fn2, test_fn_not2, *};
use std::io::BufWriter;
use std::io::Write;
use std::fs::File;
use rayon::prelude::*;
use trinity::*;
use trinity::language::{TileLang, LoopAnalysis, SHAPE_TRACKER};
use trinity::shape::{ShapeTracker, TensorShape, Dimension};
use trinity::cost::{create_fine_grained_extractor};
use egg::*;
use std::sync::Once;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

// Helper function to set up shape tracker for tests
fn setup_shape_tracker(shapes: Vec<(&str, Vec<usize>)>) {
    SHAPE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        *tracker = ShapeTracker::new();
        for (name, dims) in shapes {
            tracker.add_tensor(name, dims);
        }
    });
}

// Helper function to get tensor shape from an eclass
fn get_tensor_shape(egraph: &EGraph, id: Id) -> Option<TensorShape> {
    egraph[id].data.tensor_shape.clone()
}

#[test]
fn extract_rmsnorm_qkv_attn_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 4096]),
        ("X_norm", vec![16, 4096]),
        ("WQ", vec![4096, 4096]),
        ("WK", vec![4096, 4096]),
        ("WV", vec![4096, 4096]),
        ("Q1", vec![16, 4096]),
        ("K1", vec![16, 4096]),
        ("V1", vec![16, 4096]),
        ("Q2", vec![16, 32, 128]),
        ("K2", vec![16, 32, 128]),
        ("V2", vec![16, 32, 128]),
        ("Q", vec![32, 16, 128]),
        ("K", vec![32, 16, 128]),
        ("V", vec![32, 16, 128]),
        ("K_cache", vec![32, 1040, 128]),
        ("V_cache", vec![32, 1040, 128]),
        ("C", vec![32, 16, 1040]),
        ("C_exp", vec![32, 16, 1040]),
        ("C_sum", vec![32, 16]),
        ("C_div", vec![32, 16, 1040]),
        ("O", vec![32, 16, 128]),
        ("O1", vec![16, 32, 128]),
        ("O2", vec![16, 4096]),
        ("X2", vec![16]),
    ]);

    let expr = "
(seq
    (loop 0 4096 tile_k k
        (store (tensor X2)
            (+
                (x (load (tensor X2) (index (fulltile))) 1)
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
                            4096
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor Q1,K1,V1)
                (+
                    (x (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ,WK,WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 1024 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 1024 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1040 tile_p p
            (store (tensor C)
                (*
                    (load (tensor Q) (index (tile h) (fulltile) (fulltile)))
                    (permute3
                        (load (input K_cache) (index (tile h) (tile p) (fulltile)))
                        0 2 1
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1040 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1040 tile_p p
            (store (tensor C_sum)
                (+
                    (x (load (tensor C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1040 tile_p p
            (store (tensor C_div)
                (/
                    (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (tensor C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1040 tile_p p
            (store (tensor O)
                (+
                    (x (load (tensor O) (index (tile h) (fulltile) (fulltile))) 1)
                    (*
                        (load (tensor C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4096 128 n
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
)))))))))))))
";
    let mut runner = run_until_saturated(
        expr,
        rules(),
        8,
    );

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/llama8b_rmsnorm_qkv_attn_cost6_kern2_1024.json", 6, 2) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }

    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/llama8b_rmsnorm_qkv_attn_cost6_kern2_1024.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            println!("{:?}", tile_sets);
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/llama8b_rmsnorm_qkv_attn_cost6_kern2_1024.txt").expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{}: {}", i, new_expr)  // Convert to String here
        })
        .collect::<Vec<String>>()  // Now collecting Vec<String>
        .iter()
        .for_each(|line| {
            writeln!(writer, "{}", line).expect("Failed to write to file");
        });
    
    writer.flush().expect("Failed to flush writer");

}

#[test]
fn extract_ffn_expressions() {
    setup_shape_tracker(vec![
        ("O2", vec![16, 4096]),
        ("WO", vec![4096, 4096]),
        ("attn_O1", vec![16, 4096]),
        ("X", vec![16, 4096]),
        ("attn_O2", vec![16, 4096]),
        ("attn_O3", vec![16]),
        ("attn_O_norm", vec![16, 4096]),
        ("WFF1a", vec![4096, 14336]),
        ("WFF1b", vec![4096, 14336]),
        ("FF1a", vec![16, 14336]),
        ("FF1b", vec![16, 14336]),
        ("FF1b_silu", vec![16, 14336]),
        ("FF1", vec![16, 14336]),
        ("FF2", vec![16, 4096]),
        ("WFF2", vec![14336, 4096]),
        ("O_FF", vec![16, 4096]),
    ]);
    let expr = "
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor attn_O1)
                (+
                    (x (load (tensor attn_O1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (input O2) (index (fulltile) (tile k)))
                        (load (input WO) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 tile_n n
        (store (tensor attn_O2)
            (+
                (load (tensor attn_O1) (index (fulltile) (tile n)))
                (load (input X) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor attn_O3)
            (+
                (x (load (tensor attn_O3) (index (fulltile))) 1)
                (rsum
                    (sqr (load (tensor attn_O2) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor attn_O_norm)
            (/
                (load (tensor attn_O2) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor attn_O3) (index (fulltile)))
                            4096
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 14336 tile_p p
        (loop 0 4096 tile_k k
            (store (tensor FF1a)
                (+
                    (x (load (tensor FF1a) (index (fulltile) (tile p))) 1)
                    (*
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1a) (index (tile k) (tile p)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 14336 tile_p p
        (loop 0 4096 tile_k k
            (store (tensor FF1b)
                (+
                    (x (load (tensor FF1b) (index (fulltile) (tile p))) 1)
                    (*
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1b) (index (tile k) (tile p)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 14336 tile_p p
        (store (tensor FF1b_silu)
            (x
                (load (tensor FF1b) (index (fulltile) (tile p)))
                (sigmoid
                    (load (tensor FF1b) (index (fulltile) (tile p)))
                )
            )
            (index (fulltile) (tile p))
        )
    )
(seq
    (loop 0 14336 tile_p p
        (store (tensor FF1)
            (x
                (load (tensor FF1a) (index (fulltile) (tile p)))
                (load (tensor FF1b_silu) (index (fulltile) (tile p)))
            )
            (index (fulltile) (tile p))
        )
    )
    (loop 0 4096 tile_n n
        (loop 0 14336 tile_p p
            (store (output FF2)
                (+
                    (x (load (output FF2) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor FF1) (index (fulltile) (tile p)))
                        (load (input WFF2) (index (tile p) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
))))))))
    ";
    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
    );

    // let all_possibilities = count_expressions_all_for_root(&runner);
    // println!("{:?}", all_possibilities);

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/llama8b_ffn_cost6_kern5_wo_scheduler2.json", 6, 5) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }

    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/llama8b_ffn_cost6_kern5_wo_scheduler2.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            println!("{:?}", tile_sets);
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/llama8b_ffn_cost6_kern5_wo_scheduler2.txt").expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{}: {}", i, new_expr)  // Convert to String here
        })
        .collect::<Vec<String>>()  // Now collecting Vec<String>
        .iter()
        .for_each(|line| {
            writeln!(writer, "{}", line).expect("Failed to write to file");
        });
    
    writer.flush().expect("Failed to flush writer");
}

// egg::test_fn2!{test_ffn, rules(),
// "
// (seq
//     (loop 0 4096 tile_n n
//         (loop 0 4096 tile_k k
//             (store (tensor attn_O1)
//                 (+
//                     (x (load (tensor attn_O1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (input O2) (index (fulltile) (tile k)))
//                         (load (input WO) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 4096 tile_n n
//         (store (tensor attn_O2)
//             (+
//                 (load (tensor attn_O1) (index (fulltile) (tile n)))
//                 (load (input X) (index (fulltile) (tile n)))
//             )
//             (index (fulltile) (tile n))
//         )
//     )
// (seq
//     (loop 0 4096 tile_k k
//         (store (tensor attn_O3)
//             (+
//                 (x (load (tensor attn_O3) (index (fulltile))) 1)
//                 (rsum
//                     (sqr (load (tensor attn_O2) (index (fulltile) (tile k))))
//                     1
//                 )
//             )
//             (index (fulltile))
//         )
//     )
// (seq
//     (loop 0 4096 tile_k k
//         (store (tensor attn_O_norm)
//             (/
//                 (load (tensor attn_O2) (index (fulltile) (tile k)))
//                 (bcast
//                     (sqrt
//                         (/
//                             (load (tensor attn_O3) (index (fulltile)))
//                             4096
//                         )
//                     )
//                     1
//                 )
//             )
//             (index (fulltile) (tile k))
//         )
//     )
// (seq
//     (loop 0 14336 tile_p p
//         (loop 0 4096 tile_k k
//             (store (tensor FF1a)
//                 (+
//                     (x (load (tensor FF1a) (index (fulltile) (tile p))) 1)
//                     (*
//                         (load (tensor attn_O_norm) (index (fulltile) (tile k)))
//                         (load (input WFF1a) (index (tile k) (tile p)))
//                     )
//                 )
//                 (index (fulltile) (tile p))
//             )
//         )
//     )
// (seq
//     (loop 0 14336 tile_p p
//         (loop 0 4096 tile_k k
//             (store (tensor FF1b)
//                 (+
//                     (x (load (tensor FF1b) (index (fulltile) (tile p))) 1)
//                     (*
//                         (load (tensor attn_O_norm) (index (fulltile) (tile k)))
//                         (load (input WFF1b) (index (tile k) (tile p)))
//                     )
//                 )
//                 (index (fulltile) (tile p))
//             )
//         )
//     )
// (seq
//     (loop 0 14336 tile_p p
//         (store (tensor FF1b_silu)
//             (x
//                 (load (tensor FF1b) (index (fulltile) (tile p)))
//                 (sigmoid
//                     (load (tensor FF1b) (index (fulltile) (tile p)))
//                 )
//             )
//             (index (fulltile) (tile p))
//         )
//     )
// (seq
//     (loop 0 14336 tile_p p
//         (store (tensor FF1)
//             (x
//                 (load (tensor FF1a) (index (fulltile) (tile p)))
//                 (load (tensor FF1b_silu) (index (fulltile) (tile p)))
//             )
//             (index (fulltile) (tile p))
//         )
//     )
// (seq
//     (loop 0 4096 tile_n n
//         (loop 0 14336 tile_p p
//             (store (output FF2)
//                 (+
//                     (x (load (output FF2) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (tensor FF1) (index (fulltile) (tile p)))
//                         (load (input WFF2) (index (tile p) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
//     (loop 0 4096 tile_n n
//         (store (output O_FF)
//             (+
//                 (load (output FF2) (index (fulltile) (tile n)))
//                 (load (tensor attn_O_norm) (index (fulltile) (tile n)))
//             )
//             (index (fulltile) (tile n))
//         )
//     )
// )))))))))
// "
// =>
// "
// (seq
//     (loop 0 4096 tile_n n
//         (seq
//             (loop 0 4096 tile_k k
//                 (store
//                     (tensor attn_O1)
//                     (+
//                         (x
//                             (load
//                                 (tensor attn_O1)
//                                 (index fulltile
//                                     (tile n))) 1)
//                         (*
//                             (load
//                                 (input O2)
//                                 (index fulltile
//                                     (tile k)))
//                             (load
//                                 (input WO)
//                                 (index
//                                     (tile k)
//                                     (tile n)))))
//                     (index fulltile
//                         (tile n))))
//             (store
//                 (tensor attn_O2)
//                 (+
//                     (load
//                         (tensor attn_O1)
//                         (index fulltile
//                             (tile n)))
//                     (load
//                         (input X)
//                         (index fulltile
//                             (tile n))))
//                 (index fulltile
//                     (tile n)))))
//     (seq
//         (loop 0 14336 tile_p p
//             (seq
//                 (loop 0 4096 tile_k k
//                     (seq
//                         (store
//                             (tensor attn_O3)
//                             (+
//                                 (x
//                                     (load
//                                         (tensor attn_O3)
//                                         (index fulltile)) 1)
//                                 (rsum
//                                     (sqr
//                                         (load
//                                             (tensor attn_O2)
//                                             (index fulltile
//                                                 (tile k)))) 1))
//                             (index fulltile))
//                         (seq
//                             (store
//                                 (tensor FF1a)
//                                 (+
//                                     (x
//                                         (load
//                                             (tensor FF1a)
//                                             (index fulltile
//                                                 (tile p))) 1)
//                                     (*
//                                         (load
//                                             (tensor attn_O2)
//                                             (index fulltile
//                                                 (tile k)))
//                                         (load
//                                             (input WFF1a)
//                                             (index
//                                                 (tile k)
//                                                 (tile p)))))
//                                 (index fulltile
//                                     (tile p)))
//                             (store
//                                 (tensor FF1b)
//                                 (+
//                                     (*
//                                         (load
//                                             (tensor attn_O2)
//                                             (index fulltile
//                                                 (tile k)))
//                                         (load
//                                             (input WFF1b)
//                                             (index
//                                                 (tile k)
//                                                 (tile p))))
//                                     (x
//                                         (load
//                                             (tensor FF1b)
//                                             (index fulltile
//                                                 (tile p))) 1))
//                                 (index fulltile
//                                     (tile p))))))
//                 (seq
//                     (store
//                         (tensor FF1a)
//                         (/
//                             (load
//                                 (tensor FF1a)
//                                 (index fulltile
//                                     (tile p)))
//                             (bcast
//                                 (sqrt
//                                     (/
//                                         (load
//                                             (tensor attn_O3)
//                                             (index fulltile)) 4096)) 1))
//                         (index fulltile
//                             (tile p)))
//                     (seq
//                         (store
//                             (tensor FF1b)
//                             (/
//                                 (load
//                                     (tensor FF1b)
//                                     (index fulltile
//                                         (tile p)))
//                                 (bcast
//                                     (sqrt
//                                         (/
//                                             (load
//                                                 (tensor attn_O3)
//                                                 (index fulltile)) 4096)) 1))
//                             (index fulltile
//                                 (tile p)))
//                         (seq 
//                         (store (tensor FF1b_silu)
//                                 (x
//                                     (load (tensor FF1b) (index (fulltile) (tile p)))
//                                     (sigmoid
//                                         (load (tensor FF1b) (index (fulltile) (tile p)))
//                                     )
//                                 )
//                                 (index (fulltile) (tile p))
//                             )
//                             (store
//                                 (tensor FF1)
//                                 (x
//                                     (load
//                                         (tensor FF1a)
//                                         (index fulltile
//                                             (tile p)))
//                                     (x
//                                         (load
//                                             (tensor FF1b)
//                                             (index fulltile
//                                                 (tile p)))
//                                         (sigmoid
//                                             (load
//                                                 (tensor FF1b)
//                                                 (index fulltile
//                                                     (tile p))))))
//                                 (index fulltile
//                                     (tile p))))))))
//         (seq
//             (loop 0 4096 tile_k k
//                 (store
//                     (tensor attn_O_norm)
//                     (/
//                         (load
//                             (tensor attn_O2)
//                             (index fulltile
//                                 (tile k)))
//                         (bcast
//                             (sqrt
//                                 (/
//                                     (load
//                                         (tensor attn_O3)
//                                         (index fulltile)) 4096)) 1))
//                     (index fulltile
//                         (tile k))))
            
//                 (loop 0 4096 tile_n n
//                     (seq
//                         (sloop 0 14336 tile_p p
//                             (store
//                                 (output FF2)
//                                 (+
//                                     (x 1
//                                         (load
//                                             (output FF2)
//                                             (index fulltile
//                                                 (tile n))))
//                                     (*
//                                         (load
//                                             (tensor FF1)
//                                             (index fulltile
//                                                 (tile p)))
//                                         (load
//                                             (input WFF2)
//                                             (index
//                                                 (tile p)
//                                                 (tile n)))))
//                                 (index fulltile
//                                     (tile n))))
//                         (store
//                             (tensor O_FF)
//                             (+
//                                 (load
//                                     (tensor attn_O_norm)
//                                     (index fulltile
//                                         (tile n)))
//                                 (load
//                                     (output FF2)
//                                     (index fulltile
//                                         (tile n))))
//                             (index fulltile
//                                 (tile n)))))
//                 )))

// "
// }