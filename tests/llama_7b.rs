use egg::{test_fn2, test_fn_not2, *};
use std::io::BufWriter;
use std::io::Write;
use std::fs::File;
use rayon::prelude::*;
use TileIR::*;
use TileIR::language::{TileLang, LoopAnalysis, SHAPE_TRACKER};
use TileIR::shape::{ShapeTracker, TensorShape, Dimension};
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
fn extract_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 4096]),
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
        ("K_cache", vec![32, 512+16, 128]),
        ("V_cache", vec![32, 528, 128]),
        ("C", vec![32, 16, 528]),
        ("C_exp", vec![32, 16, 528]),
        ("C_sum", vec![32, 16]),
        ("C_div", vec![32, 16, 528]),
        ("O", vec![32, 16, 128]),
        ("O1", vec![16, 32, 128]),
        ("O2", vec![16, 4096]),

    ]);
    
    let expr = "
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor Q1)
                (+
                    (x (load (tensor Q1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (input X) (index (fulltile) (tile k)))
                        (load (input WQ) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor K1)
                (+
                    (x (load (tensor K1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (input X) (index (fulltile) (tile k)))
                        (load (input WK) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor V1)
                (+
                    (x (load (tensor V1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (input X) (index (fulltile) (tile k)))
                        (load (input WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor Q2)
            (unsqueeze (load (tensor Q1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor K2)
            (unsqueeze (load (tensor K1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor V2)
            (unsqueeze (load (tensor V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor Q)
            (permute3
                (load (tensor Q2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor K)
            (permute3
                (load (tensor K2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor V)
            (permute3
                (load (tensor V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4096 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4096 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
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
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
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
        (loop 0 528 tile_p p
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
        (loop 0 528 tile_p p
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
)))))))))))))))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        40,
        // rules(),
    );
    // postprocess_egraph(&mut runner.egraph);
    println!("----------------------------------------------");
    measure_enode_proportions(&runner.egraph, true);
    // List all expressions for the root e-class
    // println!("All equivalent expressions for your input:");
    // let root_expressions = list_expressions_all(&runner);
    // println!("[All] There are {:?} expressions", root_expressions.len());

    // let total_expressions = count_expressions_all_for_root(&runner);
    // println!("[All] There are {:?} expressions", total_expressions);

    // let cost_expressions = list_expressions_with_target_cost(&runner);
    // println!("[Cost model] There are {:?} expressions", cost_expressions.len());

    // let cost_expressions_v3 = list_expressions_with_target_cost_v3(&runner);
    // println!("[Cost model v3] There are {:?} expressions", cost_expressions_v3.len());

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/llama7b_attacc_cost3_kern2.json", 3, 2) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/llama7b_attacc_cost3_kern2.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/TileIR/expressions/llama7b_attacc_cost3_kern2.txt").expect("Failed to create file");
    // let file = File::create("tmp.txt").expect("aa");
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

egg::test_fn2!{test_expressions, rules(),
"
(seq
    (loop 0 4096 tile_k k
        (store (tensor X1)
            (sqr (load (input X) (index (fulltile) (tile k))))
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor X2)
            (+
                (x (load (tensor X2) (index (fulltile))) 1)
                (rsum
                    (load (tensor X1) (index (fulltile) (tile k)))
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
                            (load (tensor X2) (fulltile))
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
            (store (tensor Q1)
                (+
                    (x (load (tensor Q1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor K1)
                (+
                    (x (load (tensor K1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WK) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor V1)
                (+
                    (x (load (tensor V1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor Q2)
            (unsqueeze (load (tensor Q1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor K2)
            (unsqueeze (load (tensor K1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor V2)
            (unsqueeze (load (tensor V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor Q)
            (permute3
                (load (tensor Q2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor K)
            (permute3
                (load (tensor K2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor V)
            (permute3
                (load (tensor V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4096 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4096 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
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
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
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
        (loop 0 528 tile_p p
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
        (loop 0 528 tile_p p
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
))))))))))))))))))))
"
}