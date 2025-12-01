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
fn extract_rmsnorm_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 4544]),
        ("WQ", vec![4544, 4544]),
        ("WK", vec![4544, 4544]),
        ("WV", vec![4544, 4544]),
        ("Q1", vec![16, 4544]),
        ("K1", vec![16, 4544]),
        ("V1", vec![16, 4544]),
        ("Q2", vec![16, 71, 64]),
        ("K2", vec![16, 71, 64]),
        ("V2", vec![16, 71, 64]),
        ("Q", vec![71, 16, 64]),
        ("K", vec![71, 16, 64]),
        ("V", vec![71, 16, 64]),
        ("K_cache", vec![71, 512+16, 64]),
        ("V_cache", vec![71, 528, 64]),
        ("C", vec![71, 16, 528]),
        ("C_exp", vec![71, 16, 528]),
        ("C_sum", vec![71, 16]),
        ("C_div", vec![71, 16, 528]),
        ("O", vec![71, 16, 64]),
        ("O1", vec![16, 71, 64]),
        ("O2", vec![16, 4544]),
    ]);
    
    let expr = "
(seq
    (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (fulltile))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        9,
        // rules(),
    );
    // postprocess_egraph(&mut runner.egraph);
    println!("----------------------------------------------");
    measure_enode_proportions(&runner.egraph, true);
    let extractor = create_fine_grained_extractor(&runner.egraph);
    extractor.find_best(runner.roots[0]);
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

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_rmsnorm_cost3_kern2_wo_scheduler.json", 3, 2) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_rmsnorm_cost3_kern2_wo_scheduler.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/falcon7b_rmsnorm_cost3_kern2_wo_scheduler.txt").expect("Failed to create file");
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

#[test]
fn extract_qkvattn_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 4544]),
        ("WQ", vec![4544, 4544]),
        ("WK", vec![4544, 4544]),
        ("WV", vec![4544, 4544]),
        ("Q1", vec![16, 4544]),
        ("K1", vec![16, 4544]),
        ("V1", vec![16, 4544]),
        ("Q2", vec![16, 71, 64]),
        ("K2", vec![16, 71, 64]),
        ("V2", vec![16, 71, 64]),
        ("Q", vec![71, 16, 64]),
        ("K", vec![71, 16, 64]),
        ("V", vec![71, 16, 64]),
        ("K_cache", vec![71, 512+16, 64]),
        ("V_cache", vec![71, 528, 64]),
        ("C", vec![71, 16, 528]),
        ("C_exp", vec![71, 16, 528]),
        ("C_sum", vec![71, 16]),
        ("C_div", vec![71, 16, 528]),
        ("O", vec![71, 16, 64]),
        ("O1", vec![16, 71, 64]),
        ("O2", vec![16, 4544]),
    ]);
    
    let expr = "
(seq
    (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (fulltile))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 64 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4544 64 n
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
        9,
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

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_attacc_cost3_kern2_wo_scheduler.json", 3, 2) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_attacc_cost3_kern2_wo_scheduler.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/falcon7b_attacc_cost3_kern2_wo_scheduler.txt").expect("Failed to create file");
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

#[test]
fn extract_whole_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 4544]),
        ("X_norm", vec![16, 4544]),
        ("WQ", vec![4544, 4544]),
        ("WK", vec![4544, 4544]),
        ("WV", vec![4544, 4544]),
        ("Q1", vec![16, 4544]),
        ("K1", vec![16, 4544]),
        ("V1", vec![16, 4544]),
        ("Q2", vec![16, 71, 64]),
        ("K2", vec![16, 71, 64]),
        ("V2", vec![16, 71, 64]),
        ("Q", vec![71, 16, 64]),
        ("K", vec![71, 16, 64]),
        ("V", vec![71, 16, 64]),
        ("K_cache", vec![71, 512+16, 64]),
        ("V_cache", vec![71, 528, 64]),
        ("C", vec![71, 16, 528]),
        ("C_exp", vec![71, 16, 528]),
        ("C_sum", vec![71, 16]),
        ("C_div", vec![71, 16, 528]),
        ("O", vec![71, 16, 64]),
        ("O1", vec![16, 71, 64]),
        ("O2", vec![16, 4544]),
        ("X2", vec![16]),
    ]);
    
    let expr = "
(seq
    (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (fulltile))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 64 n
        (store (tensor Q2)
            (unsqueeze (load (tensor Q1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 4544 64 n
        (store (tensor K2)
            (unsqueeze (load (tensor K1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 4544 64 n
        (store (tensor V2)
            (unsqueeze (load (tensor V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor Q)
            (permute3
                (load (tensor Q2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor K)
            (permute3
                (load (tensor K2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor V)
            (permute3
                (load (tensor V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4544 64 n
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
)))))))))))))))))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        9,
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

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_rmsnorm_attacc_cost6_kern2_wo_scheduler.json", 6, 2) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_rmsnorm_attacc_cost6_kern2_wo_scheduler.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/falcon7b_rmsnorm_attacc_cost6_kern2_wo_scheduler.txt").expect("Failed to create file");
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

#[test]
fn extract_rmsnorm_qkv_attn_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 4544]),
        ("X_norm", vec![16, 4544]),
        ("WQ", vec![4544, 4544]),
        ("WK", vec![4544, 4544]),
        ("WV", vec![4544, 4544]),
        ("Q1", vec![16, 4544]),
        ("K1", vec![16, 4544]),
        ("V1", vec![16, 4544]),
        ("Q2", vec![16, 71, 64]),
        ("K2", vec![16, 71, 64]),
        ("V2", vec![16, 71, 64]),
        ("Q", vec![71, 16, 64]),
        ("K", vec![71, 16, 64]),
        ("V", vec![71, 16, 64]),
        ("K_cache", vec![71, 512+16, 64]),
        ("V_cache", vec![71, 528, 64]),
        ("C", vec![71, 16, 528]),
        ("C_exp", vec![71, 16, 528]),
        ("C_sum", vec![71, 16]),
        ("C_div", vec![71, 16, 528]),
        ("O", vec![71, 16, 64]),
        ("O1", vec![16, 71, 64]),
        ("O2", vec![16, 4544]),
        ("X2", vec![16]),
    ]);
    
    let expr = "
(seq
    (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (fulltile))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 64 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4544 64 n
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

    // match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_rmsnorm_qkv_attn_cost6_kern1_wo_scheduler2.json", 6, 1) {
    //     Ok(count) => println!("Saved {} expressions", count),
    //     Err(e) => eprintln!("Save error: {}", e),
    // }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_rmsnorm_qkv_attn_cost6_kern1_wo_scheduler2.json", usize::MAX) {
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

    let file = File::create("/home/jhpark676/Project/trinity/expressions/falcon7b_rmsnorm_qkv_attn_cost6_kern1_wo_scheduler3.txt").expect("Failed to create file");
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

#[test]
fn extract_ffn_expressions() {
    setup_shape_tracker(vec![
        ("O2", vec![16, 4544]),
        ("WO", vec![4544, 4544]),
        ("attn_O1", vec![16, 4544]),
        ("X", vec![16, 4544]),
        ("attn_O2", vec![16, 4544]),
        ("attn_O3", vec![16]),
        ("attn_O_norm", vec![16, 4544]),
        ("WFF1a", vec![4544, 4544]),
        ("WFF1b", vec![4544, 4544]),
        ("FF1a", vec![16, 4544]),
        ("FF1b", vec![16, 4544]),
        ("FF1b_silu", vec![16, 4544]),
        ("FF1", vec![16, 4544]),
        ("FF2", vec![16, 4544]),
        ("WFF2", vec![4544, 4544]),
        ("O_FF", vec![16, 4544]),
        ("O_FF1", vec![16]),
        ("O_FF_norm", vec![16, 4544]),
    ]);
    
    let expr = "
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_n n
        (store (tensor attn_O2)
            (+
                (load (tensor attn_O1) (index (fulltile) (tile n)))
                (load (input X) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_k k
        (store (tensor attn_O_norm)
            (/
                (load (tensor attn_O2) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor attn_O3) (index (fulltile)))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor FF1a)
                (+
                    (x (load (tensor FF1a) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1a) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor FF1b)
                (+
                    (x (load (tensor FF1b) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1b) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 tile_n n
        (store (tensor FF1b_silu)
            (x
                (load (tensor FF1b) (index (fulltile) (tile n)))
                (sigmoid
                    (load (tensor FF1b) (index (fulltile) (tile n)))
                )
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (store (tensor FF1)
            (x
                (load (tensor FF1a) (index (fulltile) (tile n)))
                (load (tensor FF1b_silu) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor FF2)
                (+
                    (x (load (tensor FF2) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor FF1) (index (fulltile) (tile k)))
                        (load (input WFF2) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 tile_n n
        (store (tensor O_FF)
            (+
                (load (tensor FF2) (index (fulltile) (tile n)))
                (load (tensor attn_O_norm) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4544 4544 k
        (store (tensor O_FF1)
            (rsum
                (sqr (load (tensor O_FF) (index (fulltile) (fulltile))))
                1
            )
            (index (fulltile))
        )
    )
    (loop 0 4544 4544 k
        (store (output O_FF_norm)
            (/
                (load (tensor O_FF) (index (fulltile) (fulltile)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor O_FF1) (index (fulltile)))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (fulltile))
        )
    )
)))))))))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
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

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_ffn_cost6_kern5_wo_scheduler2.json", 6, 5) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_ffn_cost6_kern5_wo_scheduler2.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/falcon7b_ffn_cost6_kern5_wo_scheduler2.txt").expect("Failed to create file");
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
            if !line.contains("dummydata") {
                writeln!(writer, "{}", line).expect("Failed to write to file");
            }
        });
    
    writer.flush().expect("Failed to flush writer");
}

#[test]
fn extract_gatedmlp_expressions() {
    setup_shape_tracker(vec![
        ("O2", vec![16, 4544]),
        ("WO", vec![4544, 4544]),
        ("attn_O1", vec![16, 4544]),
        ("X", vec![16, 4544]),
        ("attn_O2", vec![16, 4544]),
        ("attn_O3", vec![16]),
        ("attn_O_norm", vec![16, 4544]),
        ("WFF1a", vec![4544, 4544]),
        ("WFF1b", vec![4544, 4544]),
        ("FF1a", vec![16, 4544]),
        ("FF1b", vec![16, 4544]),
        ("FF1b_silu", vec![16, 4544]),
        ("FF1", vec![16, 4544]),
        ("FF2", vec![16, 4544]),
        ("WFF2", vec![4544, 4544]),
        ("O_FF", vec![16, 4544]),
        ("O_FF1", vec![16]),
        ("O_FF_norm", vec![16, 4544]),
    ]);
    
    let expr = "
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor FF1a)
                (+
                    (x (load (tensor FF1a) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1a) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor FF1b)
                (+
                    (x (load (tensor FF1b) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1b) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 tile_n n
        (store (tensor FF1b_silu)
            (x
                (load (tensor FF1b) (index (fulltile) (tile n)))
                (sigmoid
                    (load (tensor FF1b) (index (fulltile) (tile n)))
                )
            )
            (index (fulltile) (tile n))
        )
    )
    (loop 0 4544 tile_n n
        (store (output FF1)
            (x
                (load (tensor FF1a) (index (fulltile) (tile n)))
                (load (tensor FF1b_silu) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
)))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
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

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_gatedmlp_cost3_kern1_wo_scheduler.json", 3, 1) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/trinity/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/trinity/expressions/semi/falcon7b_gatedmlp_cost3_kern1_wo_scheduler.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/trinity/expressions/falcon7b_gatedmlp_cost3_kern1_wo_scheduler.txt").expect("Failed to create file");
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

egg::test_fn2!{test_rmsnorm_attacc_hacked, rules(),
"
(seq
    (loop 0 4544 tile_k k
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
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
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
    (loop 0 4544 64 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input K_cache)
            (load (tensor K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 4544 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h 
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
    (loop 0 71 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4544 64 n
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
)))))))))))))
"
=>
"
(loop 0 4544 64 n
    (seq
        (loop 0 4544 tile_k k
            (seq
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
                (store (tensor Q1,K1,V1)
                    (+
                        (x (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WQ,WK,WV) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
    (seq
        (loop 0 4544 tile_k k
            (store (tensor X_norm)
                (/
                    (load (input X) (index (fulltile) (tile k)))
                    (bcast
                        (sqrt
                            (/
                                (load (tensor X2) (index (fulltile)))
                                4544
                            )
                        )
                        1
                    )
                )
                (index (fulltile) (tile k))
            )
        )
    (seq
        (store (tensor Q1,K1,V1)
            (/
                (load (tensor Q1,K1,V1) (index (fulltile) (tile n)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile n))
        )
    (seq
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    (seq
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (elem n) (fulltile)))
                1 0 2
            )
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (store (input K_cache)
            (load (tensor K) (index (elem n) (fulltile) (fulltile)))
            (index (elem n) (const_tile 4544 16) (fulltile))
        )
    (seq
        (store (input V_cache)
            (load (tensor V) (index (elem n) (fulltile) (fulltile)))
            (index (elem n) (const_tile 4544 16) (fulltile))
        )
    (seq
        (loop 0 528 tile_p p
            (store (tensor C)
                (*
                    (load (tensor Q) (index (elem n) (fulltile) (fulltile)))
                    (permute3
                        (load (input K_cache) (index (elem n) (tile p) (fulltile)))
                        0 2 1
                    )
                )
                (index (elem n) (fulltile) (tile p))
            )
        )
    (seq
        (loop 0 528 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (elem n) (fulltile) (tile p))))
                (index (elem n) (fulltile) (tile p))
            )
        )
    (seq
        (loop 0 528 tile_p p
            (store (tensor C_sum)
                (+
                    (x (load (tensor C_sum) (index (elem n) (fulltile))) 1)
                    (rsum
                        (load (tensor C_exp) (index (elem n) (fulltile) (tile p)))
                        2
                    )
                )
                (index (elem n) (fulltile))
            )
        )
    (seq
        (loop 0 528 tile_p p
            (store (tensor C_div)
                (/
                    (load (tensor C_exp) (index (elem n) (fulltile) (tile p)))
                    (bcast
                        (load (tensor C_sum) (index (elem n) (fulltile)))
                        2
                    )
                )
                (index (elem n) (fulltile) (tile p))
            )
        )
    (seq
        (loop 0 528 tile_p p
            (store (tensor O)
                (+
                    (x (load (tensor O) (index (elem n) (fulltile) (fulltile))) 1)
                    (*
                        (load (tensor C_div) (index (elem n) (fulltile) (tile p)))
                        (load (input V_cache) (index (elem n) (tile p) (fulltile)))
                    )
                )
                (index (elem n) (fulltile) (fulltile))
            )
        )
    (seq
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (elem n) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (elem n) (fulltile))
        )
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )))))))))))))
)
"
,
"
    (loop 0 4544 64 n
        (seq
            (loop 0 4544 tile_k k
                (seq
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
                    (store (tensor Q1,K1,V1)
                        (+
                            (x (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                            (*
                                (load (input X) (index (fulltile) (tile k)))
                                (load (input WQ,WK,WV) (index (tile k) (tile n)))
                            )
                        )
                        (index (fulltile) (tile n))
                    )
                )
            )
        (seq
            (loop 0 4544 tile_k k
                (store (tensor X_norm)
                    (/
                        (load (input X) (index (fulltile) (tile k)))
                        (bcast
                            (sqrt
                                (/
                                    (load (tensor X2) (index (fulltile)))
                                    4544
                                )
                            )
                            1
                        )
                    )
                    (index (fulltile) (tile k))
                )
            )
        (seq
            (store (tensor Q1,K1,V1)
                (/
                    (load (tensor Q1,K1,V1) (index (fulltile) (tile n)))
                    (bcast
                        (sqrt
                            (/
                                (load (tensor X2) (index (fulltile)))
                                4544
                            )
                        )
                        1
                    )
                )
                (index (fulltile) (tile n))
            )
        (seq
            (store (tensor Q2,K2,V2)
                (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                (index (fulltile) (elem n) (fulltile))
            )
        (seq
            (store (tensor Q,K,V)
                (permute3
                    (load (tensor Q2,K2,V2) (index (fulltile) (elem n) (fulltile)))
                    1 0 2
                )
                (index (elem n) (fulltile) (fulltile))
            )
        (seq
            (store (input K_cache)
                (load (tensor K) (index (elem n) (fulltile) (fulltile)))
                (index (elem n) (const_tile 4544 16) (fulltile))
            )
        (seq
            (store (input V_cache)
                (load (tensor V) (index (elem n) (fulltile) (fulltile)))
                (index (elem n) (const_tile 4544 16) (fulltile))
            )
        (seq
            (loop 0 528 tile_p p
                (seq
                    (store (tensor C)
                        (*
                            (load (tensor Q) (index (elem n) (fulltile) (fulltile)))
                            (permute3
                                (load (input K_cache) (index (elem n) (tile p) (fulltile)))
                                0 2 1
                            )
                        )
                        (index (elem n) (fulltile) (tile p))
                    )
                (seq
                    (store (tensor C_exp)
                        (exp (load (tensor C) (index (elem n) (fulltile) (tile p))))
                        (index (elem n) (fulltile) (tile p))
                    )
                (seq
                    (store (tensor C_sum)
                        (+
                            (x (load (tensor C_sum) (index (elem n) (fulltile))) 1)
                            (rsum
                                (load (tensor C_exp) (index (elem n) (fulltile) (tile p)))
                                2
                            )
                        )
                        (index (elem n) (fulltile))
                    )
                    (store (tensor O)
                        (+
                            (x (load (tensor O) (index (elem n) (fulltile) (fulltile))) 1)
                            (*
                                (load (tensor C_exp) (index (elem n) (fulltile) (tile p)))
                                (load (input V_cache) (index (elem n) (tile p) (fulltile)))
                            )
                        )
                        (index (elem n) (fulltile) (fulltile))
                    )
                )
                )
                )
            )
        (seq
            (loop 0 528 tile_p p
                (store (tensor C_div)
                    (/
                        (load (tensor C_exp) (index (elem n) (fulltile) (tile p)))
                        (bcast
                            (load (tensor C_sum) (index (elem n) (fulltile)))
                            2
                        )
                    )
                    (index (elem n) (fulltile) (tile p))
                )
            )
        (seq
            (store (tensor O)
                (/
                    (load (tensor O) (index (elem n) (fulltile) (fulltile)))
                    (bcast
                        (load (tensor C_sum) (index (elem n) (fulltile)))
                        2
                    )
                )
                (index (elem n) (fulltile) (fulltile))
            )
        (seq
            (store (tensor O1)
                (permute3
                    (load (tensor O) (index (elem n) (fulltile) (fulltile)))
                    1 0 2
                )
                (index (fulltile) (elem n) (fulltile))
            )
            (store (output O2)
                (squeeze
                    (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                    1
                )
                (index (fulltile) (tile n))
            )
        )))))))))))
    )
"
}