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

static INIT_ATTN_ONLY: Once = Once::new();

// Custom rules function that sets up shapes before returning rules
fn rules_with_shapes_attn_only() -> Vec<egg::Rewrite<TileLang, LoopAnalysis>> {
    INIT_ATTN_ONLY.call_once(|| {
        setup_shape_tracker(vec![
            ("Q", vec![32, 16, 64]),
            ("K_cache", vec![32, 2064, 64]),
            ("C", vec![32, 16, 2064]),
            ("C_exp", vec![32, 16, 2064]),
            ("C_sum", vec![32, 16]),
            ("C_div", vec![32, 16, 2064]),
            ("V_cache", vec![32, 2064, 64]),
            ("O", vec![32, 16, 64]),
        ]);
    });
    rules()
}

#[test]
fn extract_attn_only() {
    setup_shape_tracker(vec![
        ("Q", vec![32, 16, 64]),
        ("K_cache", vec![32, 2064, 64]),
        ("C", vec![32, 16, 2064]),
        ("C_exp", vec![32, 16, 2064]),
        ("C_sum", vec![32, 16]),
        ("C_div", vec![32, 16, 2064]),
        ("V_cache", vec![32, 2064, 64]),
        ("O", vec![32, 16, 64]),
    ]);
    let expr = "
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C)
                (*
                    (load (input Q) (index (tile h) (fulltile) (fulltile)))
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
        (loop 0 2064 tile_p p
            (store (input C_exp)
                (exp (load (input C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_sum)
                (+
                    (x (load (input C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_div)
                (/
                    (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (input C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input O)
                (+
                    (x (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
                    (*
                        (load (input C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
        // rules(),
    );
    // postprocess_egraph(&mut runner.egraph);
    
    println!("----------------------------------------------");
    // measure_enode_proportions(&runner.egraph, true);

    // List all expressions for the root e-class
    // println!("All equivalent expressions for your input:");

    // let total_expressions = list_expressions_all(&runner);
    // println!("[All] There are {:?} expressions", total_expressions.len());

    // let num_total_expressions = count_expressions_all_for_root(&runner);
    // println!("[All] There are {:?} expressions", num_total_expressions);

    // let cost_expressions = list_expressions_with_target_cost(&runner);
    // println!("[Cost model] There are {:?} expressions", cost_expressions.len());

    // let cost_expressions_v2 = list_expressions_with_target_cost_v2(&runner);
    // println!("[Cost model v2] There are {:?} expressions", cost_expressions_v2.len());

    // let cost_expressions_v3 = list_expressions_with_target_cost_v3(&runner);
    // println!("[Cost model v3] There are {:?} expressions", cost_expressions_v3.len());

    // Save expressions
    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attn_only.json", 2, 2) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attn_only.json") {
    // let expressions = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attn_only.json", usize::MAX) {
    //     Ok(expressions) => {
    //         expressions
    //     },
    //     Err(e) => {
    //         eprintln!("Load error: {}", e);
    //         vec![]
    //     },
    // };

    // let num_cost_expressions = count_expressions_num_sloop(&runner);
    // let percentage_x10000 = (&num_cost_expressions * 10000u32) / &num_total_expressions;
    // println!("[Cost model] There are {:?} expressions ({}.{:02}%)", 
    //         num_cost_expressions, 
    //         &percentage_x10000 / 100u32, 
    //         &percentage_x10000 % 100u32);
    
    // let root_expressions = list_expressions_num_kernel(&runner, 0.1);
    // println!("[Num Kernel] There are {:?} expressions", root_expressions.len());

    // let total_expressions = count_expressions_num_kernel_for_root(&runner, 0.0);
    // println!("[Num Kernel] There are {:?} expressions", total_expressions);

    // let extractor = create_extractor(&runner.egraph);
    // let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    // println!("Cost {}: {}", cost, best_expr);

    // let file = File::create("expressions/attn_only.txt").expect("Failed to create file");
    // let mut writer = BufWriter::new(file);
    
    // expressions
    //     .par_iter()
    //     .enumerate()
    //     .map(|(i, expr)| {
    //         let new_expr = postprocess(expr);
    //         format!("{}: {}", i, new_expr)  // Convert to String here
    //     })
    //     .collect::<Vec<String>>()  // Now collecting Vec<String>
    //     .iter()
    //     .for_each(|line| {
    //         writeln!(writer, "{}", line).expect("Failed to write to file");
    //     });
    // writer.flush().expect("Failed to flush writer");
}

#[test]
fn extract_attn_concat_only() {
        let expr = "
(seq
    (loop 0 32 tile_h h
        (store (input K_cache)
            (load (input K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (input V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C)
                (*
                    (load (input Q) (index (tile h) (fulltile) (fulltile)))
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
        (loop 0 2064 tile_p p
            (store (input C_exp)
                (exp (load (input C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_sum)
                (+
                    (x (load (input C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_div)
                (/
                    (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (input C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input O)
                (+
                    (x (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
                    (*
                        (load (input C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
))))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
        // rules(),
    );

    
    // postprocess_egraph(&mut runner.egraph);
    println!("----------------------------------------------");
    // measure_enode_proportions(&runner.egraph, true);
    // List all expressions for the root e-class
    // println!("All equivalent expressions for your input:");
    // let root_expressions = list_expressions_all(&runner);
    // println!("[All] There are {:?} expressions", root_expressions.len());

    let total_expressions = count_expressions_all_for_root(&runner);
    println!("[All] There are {:?} expressions", total_expressions);

    let cost_expressions = list_expressions_with_target_cost(&runner);
    println!("[Cost model] There are {:?} expressions", cost_expressions.len());
    
    let cost_expressions_v2 = list_expressions_with_target_cost_v2(&runner);
    println!("[Cost model v2] There are {:?} expressions", cost_expressions_v2.len());


    // let cost_expressions = count_expressions_num_sloop(&runner);
    // let percentage_x10000 = (&cost_expressions * 10000u32) / &total_expressions;
    // println!("[Cost model] There are {:?} expressions ({}.{:02}%)", 
    //         cost_expressions, 
    //         &percentage_x10000 / 100u32, 
    //         &percentage_x10000 % 100u32);

    // let root_expressions = list_expressions_num_kernel(&runner, 0.1);
    // println!("[Num Kernel] There are {:?} expressions", root_expressions.len());

    // let total_expressions = count_expressions_num_kernel_for_root(&runner, 0.0);
    // println!("[Num Kernel] There are {:?} expressions", total_expressions);

    // let extractor = create_extractor(&runner.egraph);
    // let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    // println!("Cost {}: {}", cost, best_expr);
}

#[test]
fn extract_attn_concat_permute_only() {
        let expr = "
(seq
    (loop 0 32 tile_h h
        (store (input Q)
            (permute3
                (load (input Q2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K)
            (permute3
                (load (input K2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V)
            (permute3
                (load (input V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K_cache)
            (load (input K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (input V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C)
                (*
                    (load (input Q) (index (tile h) (fulltile) (fulltile)))
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
        (loop 0 2064 tile_p p
            (store (input C_exp)
                (exp (load (input C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_sum)
                (+
                    (x (load (input C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_div)
                (/
                    (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (input C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input O)
                (+
                    (x (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
                    (*
                        (load (input C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
)))))))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
        // rules(),
    );
    // postprocess_egraph(&mut runner.egraph);
    println!("----------------------------------------------");
    // measure_enode_proportions(&runner.egraph, true);
    // List all expressions for the root e-class
    // println!("All equivalent expressions for your input:");
    // let root_expressions = list_expressions_all(&runner);
    // println!("[All] There are {:?} expressions", root_expressions.len());

    let total_expressions = count_expressions_all_for_root(&runner);
    println!("[All] There are {:?} expressions", total_expressions);

    let cost_expressions = list_expressions_with_target_cost(&runner);
    println!("[Cost model] There are {:?} expressions", cost_expressions.len());

    let cost_expressions_v2 = list_expressions_with_target_cost_v2(&runner);
    println!("[Cost model v2] There are {:?} expressions", cost_expressions_v2.len());


    // let cost_expressions = count_expressions_num_sloop(&runner);
    // let percentage_x10000 = (&cost_expressions * 10000u32) / &total_expressions;
    // println!("[Cost model] There are {:?} expressions ({}.{:02}%)", 
    //         cost_expressions, 
    //         &percentage_x10000 / 100u32, 
    //         &percentage_x10000 % 100u32);
    // let root_expressions = list_expressions_num_kernel(&runner, 0.1);
    // println!("[Num Kernel] There are {:?} expressions", root_expressions.len());

    // let total_expressions = count_expressions_num_kernel_for_root(&runner, 0.0);
    // println!("[Num Kernel] There are {:?} expressions", total_expressions);

    // let extractor = create_extractor(&runner.egraph);
    // let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    // println!("Cost {}: {}", cost, best_expr);
}

#[test]
fn extract_attn_concat_permute_unsqueeze_only() {
        let expr = "
(seq
    (loop 0 2048 64 n
        (store (input Q2)
            (unsqueeze (load (input Q1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 2048 64 n
        (store (input K2)
            (unsqueeze (load (input K1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 2048 64 n
        (store (input V2)
            (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input Q)
            (permute3
                (load (input Q2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K)
            (permute3
                (load (input K2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V)
            (permute3
                (load (input V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K_cache)
            (load (input K) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (input V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C)
                (*
                    (load (input Q) (index (tile h) (fulltile) (fulltile)))
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
        (loop 0 2064 tile_p p
            (store (input C_exp)
                (exp (load (input C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_sum)
                (+
                    (x (load (input C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input C_div)
                (/
                    (load (input C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (input C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
            (store (input O)
                (+
                    (x (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
                    (*
                        (load (input C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
))))))))))))
";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
        // rules(),
    );
    // postprocess_egraph(&mut runner.egraph);
    println!("----------------------------------------------");
    // measure_enode_proportions(&runner.egraph, true);
    // List all expressions for the root e-class
    // println!("All equivalent expressions for your input:");
    // let root_expressions = list_expressions_all(&runner);
    // println!("[All] There are {:?} expressions", root_expressions.len());

    // let total_expressions = count_expressions_all_for_root(&runner);
    // println!("[All] There are {:?} expressions", total_expressions);

    // let cost_expressions = list_expressions_with_target_cost(&runner);
    // println!("[Cost model] There are {:?} expressions", cost_expressions.len());

    let cost_expressions_v2 = list_expressions_with_target_cost_v2(&runner);
    println!("[Cost model v2] There are {:?} expressions", cost_expressions_v2.len());

    // let cost_expressions = count_expressions_num_sloop(&runner);
    // let percentage_x10000 = (&cost_expressions * 10000u32) / &total_expressions;
    // println!("[Cost model] There are {:?} expressions ({}.{:02}%)", 
    //         cost_expressions, 
    //         &percentage_x10000 / 100u32, 
    //         &percentage_x10000 % 100u32);

    // let root_expressions = list_expressions_num_kernel(&runner, 0.1);
    // println!("[Num Kernel] There are {:?} expressions", root_expressions.len());

    // let total_expressions = count_expressions_num_kernel_for_root(&runner, 0.0);
    // println!("[Num Kernel] There are {:?} expressions", total_expressions);

    // let extractor = create_extractor(&runner.egraph);
    // let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    // println!("Cost {}: {}", cost, best_expr);
}

#[test]
fn extract_lora() {
    let expr = "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (tensor C)
                    (+ (x (load (tensor C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (tensor D)
                (+ (x (load (tensor D) (index (fulltile) (fulltile))) 1)
                (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))
                ))
                (index (fulltile) (fulltile))
            )
        )

    (seq
        (loop 0 P tile_p p 
            (store (tensor E)
                (* (load (tensor D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))
                )
                (index (fulltile) (tile p))
            )
        )

        (loop 0 P tile_p p
            (store (output O)
                (+ (load (tensor C) (index (fulltile) (tile p))) (load (tensor E) (index (fulltile) (tile p))))
                (index (fulltile) (tile p))
            )
        )   
    )
    )
    )
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        40,
        // rules(),
    );

    // match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/lora_cost3_kern2.json") {
    //     Ok(count) => println!("Saved {} expressions", count),
    //     Err(e) => eprintln!("Save error: {}", e),
    // }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/lora_cost3_kern2.json") {
    let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/lora_cost3_kern2.json", 0) {
        Ok(expressions) => {
            println!("Loaded {} final expressions", expressions.len());
            expressions
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/TileIR/expressions/lora_cost3_kern2_index0.txt").expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess(expr);
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
fn extract_gated_mlp_skip_ft() {
    let expr = "
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (tensor C1) 
                    (+
                        (x (load (tensor C1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input W1) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
        
    (seq
        (loop 0 N tile_n n 
            (store (tensor C1_exp) 
                (exp
                    (load (tensor C1) (index (fulltile) (tile n)))
                )
                (index (fulltile) (tile n))
            )
        )
    
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (tensor C2) 
                    (+
                        (x (load (tensor C2) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input W2) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
        (loop 0 N tile_n n 
            (store (output O) 
                (x
                    (load (tensor C1_exp) (index (fulltile) (tile n)))
                    (load (tensor C2) (index (fulltile) (tile n)))
                )
                (index (fulltile) (tile n))
            )
        )
    )
    )
    )
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        40,
        // rules(),
    );

    // match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/gated_mlp_skipft_cost3_kern2.json") {
    //     Ok(count) => println!("Saved {} expressions", count),
    //     Err(e) => eprintln!("Save error: {}", e),
    // }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/gated_mlp_skipft_cost3_kern2.json") {
    let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/gated_mlp_skipft_cost3_kern2.json", 17) {
        Ok(expressions) => {
            println!("Loaded {} final expressions", expressions.len());
            expressions
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/TileIR/expressions/gated_mlp_skipft_cost3_kern2_index17.txt").expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess(expr);
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
fn extract_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![16, 2048]),
        ("WQ", vec![2048, 2048]),
        ("WK", vec![2048, 2048]),
        ("WV", vec![2048, 2048]),
        ("Q1", vec![16, 2048]),
        ("K1", vec![16, 2048]),
        ("V1", vec![16, 2048]),
        ("Q2", vec![16, 32, 64]),
        ("K2", vec![16, 32, 64]),
        ("V2", vec![16, 32, 64]),
        ("Q", vec![32, 16, 64]),
        ("K", vec![32, 16, 64]),
        ("V", vec![32, 16, 64]),
        ("K_cache", vec![32, 2064, 64]),
        ("V_cache", vec![32, 2064, 64]),
        ("C", vec![32, 16, 2064]),
        ("C_exp", vec![32, 16, 2064]),
        ("C_sum", vec![32, 16]),
        ("C_div", vec![32, 16, 2064]),
        ("O", vec![32, 16, 64]),
        ("O1", vec![16, 32, 64]),
        ("O2", vec![16, 2048]),

    ]);
    
    let expr = "
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
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
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
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
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
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
    (loop 0 2048 64 n
        (store (tensor Q2)
            (unsqueeze (load (tensor Q1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 2048 64 n
        (store (tensor K2)
            (unsqueeze (load (tensor K1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 2048 64 n
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
            (index (tile h) (const_tile 2048 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input V_cache)
            (load (tensor V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 2048 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
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
        (loop 0 2064 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 2064 tile_p p
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
        (loop 0 2064 tile_p p
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
        (loop 0 2064 tile_p p
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
    (loop 0 2048 64 n
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

    // match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 3, 2) {
    //     Ok(count) => println!("Saved {} expressions", count),
    //     Err(e) => eprintln!("Save error: {}", e),
    // }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/TileIR/expressions/attacc_cost3_kern2_with_cost.txt").expect("Failed to create file");
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
fn extract_rms_norm() {
    let expr = "
(seq
    (loop 0 4096 tile_k k
        (seq
            (store (tensor X1)
                (sqr (load (input X) (index (fulltile) (tile k))))
                (index (fulltile) (tile k))
            )
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
    )    
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (seq
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
            (seq
                (store (tensor Q1)
                    (+
                        (x (load (tensor Q1) (index (fulltile) (tile n))) 1)
                        (*
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
                            (load (input WQ) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            (seq
                (store (tensor K1)
                    (+
                        (x (load (tensor K1) (index (fulltile) (tile n))) 1)
                        (*
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
                            (load (input WK) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
                (store (tensor V1)
                    (+
                        (x (load (tensor V1) (index (fulltile) (tile n))) 1)
                        (*
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
                            (load (input WV) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
            )
            )
        )
    )
)
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

    match list_expressions_with_target_cost_v3_part1(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/rmsnorm_cost3_kern1.json", 3, 1) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }
    
    // Load expressions
    // let expressions = match list_expressions_with_target_cost_v3_part2(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json") {
    // let expressions = match list_expressions_from_semi_all(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 27) {
    // let expressions = match list_expressions_from_semi_naive(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/attacc_cost3_kern2.json", 27) {
    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, "/home/jhpark676/Project/TileIR/expressions/semi/rmsnorm_cost3_kern1.json", usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create("/home/jhpark676/Project/TileIR/expressions/rmsnorm_cost3_kern1.txt").expect("Failed to create file");
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
fn postprocess_expression() {
    let expr = "
    (loop 0 M tile_m m
        (seq
            (loop 0 N tile_n n
                (seq
                    (store (input C) 
                        (*
                            (load (input Q) (index (tile m) (fulltile)))
                            (load (input K) (index (fulltile) (tile n)))
                        )
                        (index (tile m) (tile n))
                    )
                (seq
                    (store (input C_exp)
                        (exp 
                            (load (input C) (index (tile m) (tile n)))
                        )
                        (index (tile m) (tile n))
                    )
                (seq
                    (store (input C_sum)
                        (+
                            (x (load (input C_sum) (index (tile m))) 1)
                            (rsum 
                                (load (input C_exp) (index (tile m) (tile n)))
                                1
                            )
                        )
                        (index (tile m))
                    )
                    (store (output O)
                        (+
                            (x (load (output O) (index (tile m) (fulltile))) 1)
                            (*
                                (load (input C_exp) (index (tile m) (tile n)))
                                (load (input V) (index (tile n) (fulltile)))
                            )
                        )
                        (index (tile m) (fulltile))
                    )
                )
                )
                )
            )
        (seq
            (store (output O)
                (/
                    (load (output O) (index (tile m) (fulltile)))
                    (bcast (load (input C_sum) (index (tile m))) 1)
                )
                (index (tile m) (fulltile))
            )
            (loop 0 N tile_n n
                (store (input C_div)
                    (/
                        (load (input C_exp) (index (tile m) (tile n)))
                        (bcast (load (input C_sum) (index (tile m))) 1)
                    )
                    (index (tile m) (tile n))
                )
            )
        )
        )
    )
    ";
    // let expr = "(loop 0 N tile_n n
    //     (seq
    //         (store (input B) (+ 1 3) (index))
    //         (store (output C) (* (load (input B) (index)) 3) (index))
    //     )
    // )";

    let expr = "
(loop 0 2048 64 n
    (seq
        (loop 0 2048 tile_k k
            (seq
                (store (input Q1)
                    (+
                        (x (load (input Q1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WQ) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            (seq
                (store (input K1)
                    (+
                        (x (load (input K1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WK) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )

                (store (input V1)
                    (+
                        (x (load (input V1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WV) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            
            )
            )
        )
    (seq
        (store (input Q2)
            (unsqueeze (load (input Q1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    (seq
        (store (input K2)
            (unsqueeze (load (input K1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    (seq
        (store (input V2)
            (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )        
    (seq
        (store (input Q)
            (permute3
                (load (input Q2) (index (fulltile) (elem n) (fulltile)))
                1 0 2
            )
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (store (input K)
            (permute3
                (load (input K2) (index (fulltile) (elem n) (fulltile)))
                1 0 2
            )
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (store (input V)
            (permute3
                (load (input V2) (index (fulltile) (elem n) (fulltile)))
                1 0 2
            )
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (store (input K_cache)
            (load (input K) (index (elem n) (fulltile) (fulltile)))
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (store (input V_cache)
            (load (input V) (index (elem n) (fulltile) (fulltile)))
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (loop 0 2064 tile_p p
            (seq
                (store (input C)
                    (*
                        (load (input Q) (index (elem n) (fulltile) (fulltile)))
                        (permute3
                            (load (input K_cache) (index (elem n) (tile p) (fulltile)))
                            0 2 1
                        )
                    )
                    (index (elem n) (fulltile) (tile p))
                )
            (seq
                (store (input C_exp)
                    (exp (load (input C) (index (elem n) (fulltile) (tile p))))
                    (index (elem n) (fulltile) (tile p))
                )
            (seq
                (store (input C_sum)
                    (+
                        (x (load (input C_sum) (index (elem n) (fulltile))) 1)
                        (rsum
                            (load (input C_exp) (index (elem n) (fulltile) (tile p)))
                            2
                        )
                    )
                    (index (elem n) (fulltile))
                )
                (store (input O)
                    (+
                        (x (load (input O) (index (elem n) (fulltile) (fulltile))) 1)
                        (*
                            (load (input C_exp) (index (elem n) (fulltile) (tile p)))
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
        (store (input O)
            (/
                (load (input O) (index (elem n) (fulltile) (fulltile)))
                (bcast
                    (load (input C_sum) (index (elem n) (fulltile)))
                    2
                )
            )
            (index (elem n) (fulltile) (fulltile))
        )
    (seq
        (loop 0 2064 tile_p p
            (store (input C_div)
                (/
                    (load (input C_exp) (index (elem n) (fulltile) (tile p)))
                    (bcast
                        (load (input C_sum) (index (elem n) (fulltile)))
                        2
                    )
                )
                (index (elem n) (fulltile) (tile p))
            )
        )
    (seq
        (store (input O1)
            (permute3
                (load (input O) (index (elem n) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (elem n) (fulltile))
        )
        (store (input O2)
            (squeeze
                (permute3
                    (load (input O) (index (elem n) (fulltile) (fulltile)))
                    1 0 2
                )
                1
            )
            (index (fulltile) (tile n))
        )
    )))))))))))))
)
";
let expr = "(loop 0 2048 64 n (seq (loop 0 2048 tile_k k (seq (store (tensor K1) (+ (x 1 (load (tensor K1) (index fulltile (tile n)))) (* (load (input X) (index fulltile (tile k))) (load (input WK) (index (tile k) (tile n))))) (index fulltile (tile n))) (seq (store (tensor V1) (+ (x 1 (load (tensor V1) (index fulltile (tile n)))) (* (load (input X) (index fulltile (tile k))) (load (input WV) (index (tile k) (tile n))))) (index fulltile (tile n))) (store (tensor Q1) (+ (x (load (tensor Q1) (index fulltile (tile n))) 1) (* (load (input X) (index fulltile (tile k))) (load (input WQ) (index (tile k) (tile n))))) (index fulltile (tile n)))))) (seq (store (tensor Q2) (unsqueeze (load (tensor Q1) (index fulltile (tile n))) 1) (index fulltile (tile h) fulltile)) (seq (store (tensor K2) (unsqueeze (load (tensor K1) (index fulltile (tile n))) 1) (index fulltile (tile h) fulltile)) (seq (store (tensor V2) (unsqueeze (load (tensor V1) (index fulltile (tile n))) 1) (index fulltile (tile h) fulltile)) (seq (store (tensor Q) (permute3 (load (tensor Q2) (index fulltile (tile h) fulltile)) 1 0 2) (index (tile h) fulltile fulltile)) (seq (store (tensor K) (permute3 (load (tensor K2) (index fulltile (tile h) fulltile)) 1 0 2) (index (tile h) fulltile fulltile)) (seq (store (tensor V) (permute3 (load (tensor V2) (index fulltile (tile h) fulltile)) 1 0 2) (index (tile h) fulltile fulltile)) (seq (store (input K_cache) (permute3 (load (tensor K2) (index fulltile (tile h) fulltile)) 1 0 2) (index (tile h) fulltile fulltile)) (seq (store (input V_cache) (permute3 (load (tensor V2) (index fulltile (tile h) fulltile)) 1 0 2) (index (tile h) fulltile fulltile)) (seq (loop 0 2064 tile_p p (seq (store (tensor C) (* (load (tensor Q) (index (tile h) fulltile fulltile)) (permute3 (load (input K_cache) (index (tile h) (tile p) fulltile)) 0 2 1)) (index (tile h) fulltile (tile p))) (seq (store (tensor C_exp) (exp (load (tensor C) (index (tile h) fulltile (tile p)))) (index (tile h) fulltile (tile p))) (seq (store (tensor C_sum) (+ (x 1 (load (tensor C_sum) (index (tile h) fulltile))) (rsum (load (tensor C_exp) (index (tile h) fulltile (tile p))) 2)) (index (tile h) fulltile)) (store (tensor O) (+ (x 1 (load (tensor O) (index (tile h) fulltile fulltile))) (* (load (tensor C_exp) (index (tile h) fulltile (tile p))) (load (input V_cache) (index (tile h) (tile p) fulltile)))) (index (tile h) fulltile fulltile)))))) (seq (loop 0 2064 tile_p p (store (tensor C_div) (/ (load (tensor C_exp) (index (tile h) fulltile (tile p))) (bcast (load (tensor C_sum) (index (tile h) fulltile)) 2)) (index (tile h) fulltile (tile p)))) (seq (store (tensor O) (/ (load (tensor O) (index (tile h) fulltile fulltile)) (bcast (load (tensor C_sum) (index (tile h) fulltile)) 2)) (index (tile h) fulltile fulltile)) (seq (store (tensor O1) (permute3 (load (tensor O) (index (tile h) fulltile fulltile)) 1 0 2) (index fulltile (tile h) fulltile)) (store (output O2) (squeeze (load (tensor O1) (index fulltile (tile h) fulltile)) 1) (index fulltile (tile n)))))))))))))))))";

    let new_expr = postprocess(expr);
    println!("{}", new_expr);
}

#[test]
fn visualizer() {
    let expr = "
    (seq
        (dloop 0 1024 tile_n n body1)
    (seq
        (dloop 0 1024 tile_n n body2)
        (dloop 0 512 tile_n n body3)
    )
    )
    ";
    let runner = run_until_saturated(
        expr,
        custom_rules(),
        10,
    );

    save_egraph(&runner, "egraph.dot");
}
