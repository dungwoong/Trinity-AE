use egg::{test_fn2, test_fn_not2, *};
use TileIR::*;


#[test]
fn saturate_gated_mlp_skip_ft() {
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
    let runner = run_until_saturated(
        expr,
        rules(),
    );
    // let runner = run_until_saturated(
    //     &generate_nested_seq(1, "z"),
    //     rules(),
    // );
}

#[test]
fn saturate_gated_mlp() {
    let expr = "
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (tensor C1) 
                        (+
                            (x (load (tensor C1) (index (tile m) (tile n))) 1)
                            (*
                                (load (input X) (index (tile m) (tile k)))
                                (load (input W1) (index (tile k) (tile n)))
                            )
                        )
                     (index (tile m) (tile n))
                    )
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (tensor C1_exp) 
                    (exp
                        (load (tensor C1) (index (tile m) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (input C2) 
                        (+
                            (x (load (input C2) (index (tile m) (tile n))) 1)
                            (*
                                (load (input X) (index (tile m) (tile k)))
                                (load (input W2) (index (tile k) (tile n)))
                            )
                        )
                     (index (tile m) (tile n))
                    )
                )
            )
        )
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (output O) 
                    (x
                        (load (input C1_exp) (index (tile m) (tile n)))
                        (load (input C2) (index (tile m) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    )
    )
    )
    ";
    let runner = run_until_saturated(
        expr,
        rules(),
    );
    // let check_expr: RecExpr<TileLang> = "(load (input C1_exp) (index (tile m) (tile n)))".parse().unwrap();

    // let found = contains_expr(&runner.egraph, runner.roots[0], &check_expr);
    // println!("{:?} => {:?}", check_expr, found);

}

#[test]
fn saturate_lora() {
    let expr = "
    (seq
        (loop 0 M tile_m m
            (loop 0 P tile_p p 
                (loop 0 N tile_n n
                    (store (input C)
                        (+ (x (load (input C) (index (tile m) (tile p))) 1)
                        (* (load (input X) (index (tile m) (tile n))) (load (input W) (index (tile n) (tile p)))
                        ))
                        (index (tile m) (tile p))
                    )
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 K tile_k k 
                (loop 0 N tile_n n
                    (store (input D)
                        (+ (x (load (input D) (index (tile m) (tile k))) 1)
                        (* (load (input X) (index (tile m) (tile n))) (load (input A) (index (tile n) (tile k)))
                        ))
                        (index (tile m) (tile k))
                    )
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 P tile_p p 
                (loop 0 K tile_k k
                    (store (input E)
                        (+ (x (load (input E) (index (tile m) (tile p))) 1)
                        (* (load (input D) (index (tile m) (tile k))) (load (input B) (index (tile k) (tile p)))
                        ))
                        (index (tile m) (tile p))
                    )
                )
            )
        )

        (loop 0 M tile_m m
            (loop 0 P tile_p p
                (store (output O)
                    (+ (load (input C) (index (tile m) (tile p))) (load (input E) (index (tile m) (tile p))))
                    (index (tile m) (tile p))
                )
            )
        )
    )
    )
    )
    ";
    let runner = run_until_saturated(
        expr,
        rules(),
    );
}

#[test]
fn saturate_lora_skip_ft() {
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
    let runner = run_until_saturated(
        expr,
        rules(),
    );

    println!("All equivalent expressions for your input:");
    let root_expressions = list_expressions_for_root(&runner);
    println!("There are {:?} expressions", root_expressions.len());

    // root_expressions
    //     .par_iter()
    //     .enumerate()
    //     .for_each(|(i, expr)| {
    //         let new_expr = postprocess(expr);
    //         // println!("  {}: {}", i, new_expr);
    //     });
}

#[test]
fn saturate_attacc_skip_ft() {
    let expr = "
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
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
        )
    )
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
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
        )
    )
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
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

)))))))))))))))
";
    let mut runner = run_until_saturated(
        expr,
        rules(),
    );
    postprocess_egraph(&mut runner.egraph);

    // List all expressions for the root e-class
    println!("All equivalent expressions for your input:");
    let root_expressions = list_expressions_for_root(&runner);
    println!("There are {:?} expressions", root_expressions.len());

    // save_egraph(&runner, "egraph.dot");
}

#[test]
fn saturate_flashattn2_skip_ft() {
    let expr = "
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C) 
                    (*
                        (load (input Q) (index (tile m) (fulltile)))
                        (load (input K) (index (fulltile) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_exp)
                    (exp (load (input C) (index (tile m) (tile n))))
                    (index (tile m) (tile n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_sum)
                    (+
                        (x (load (input C_sum) (index (tile m))) 1)
                        (rsum (load (input C_exp) (index (tile m) (tile n))) 1)
                    )
                    (index (tile m))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
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
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (output O)
                    (+
                        (x (load (output O) (index (tile m) (fulltile))) 1)
                        (*
                            (load (input C_div) (index (tile m) (tile n)))
                            (load (input V) (index (tile n) (fulltile)))
                        )
                    )
                    (index (tile m) (fulltile))
                )
            )
        )
    ))))
    ";
    // let expr = "(seq 
    //     (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n))) (index (tile n))))
    //     (loop 0 N tile_n n (store (output C) (load (input B) (index (tile n))) (index (tile n))))
    // )";
    // let expr = "
    // (seq
    //     (loop 0 P tile_p p 
    //         (loop 0 N tile_n n
    //             (store (input C)
    //                 (+ (x (load (input C) (index (fulltile) (tile p))) 1)
    //                 (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
    //                 ))
    //                 (index (fulltile) (tile p))
    //             )
    //         )
    //     )
    // (seq
    //     (loop 0 N tile_n n
    //         (store (input D)
    //             (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
    //             (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))
    //             ))
    //             (index (fulltile) (fulltile))
    //         )
    //     )

    // (seq
    //     (loop 0 P tile_p p 
    //         (store (input E)
    //             (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))
    //             )
    //             (index (fulltile) (tile p))
    //         )
    //     )

    //     (loop 0 P tile_p p
    //         (store (output O)
    //             (+ (load (input C) (index (fulltile) (tile p))) (load (input E) (index (fulltile) (tile p))))
    //             (index (fulltile) (tile p))
    //         )
    //     )   
    // )
    // )
    // )
    // ";
    // let expr = "(seq (seq a b) (seq c (seq d (seq (seq e (seq f g)) h))))";
    // let expr = "
    // (seq
    //     (loop 0 N tile_n n
    //         (store (input B) (+ (x (load (input B) (index)) 1) (x 10 (rsum (load (input A) (index (tile n))) 1))) (index))
    //     )
    // (seq
    //     (loop 0 N tile_n n
    //         (store (input B) (+ (x (load (input B) (index)) 1) (x 10 (rsum (load (input A) (index (tile n))) 1))) (index))
    //     )
    //     (store (input B) (x (load (input B) (index)) 20) (index))
    // )
    // )
    // ";

    // let new_expr = postprocess(expr);
    // println!("{}", new_expr);

    // let mut runner = run_until_saturated(
    //     expr,
    //     rules(),
    // );
    // // postprocess_egraph(&mut runner.egraph);

    // // List all expressions for the root e-class
    // println!("All equivalent expressions for your input:");
    // let root_expressions = list_expressions_for_root(&runner);
    // println!("There are {:?} expressions", root_expressions.len());
    
    // save_egraph(&runner, "egraph.dot");

    // // let egraph = &runner.egraph;
    // // let root_id = runner.roots[0];
    // // let extractor = Extractor::new(egraph, AstSize);
    // // let (best_cost, best_expr) = extractor.find_best(root_id);

    // let file = File::create("expressions/attention_skip_ft.txt").expect("Failed to create file");
    // let mut writer = BufWriter::new(file);
    
    // root_expressions
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
    // println!("Expressions written to expressions.txt");
    
    // for (i, expr) in root_expressions.iter().enumerate() {
    //     let new_expr = postprocess(expr);
    //     // let new_expr = expr;
    //     // println!("  {}: {}", i, new_expr);
    // }
}