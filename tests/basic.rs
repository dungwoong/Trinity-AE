use egg::{test_fn2, test_fn_not2};
use TileIR::*;

egg::test_fn2! {test_sequence, rules(),
    "(seq (seq a (seq b c)) (seq d (seq e f)))"
    =>
    "(seq a (seq b (seq c (seq d (seq e f)))))",
}
egg::test_fn2! {fusible1, rules(),
    /*
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n]
    for n in (N, tile_n):
        C[n:n+tile_n] = B[n:n+tile_n]
    =>
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n]
        C[n:n+tile_n] = B[n:n+tile_n]
    */
    "(seq 
        (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n))) (index (tile n))))
        (loop 0 N tile_n n (store (output C) (load (input B) (index (tile n))) (index (tile n))))
    )"
    =>
    "
    (loop 0 N tile_n n
        (seq 
            (store (input B) (load (input A) (index (tile n))) (index (tile n)))
            (store (output C) (load (input B) (index (tile n))) (index (tile n)))
        )
    )
    "
    ,
    // "
    // (loop 0 N tile_n n
    //     (seq 
    //         (dummy)  
    //         (store (output C) (load (input A) (index (tile n))) (index (tile n)))
    //     )
    // )
    // "
}

egg::test_fn2! {fusible2, rules(),
    /*
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n, m:m+tile_m] = A[n:n+tile_n, m:m+tile_m] + 3
    for n in (N, tile_n):
        for m in (M, tile_m):
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n, m:m+tile_m] + 2
    =>
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n, m:m+tile_m] = A[n:n+tile_n, m:m+tile_m] + 3
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n, m:m+tile_m] + 2
    */
    "
    (seq
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile n) (tile m)))
                        3
                    )
                    (index (tile n) (tile m))
                )
            )    
        )

        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n) (tile m)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )    
        )
    )
    "
    =>
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m 
            (seq
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile n) (tile m)))
                        3
                    )
                    (index (tile n) (tile m))
                )
                (store (output C) 
                    (+ 
                        (load (input A) (index (tile n) (tile m)))
                        (+ 2 3)
                    )
                    (index (tile n) (tile m))
                )
            )
        )    
    )
    "
}
egg::test_fn2! {fusible3, rules(),
    /*
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = B[n:n+tile_n] + A[n:n+tile_n, m:m+tile_m]
    for n in (N, tile_n):
        for m in (M, tile_m):
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    =>
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = B[n:n+tile_n] + A[n:n+tile_n, m:m+tile_m]
        
        for m in (M, tile_m):
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    */
    "(seq
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (input B) 
                    (+ 
                        (load (input B) (index (tile n)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n))
                )
            )    
        )

        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )    
        )
    )"
    =>
    "
    (loop 0 N tile_n n
        (seq
            (loop 0 M tile_m m 
                (store (input B) 
                    (+ 
                        (load (input B) (index (tile n)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n))
                )
            ) 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )
        )
    )
    "
}
egg::test_fn_not2! {not_fusible1, rules(),
    /*
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = B[n:n+tile_n] + A[n:n+tile_n, m:m+tile_m]
    for n in (N, tile_n):
        for m in (M, tile_m):
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    !=
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = B[n:n+tile_n] + A[n:n+tile_n, m:m+tile_m]
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    */
    "(seq
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (input B) 
                    (+ 
                        (load (input B) (index (tile n)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n))
                )
            )    
        )
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )    
        )
    )"
    != 
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m 
            (seq
                (store (input B) 
                    (+ 
                        (load (input B) (index (tile n)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n))
                )
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )
        )    
    )
    "
}
egg::test_fn2! {fusible4, rules(),
    /*
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = A[n:n+tile_n] + 2
    for n in (N, tile_n):
        for m in (M, tile_m):
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    =>
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = A[n:n+tile_n] + 2
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    */
    "(seq
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile n)))
                        2
                    )
                    (index (tile n))
                )
            )
        )
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )    
        )
    )"
    =>
    // "(seq
    //     (loop 0 N tile_n n 
    //         (dummy)
    //     )
    //     (loop 0 N tile_n n 
    //         (loop 0 M tile_m m 
    //             (store (output C) 
    //                 (+ 
    //                     (load (input B) (index (tile n)))
    //                     2
    //                 )
    //                 (index (tile n) (tile m))
    //             )
    //         )    
    //     )
    // )"
    // ,
    // "
    // (loop N tile_n n
    //     (seq
    //         (store (input B) 
    //             (+ 
    //                 (load (input A) (index (tile n)))
    //                 2
    //             )
    //             (index (tile n))
    //         )
    //         (loop M tile_m m 
    //             (store (input C) 
    //                 (+ 
    //                     (load (input B) (index (tile n)))
    //                     2
    //                 )
    //                 (index (tile n) (tile m))
    //             )
    //         )    
    //     )
    // )
    // "
    // ,
    // "
    // (loop N tile_n n
    //     (loop M tile_m m
    //         (seq
    //             (store (input B) 
    //                 (+ 
    //                     (load (input A) (index (tile n)))
    //                     2
    //                 )
    //                 (index (tile n))
    //             )
    //             (store (input C) 
    //                 (+ 
    //                     (load (input B) (index (tile n)))
    //                     2
    //                 )
    //                 (index (tile n) (tile m))
    //             )
    //         )
    //     )    
    // )
    // "
    // ,
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m 
            (seq
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile n)))
                        2
                    )
                    (index (tile n))
                )
                (store (output C) 
                    (+ 
                        (+ (load (input A) (index (tile n))) 2)
                        2
                    )
                    (index (tile n) (tile m))
                )
            )
        )    
    )
    "
}

egg::test_fn2! {fusible5, rules(),
    /*
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n] + 2
    for n in (N, tile_n):
        for m in (M, tile_m):
            C[n:n+tile_n, m:m+tile_m] = B[n:n+tile_n] + 2
    =>
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = A[n:n+tile_n] + 2
            C[n:n+tile_n, m:m+tile_m] = A[n:n+tile_n] + 2 + 2
    */
    "(seq
        (loop 0 N tile_n n 
            (store (input B) 
                (+ 
                    2
                    (load (input A) (index (tile n)))
                )
                (index (tile n))
            )
        )
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n)))
                        2
                    )
                    (index (tile n) (tile m))
                )
            )    
        )
    )"
    => 
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m 
            (seq
                (store (input B) 
                    (+ 
                        2
                        (load (input A) (index (tile n)))
                    )
                    (index (tile n))
                )
                (store (output C) 
                    (+ 
                        (load (input A) (index (tile n)))
                        (+ 2 2)
                    )
                    (index (tile n) (tile m))
                )
            )
        )    
    )
    "
}
egg::test_fn2! {loop_deletion, rules(),
    /*
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = A[n:n+tile_n] + 2
    =>
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n] + 2
    */
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m 
            (store (output B) 
                (+ 
                    2
                    (load (input A) (index (tile n)))
                )
                (index (tile n))
            )
        )    
    )
    "
    => 
    "
    (loop 0 N tile_n n 
        (store (output B) 
            (+ 
                2
                (load (input A) (index (tile n)))
            )
            (index (tile n))
        )
    )
    "
}
egg::test_fn_not2! {no_loop_deletion, rules(),
    /*
    for n in (N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = B[n:n+tile_n] + A[n:n+tile_n, m:m+tile_m]
    !=
    for n in (N, tile_n):
        B[n:n+tile_n] = B[n:n+tile_n] + A[n:n+tile_n, m:m+tile_m]
    */
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m 
            (store (output B) 
                (+ 
                    (load (output B) (index (tile n)))
                    (load (input A) (index (tile n) (tile m)))
                )
                (index (tile n))
            )
        )    
    )
    "
    !=
    "
    (loop 0 N tile_n n 
        (store (output B) 
            (+ 
                (load (output B) (index (tile n)))
                (load (input A) (index (tile n) (tile m)))
            )
            (index (tile n))
        )
    )
    "
}
egg::test_fn2! {seq_fusion, rules(),
    /*
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n]
    for n in (N, tile_n):
        C[n:n+tile_n] = B[n:n+tile_n]
    for n in (N, tile_n):
        D[n:n+tile_n] = C[n:n+tile_n]
    =>
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n]
        C[n:n+tile_n] = B[n:n+tile_n]
        D[n:n+tile_n] = C[n:n+tile_n]
    */
    "
    (seq
        (seq 
            (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n))) (index (tile n))))
            (loop 0 N tile_n n (store (input C) (load (input B) (index (tile n))) (index (tile n))))
        )
        (loop 0 N tile_n n (store (output D) (load (input C) (index (tile n))) (index (tile n))))
    )
    "
    =>
    
    "
    (loop 0 N tile_n n 
        (seq
            (store (input B) (load (input A) (index (tile n))) (index (tile n)))
            (seq
                (store (input C) (load (input B) (index (tile n))) (index (tile n)))
                (store (output D) (load (input A) (index (tile n))) (index (tile n)))
            )
        )
    )
    "
}
egg::test_fn2! {seq_comm1, rules(),
    "
    (seq
        (store (output B) (load (input A) (index (tile n))) (index (tile n)))
        (store (output C) (load (input A) (index (tile n))) (index (tile n)))
    )
    "
    =>
    "
    (seq
        (store (output C) (load (input A) (index (tile n))) (index (tile n)))
        (store (output B) (load (input A) (index (tile n))) (index (tile n)))
    )
    "
}
egg::test_fn2! {seq_comm2, rules(),
    "
    (seq
        (store (output B) (load (input A) (index (tile n))) (index (tile n)))
        (seq
            (store (output C) (load (input A) (index (tile n))) (index (tile n)))
            (store (output D) (load (input A) (index (tile n))) (index (tile n)))
        )
    )
    "
    =>
    "
    (seq
        (store (output D) (load (input A) (index (tile n))) (index (tile n)))
        (seq
            (store (output B) (load (input A) (index (tile n))) (index (tile n)))
            (store (output C) (load (input A) (index (tile n))) (index (tile n)))
        )
    )
    "
}
egg::test_fn_not2! {no_seq_comm2, rules(),
    "
    (seq
        (store (output B) (+ (load (output B) (index)) 2) (index))
    (seq
        (store (output C) (* (load (output B) (index)) 3) (index))
        done
    ))
    "
    !=
    "
    (seq
        (store (output C) (* (load (output B) (index)) 3) (index))
    (seq
        (store (output B) (+ (load (output B) (index)) 2) (index))
        done
    ))
    "

}
egg::test_fn2! {fusible6, rules(),
    "
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (input C1) 
                        (+
                            (load (input C1) (index (tile m) (tile n)))
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
                (store (output C1_exp) 
                    (exp
                        (load (input C1) (index (tile m) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
            )
        )
        body
    )
    )
    "
    =>
    "
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (seq
                    (loop 0 K tile_k k 
                        (store (input C1) 
                            (+
                                (load (input C1) (index (tile m) (tile n)))
                                (*
                                    (load (input X) (index (tile m) (tile k)))
                                    (load (input W1) (index (tile k) (tile n)))
                                )
                            )
                         (index (tile m) (tile n))
                        )
                    )
                    (store (output C1_exp) 
                        (exp
                            (load (input C1) (index (tile m) (tile n)))
                        )
                        (index (tile m) (tile n))
                    )
                )
            )
        )
        body
    )
    "
}
egg::test_fn_not2! {loop_insertion_fuse1, rules(),
    "
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (input C1) 
                        (+
                            (load (input C1) (index (tile m) (tile n)))
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
                (store (output C1_exp) 
                    (exp
                        (load (input C1) (index (tile m) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
            )
        )
        body
    )
    )
    "
    !=
    "
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (seq
                        (store (input C1) 
                            (+
                                (load (input C1) (index (tile m) (tile n)))
                                (*
                                    (load (input X) (index (tile m) (tile k)))
                                    (load (input W1) (index (tile k) (tile n)))
                                )
                            )
                            (index (tile m) (tile n))
                        )
                        (store (output C1_exp) 
                            (exp
                                (load (input C1) (index (tile m) (tile n)))
                            )
                            (index (tile m) (tile n))
                        )
                    )
                )
                
            )
        )
        body
    )
    "
}
egg::test_fn2! {sequence_fuse1, rules(),  
    "
    (seq 
        (loop 0 N tile_n n (store (output B) (load (input A) (index (tile n))) (index (tile n))))
    (seq
        (loop 0 N tile_n n (store (output C) (load (output B) (index (tile n))) (index (tile n))))
    (seq
        (loop 0 N tile_n n (store (output D) (load (output C) (index (tile n))) (index (tile n))))
        (loop 0 N tile_n n (store (output E) (load (output D) (index (tile n))) (index (tile n))))
    )
    )
    )"
    =>
    "
    (loop 0 N tile_n n
        (seq 
            (store (output B) (load (input A) (index (tile n))) (index (tile n)))
        (seq
            (store (output C) (load (input A) (index (tile n))) (index (tile n)))
        (seq
            (store (output D) (load (input A) (index (tile n))) (index (tile n)))
            (store (output E) (load (input A) (index (tile n))) (index (tile n)))
        )
        )
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n
            (seq
                (store (output B) (load (input A) (index (tile n))) (index (tile n)))
                (store (output C) (load (input A) (index (tile n))) (index (tile n)))
            )
        )
        (loop 0 N tile_n n
            (seq
                (store (output D) (load (input A) (index (tile n))) (index (tile n)))
                (store (output E) (load (input A) (index (tile n))) (index (tile n)))
            )
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n (store (output B) (load (input A) (index (tile n))) (index (tile n))))
    (seq
        (loop 0 N tile_n n
            (seq
                (store (output C) (load (input A) (index (tile n))) (index (tile n)))
                (store (output D) (load (input A) (index (tile n))) (index (tile n)))
            )
        )
        (loop 0 N tile_n n (store (output E) (load (input A) (index (tile n))) (index (tile n))))
    )
    )
    "
}
egg::test_fn2! {loop_insertion_fuse2, rules(),
    /*
    for n in (N, tile_n):
        B[n:n+tile_n] = A[n:n+tile_n]
    for m in (M, tile_m):
        D[m:m+tile_m] = C[m:m+tile_m]
    =>
    for n in(N, tile_n):
        for m in (M, tile_m):
            B[n:n+tile_n] = A[n:n+tile_n]
            D[m:m+tile_m] = C[m:m+tile_m]
    */
    "
    (seq
        (loop 0 N tile_n n (store (output B) (load (input A) (index (tile n))) (index (tile n))))
        (loop 0 M tile_m m (store (output D) (load (input C) (index (tile m))) (index (tile m))))
    )
    "
    =>
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m
            (seq
                (store (output B) (load (input A) (index (tile n))) (index (tile n)))
                (store (output D) (load (input C) (index (tile m))) (index (tile m)))
            )
        
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (loop 0 M tile_m m
                (store (output B) (load (input A) (index (tile n))) (index (tile n)))            
            )
        )
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (output D) (load (input C) (index (tile m))) (index (tile m)))
            )
        )
    )
    "
}

egg::test_fn2! {loop_insertion_fuse3, rules(),
    /*
    for n in (N, tile_n):
        B = B + A[n:n+tile_n]
    for m in (M, tile_n):
        D[m:m+tile_m] = E * C[m:m+tile_m]
    =>
    for m in (M, tile_m):
        for n in (N, tile_n):
            B = B + A[n:n+tile_n]
            D[m:m+tile_m] = E * C[m:m+tile_m]
    */
    "
    (seq
        (loop 0 N tile_n n
            (store (output B) (+ (load (output B) (index)) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 M tile_m m
            (store (output D) (* (load (input E) (index)) (load (input C) (index (tile m)))) (index (tile m)))
        )
    )
    "
    =>
    "
    (loop 0 M tile_m m
        (loop 0 N tile_n n
            (seq
                (store (output B) (+ (load (output B) (index)) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input E) (index)) (load (input C) (index (tile m)))) (index (tile m)))
            )
        )
    )
    "
}
egg::test_fn2! {value_forward_substitute1, rules(),
    "
    (seq
        (store (output B) (load (input A) (index)) (index))
    (seq
        (store (output C) (load (output B) (index)) (index))
        done
    )
    )
    "
    =>
    "
    (seq
        (store (output B) (load (input A) (index)) (index))
    (seq
        (store (output C) (load (input A) (index)) (index))
        done
    )
    )
    "
}
egg::test_fn2! {value_forward_substitute2, rules(),
    "
    (seq
        (store (output B) (+ (load (input A) (index (tile n))) 2) (index (tile n)))
        (store (output C) (* (load (output B) (index (tile n))) 3) (index (tile n)))
    )
    "
    =>
    "
    (seq
        (store (output B) (+ (load (input A) (index (tile n))) 2) (index (tile n)))
        (store (output C) (* (+ (load (input A) (index (tile n))) 2) 3) (index (tile n)))
    )
    "
    ,
    "
    (seq
        (store (output B) (+ (load (input A) (index (tile n))) 2) (index (tile n)))
        (store (output C) (* (+ (load (input A) (index (tile n))) 2) 3) (index (tile n)))
    )
    "
}
egg::test_fn2! {loop_dist1, rules(),
    /*
    for n in (N, tile_n):
        C[n] = A[n] + 1
        B = Bx3 + A[n]
        D[n] = A[n] * C[n]
    E = B x 10
    =>
    for n in (N, tile_n):
        C[n] = A[n] + 1
        E = Ex3 + A[n] x 10
        D[n] = A[n] * (A[n] + 1)
    */
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input A) (index (tile n))) (load (output C) (index (tile n)))) (index (tile n)))
            )
            )
        )
        (store (output E) (x (load (output B) (index)) 10) (index))
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
            )
        )
        (store (output E) (x (load (output B) (index)) 10) (index))
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
            )
        )
        (store (output E) (x (load (output B) (index)) 10) (index))
    )
    "
    ,
    "
    (seq
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
        )
    )
        (store (output E) (x (load (output B) (index)) 10) (index))
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
        )
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
        )
        (store (output E) (x (load (output B) (index)) 10) (index))
    ))
    "
    ,
    "
    (loop 0 N tile_n n 
        (seq
            (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
        (seq
            (store (output E) (+ (x (load (output E) (index)) 3) (x 10 (load (input A) (index (tile n))))) (index))
            (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
        )
        )
    )
    "
}

egg::test_fn2! {loop_factor1, rules(),
    "
    (loop 0 N tile_n n 
        (seq
            (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
        (seq
            (store (output B) (+ (x (load (output B) (index)) 3) (x 10 (load (input A) (index (tile n))))) (index))
            (store (output D) (* (load (input A) (index (tile n))) (load (output C) (index (tile n)))) (index (tile n)))
        )
        )
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
            )
        )
        (store (output B) (x (load (output B) (index)) 10) (index))
    )
    "
}

egg::test_fn2! {loop_factor2, rules(),
    /*
    for n in N:
        for m in M:
            B[n] = B[n] + reduce_sum(A[n,m])
    for n in N:
        for m in M:
            D[n,m] = C[n,m] / bcast(B[n])
            E[n,:] = E[n,:] + D[n, m] * F[m, :]
    =>
    for n in N:
        for m in M:
            B[n] = B[n] + reduce_sum(A[n,m])
            E[n,:] = E[n,:] + C[n, m] * F[m, :]
        E[n,:] = E[n,:] / bcast(B[n])

        for m in M:
            D[n,m] = C[n,m] / bcast(B[n])
    */
    "
    (seq
        (loop 0 N tile_n n
            (loop 0 M tile_m m
                (store (input B) 
                    (+
                        (x (load (input B) (index (tile n))) 1)
                        (rsum (load (input A) (index (tile n) (tile m))) 1)
                    )
                    (index (tile n))
                )
            )
        )
        (loop 0 N tile_n n
            (loop 0 M tile_m m
                (seq
                    (store (input D)
                        (/
                            (load (input C) (index (tile n) (tile m)))
                            (bcast (load (input B) (index (tile n))) 1)
                        )
                        (index (tile n) (tile m))
                    )
                    (store (input E)
                        (+
                            (x (load (input E) (index (tile n) (fulltile))) 1)
                            (*
                                (load (input D) (index (tile n) (tile m)))
                                (load (input F) (index (tile m) (fulltile)))
                            )
                        )
                        (index (tile n) (fulltile))
                    )
                )
            )
        )
    )
    "
    =>
    "
    (loop 0 N tile_n n
        (seq
            (loop 0 M tile_m m
                (seq
                    (store (input B) 
                        (+
                            (x (load (input B) (index (tile n))) 1)
                            (rsum (load (input A) (index (tile n) (tile m))) 1)
                        )
                        (index (tile n))
                    )
                    (store (input E)
                        (+
                            (x (load (input E) (index (tile n) (fulltile))) 1)
                            (*
                                (load (input C) (index (tile n) (tile m)))
                                (load (input F) (index (tile m) (fulltile)))
                            )
                        )
                        (index (tile n) (fulltile))
                    )
                )
            )
        (seq
            (store (input E)
                (/
                    (load (input E) (index (tile n) (fulltile)))
                    (bcast (load (input B) (index (tile n))) 1)
                )
                (index (tile n) (fulltile))
            )
            (loop 0 M tile_m m
                (store (input D)
                    (/
                        (load (input C) (index (tile n) (tile m)))
                        (bcast (load (input B) (index (tile n))) 1)
                    )
                    (index (tile n) (tile m))
                )
            )
        )
        )
    )
    "
    ,
    "
    (loop 0 N tile_n n
        (seq
            (loop 0 M tile_m m
                (seq
                    (store (input B) 
                        (+
                            (x (load (input B) (index (tile n))) 1)
                            (rsum (load (input A) (index (tile n) (tile m))) 1)
                        )
                        (index (tile n))
                    )
                    (store (input E)
                        (+
                            (x (load (input E) (index (tile n) (fulltile))) 1)
                            (*
                                (load (input C) (index (tile n) (tile m)))
                                (load (input F) (index (tile m) (fulltile)))
                            )
                        )
                        (index (tile n) (fulltile))
                    )
                )
            )
        (seq
            (loop 0 M tile_m m
                (store (input D)
                    (/
                        (load (input C) (index (tile n) (tile m)))
                        (bcast (load (input B) (index (tile n))) 1)
                    )
                    (index (tile n) (tile m))
                )
            )
            (store (input E)
                (/
                    (load (input E) (index (tile n) (fulltile)))
                    (bcast (load (input B) (index (tile n))) 1)
                )
                (index (tile n) (fulltile))
            )
        )
        )
    )
    "
}

egg::test_fn2! {loop_split1, rules(),
    "
    (seq
        (loop 0 N tile_n n
            (store (input B) (+ (x (load (input B) (index)) 1) (x 10 (rsum (load (input A) (index (tile n))) 1))) (index))
        )
        (store (input D) (exp (load (input C) (index))) (index))
    )
    "
    =>
    "
    (seq
        (dloop 0 N new_tile_n new_n
            (dloop new_n (+ new_n new_tile_n) tile_n n
                (store (input new_B)
                    (+
                        (x (load (input new_B) (index (elem new_n) (index))) 1)
                        (x 10 (rsum (load (input A) (index (tile n))) 1))
                    )
                    (index (elem new_n) (index))
                )
            )
        )
    (seq
        (store (input B) (rsum (load (input new_B) (index (fulltile) (index))) 0) (index))
        (store (input D) (exp (load (input C) (index))) (index))
    )
    )
    "
    ,


}
egg::test_fn2! {forward_and_fission1, rules(),
    /*
    for n in (N, tile_n)
        B[n] = A[n] x 2
        D = D + B[n] * C[n]
    =>
    for n in (N, tile_n)
        B[n] = A[n] x 2
        D = D + (A[n] x 2) * C[n]
    =>
    for n in (N, tile_n)
        B[n] = A[n] x 2
    for n in (N, tile_n)
        D = D + (A[n] x 2) * C[n]
    =>
    for n in (N, tile_n)
        D = D + (A[n] x 2) * C[n]
    for n in (N, tile_n)
        B[n] = A[n] x 2
    */
    "
    (loop 0 N tile_n n 
        (seq
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (load (output B) (index (tile n))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
    )
    "
    =>
    "
    (loop 0 N tile_n n 
        (seq
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
    )
    "
    // ,
    // "
    // (seq
    //     (loop 0 N tile_n n 
    //         (store (output D) 
    //             (+ (x (load (output D) (index)) 1)
    //                 (* (x 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
    //             )
    //             (index)
    //         )
    //     )
    //     (loop 0 N tile_n n 
    //         (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
    //     )
    // )
    // "
}
egg::test_fn2! {seq_comm3, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
    )
    "
}
egg::test_fn2! {forward_and_dist1, rules(),
    /*
    for n in (N, tile_n)
        B[n] = A[n] x 2
        D = D + B[n] x C[n]
    =>
    for n in (N, tile_n)
        B[n] = A[n] x 2
        D = D + (A[n] x 2) x C[n]
    =>
    for n in (N, tile_n)
        B[n] = A[n] x 2
        D = D + A[n] x C[n]
    D = D x 2
    */
    "
    (loop 0 N tile_n n 
        (seq
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (load (output B) (index (tile n))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
    )
    "
    =>
    "
    (loop 0 N tile_n n 
        (seq
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (x 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
    )
    "
    ,
    "
    (loop 0 N tile_n n 
        (seq
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (x (load (input A) (index (tile n))) (load (input C) (index (tile n)))) 2)
                )
                (index)
            )
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (x (load (input A) (index (tile n))) (load (input C) (index (tile n)))) 2)
                )
                (index)
            )
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (seq
            (loop 0 N tile_n n 
                (store (output D) 
                    (+ (x (load (output D) (index)) 1)
                        (x (load (input A) (index (tile n))) (load (input C) (index (tile n))))
                    )
                    (index)
                )
            )
            (store (output D) (x (load (output D) (index)) 2) (index))
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output B) (x 2 (load (input A) (index (tile n)))) (index (tile n)))
                (store (output D) 
                    (+ (x (load (output D) (index)) 1)
                        (x (load (input A) (index (tile n))) (load (input C) (index (tile n))))
                    )
                    (index)
                )
            )
        )
        (store (output D) (x (load (output D) (index)) 2) (index))
    )
    "
}
egg::test_fn2! {dist_and_fusion1, rules(),
    /*
    for n in (N, tile_n):
        B = B + A[n]
    for n in (N, tile_n):
        D = D + C[n] / B
    =>
    for n in (N, tile_n):
        B = B + A[n]
    for n in (N, tile_n):
        D = D + C[n]
    D = D / B
    =>
    for n in (N, tile_n):
        B = B + A[n]
        D = D + C[n]
    D = D / B
    */
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 N tile_n n
            (store (output D) (+ (x (load (output D) (index)) 1) (/ (load (input C) (index (tile n))) (load (output B) (index)))) (index))
        )
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
        )
    (seq
        (loop 0 N tile_n n
            (store (output D) (+ (x (load (output D) (index)) 1) (load (input C) (index (tile n)))) (index))
        )
        (store (output D) (/ (load (output D) (index)) (load (output B) (index))) (index))
    )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
                (store (output D) (+ (x (load (output D) (index)) 1) (load (input C) (index (tile n)))) (index))
            )
        )
        (store (output D) (/ (load (output D) (index)) (load (output B) (index))) (index))
    )
    "
}
egg::test_fn_not2! {not_fusible2, rules(),
    /*
    for n in (N, tile_n):
        B = B + A[n]
    for n in (N, tile_n):
        D = D + C[n] / B
    !=
    for n in (N, tile_n):
        B = B + A[n]
        D = D + C[n] / B
    */
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 N tile_n n
            (store (output D) (+ (x (load (output D) (index)) 1) (/ (load (input C) (index (tile n))) (load (output B) (index)))) (index))
        )
    )
    "
    !=
    "
    (loop 0 N tile_n n
        (seq
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
            (store (output D) (+ (x (load (output D) (index)) 1) (/ (load (input C) (index (tile n))) (load (output B) (index)))) (index))
        )
    )
    "
}

egg::test_fn2! {equivalent, rules(),
    "(seq 
        (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n))) (index (tile n))))
        (loop 0 N tile_n n (store (output C) (load (input B) (index (tile n))) (index (tile n))))
    )"
    =>
    "(seq 
        (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n))) (index (tile n))))
        (loop 0 N tile_n n (store (output C) (load (input A) (index (tile n))) (index (tile n))))
    )"
}

egg::test_fn2! {loop_split2, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C1) 
                    (+
                        (x (load (input C1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input W1) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
        (store (input A) (+ (load (input B) (index)) 1) (index))
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C1) 
                    (+
                        (x (load (input C1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input W1) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
        (store (input A) (+ (load (input B) (index)) 1) (index))
    )
    "
}

egg::test_fn2! {const_loop_fusion_same1, rules(),
    "
    (seq
        (loop 0 N 128 n 
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
        )
        (loop 0 N tile_n n 
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
    =>
    "
    (loop 0 N 128 n 
        (seq
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
}

egg::test_fn_not2! {no_const_loop_fusion_same1, rules(),
    "
    (seq
        (loop 0 N 128 n 
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
        )
        (loop 0 N tile_n n 
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
    !=
    "
    (loop 0 N tile_n n 
        (seq
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
}
egg::test_fn2! {const_loop_fusion_same2, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
        )
        (loop 0 N 128 n 
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
    =>
    "
    (loop 0 N 128 n 
        (seq
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
}
egg::test_fn_not2! {no_const_loop_fusion_same2, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
        )
        (loop 0 N 128 n 
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
    !=
    "
    (loop 0 N tile_n n 
        (seq
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
}
egg::test_fn2! {const_loop_fusion_diff1, rules(),
    /*
    for m in (2048, tile_m):
        for n in (1024, 256):
            Load A[m:m+tile_m, n:n+D]
            Compute A[m:m+tile_m, n:n+D]
            Store B[m:m+tile_m, n//D, 0:D]
    for m in (2048, tile_m):
        for h in (4, tile_h):
            Load B[m:m+tile_m, h:h+tile_h, 0:D]
            Compute B[m:m+tile_m, h:h+tile_h, 0:D] x 10
            Store C[m:m+tile_m, h:h+tile_h, 0:D]
    =>
    for m in (2048, tile_m):
        for n in (1024, 256):
            Load A[m:m+tile_m, n:n+D]
            Compute A[m:m+tile_m, n:n+D]
            Store B[m:m+tile_m, n//D, 0:D]

            Load B[m:m+tile_m, n//D, 0:D]
            Compute B[m:m+tile_m, n//D, 0:D] x 10
            Store C[m:m+tile_m, n//D, 0:D]
    */
    "
    (seq
        (loop 0 2048 tile_m m
            (loop 0 1024 256 n
                (store (input B)
                    (load (input A) (index (tile m) (tile n)))
                    (index (tile m) (elem n) (fulltile))
                )
            )
        )
        (loop 0 2048 tile_m m
            (loop 0 4 tile_h h
                (store (input C)
                    (x
                        (load (input B) (index (tile m) (tile h) (fulltile)))
                        10
                    )
                    (index (tile m) (tile h) (fulltile))
                )
            )
        )
    )
    "
    =>
    "
    (loop 0 2048 tile_m m
        (loop 0 1024 256 n
            (seq
                (store (input B)
                    (load (input A) (index (tile m) (tile n)))
                    (index (tile m) (elem n) (fulltile))
                )
                (store (input C)
                    (x
                        (load (input B) (index (tile m) (elem n) (fulltile)))
                        10
                    )
                    (index (tile m) (elem n) (fulltile))
                )
            )
        )
    )
    "
}

egg::test_fn2! {const_loop_fusion_same3, rules(),
    "
    (seq
        (loop 0 N 128 n 
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
        )
        (loop 0 N 128 n 
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
    =>
    "
    (loop 0 N 128 n 
        (seq
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
    )
    "
}

egg::test_fn2! {const_loop_fusion_same4, rules(),
    "
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
        (loop 0 2048 64 n
            (store (input V2)
                (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
                (index (fulltile) (elem n) (fulltile))
            )
        )
    )))))
    "
    =>
    "
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

            (store (input V2)
                (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
                (index (fulltile) (elem n) (fulltile))
            )
        )        
        )
        )
    )
    "
}
egg::test_fn2! {const_loop_fusion_same5, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (store (input B)
                (+ (load (input A) (index (tile n))) 10)
                (index (tile n))
            )
        )
    (seq
        (loop 0 N 128 n 
            (store (input C)
                (+ (load (input B) (index (tile n))) 10)
                (index (tile n))
            )
        )
        others
    )
    )
    "
    =>
    "
    (seq
        (loop 0 N 128 n 
            (seq
                (store (input B)
                    (+ (load (input A) (index (tile n))) 10)
                    (index (tile n))
                )
                (store (input C)
                    (+ (load (input B) (index (tile n))) 10)
                    (index (tile n))
                )
            )
        )
        others
    )
    "
}
egg::test_fn2! {const_loop_fusion_same6, rules(),
"
(seq
    (loop 0 2048 tile_n n
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
    )
(seq
    (loop 0 2048 64 n
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

            (store (input V2)
                (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
                (index (fulltile) (elem n) (fulltile))
            )
        )        
        )
    )
    others
))
"
=>
"
(seq
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

            (store (input V2)
                (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
                (index (fulltile) (elem n) (fulltile))
            )
        )        
        )
        )
    )
    others
)
"
}

egg::test_fn_not2!{multi_input1, rules(),
"
(seq
    (loop 0 N tile_n n
        (store (tensor A,B,C)
            (+
                (load (tensor a,b,c) (index))
                3
            )
            (index)
        )
    )
    (loop 0 N tile_n n
        (store (tensor D)
            (load (tensor B) (index))
            (index)
        )
    )
)
"
!=
"
    (loop 0 N tile_n n
        (seq
            (store (tensor D)
                (load (tensor B) (index))
                (index)
            )
            (store (tensor A,B,C)
                (+
                    (load (tensor a,b,c) (index))
                    3
                )
                (index)
            )
        )
    )
"
}

egg::test_fn_not2!{not_fusible3, rules(),
"
(seq
    (loop 0 4544 tile_n n
        (store (tensor attn_O_norm)
            (/
                (load (tensor attn_O2) (index (fulltile) (tile n)))
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
            (index (fulltile) (tile n))
        )
    )
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
)
"
!=
"
    (loop 0 4544 tile_n n
        (seq
            (store (tensor attn_O_norm)
                (/
                    (load (tensor attn_O2) (index (fulltile) (tile n)))
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
                (index (fulltile) (tile n))
            )
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
    )

"

}

egg::test_fn2!{loop_insertion_fuse4, rules(),
"
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
)
"
=>
"
    
    (loop 0 4544 tile_n n
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
    )

"
,
"
    
    (loop 0 4544 tile_n n
            (loop 0 4544 tile_k k
                (seq
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
    )

"

}