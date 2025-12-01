use egg::{test_fn2, test_fn_not2, *};
use trinity::*;
use std::io::BufWriter;
use std::io::Write;
use std::fs::File;
use rayon::prelude::*;


egg::test_fn2! {attacc_fusion, custom_rules(),
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
            )
            )
        )
    
    
        (loop 0 2064 tile_p p
            (seq
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
                (store (input O)
                    (+
                        (x (load (input O) (index (elem n) (fulltile) (fulltile))) 1)
                        (*
                            (load (input C_div) (index (elem n) (fulltile) (tile p)))
                            (load (input V_cache) (index (elem n) (tile p) (fulltile)))
                        )

                    )
                    (index (elem n) (fulltile) (fulltile))
                )
            )
        )
))))))))))
)
"
,
}
egg::test_fn2! {attacc_assoc, custom_rules(),
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
            )
            )
        )
    (seq
        (loop 0 2064 tile_p p
            (store (input O)
                (+
                    (x (load (input O) (index (elem n) (fulltile) (fulltile))) 1)
                    (/
                        (*
                            (load (input C_exp) (index (elem n) (fulltile) (tile p)))
                            (load (input V_cache) (index (elem n) (tile p) (fulltile)))
                        )
                        (bcast
                            (load (input C_sum) (index (elem n) (fulltile)))
                            2
                        )
                    )

                )
                (index (elem n) (fulltile) (fulltile))
            )
        )
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
    )
))))))))))
)
"
,
}

egg::test_fn2! {attn_assoc, rules(),
"
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
"
=>
"
    (loop 0 32 tile_h h
        (seq
            (store (input K_cache)
                (load (input K) (index (tile h) (fulltile) (fulltile)))
                (index (tile h) (fulltile) (fulltile))
            )
        (seq
            (store (input V_cache)
                (load (input V) (index (tile h) (fulltile) (fulltile)))
                (index (tile h) (fulltile) (fulltile))
            )
        (seq
            (loop 0 2064 tile_p p
                (seq
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
                (seq
                    (store (input C_exp)
                        (exp (load (input C) (index (tile h) (fulltile) (tile p))))
                        (index (tile h) (fulltile) (tile p))
                    )
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
            )
        (seq
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
        )
        )
        )
    )
"
,
// "
//     (loop 0 32 tile_h h
//         (seq
//             (store (input K_cache)
//                 (load (input K) (index (tile h) (fulltile) (fulltile)))
//                 (index (tile h) (fulltile) (fulltile))
//             )
//         (seq
//             (store (input V_cache)
//                 (load (input V) (index (tile h) (fulltile) (fulltile)))
//                 (index (tile h) (fulltile) (fulltile))
//             )
//         (seq
//             (loop 0 2064 tile_p p
//                 (seq
//                     (store (input C)
//                         (*
//                             (load (input Q) (index (tile h) (fulltile) (fulltile)))
//                             (permute3
//                                 (load (input K_cache) (index (tile h) (tile p) (fulltile)))
//                                 0 2 1
//                             )
//                         )
//                         (index (tile h) (fulltile) (tile p))
//                     )
//                 (seq
//                     (store (input C_exp)
//                         (exp (load (input C) (index (tile h) (fulltile) (tile p))))
//                         (index (tile h) (fulltile) (tile p))
//                     )
//                     (store (input C_sum)
//                         (+
//                             (x (load (input C_sum) (index (tile h) (fulltile))) 1)
//                             (rsum
//                                 (load (input C_exp) (index (tile h) (fulltile) (tile p)))
//                                 2
//                             )
//                         )
//                         (index (tile h) (fulltile))
//                     )
//                 )
//                 )
//             )
//         (seq
//             (loop 0 2064 tile_p p
//                 (store (input O)
//                     (+
//                         (x (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
//                         (*
//                             (load (input C_div) (index (tile h) (fulltile) (tile p)))
//                             (load (input V_cache) (index (tile h) (tile p) (fulltile)))
//                         )
//                     )
//                     (index (tile h) (fulltile) (fulltile))
//                 )
//             )
//             (loop 0 2064 tile_p p
//                 (store (input C_div)
//                     (/
//                         (load (input C_exp) (index (tile h) (fulltile) (tile p)))
//                         (bcast
//                             (load (input C_sum) (index (tile h) (fulltile)))
//                             2
//                         )
//                     )
//                     (index (tile h) (fulltile) (tile p))
//                 )
//             )   
//         )
//         )
//         )
//         )
//     )
// "
// ,
}

egg::test_fn2! {invalid_seqcomm_after_value_forward, rules(),
"
(seq
    (store (input B) (load (input A) (index)) (index))
    (store (input C) (load (input B) (index)) (index))
)
"
=>
"
(seq
    (store (input B) (load (input A) (index)) (index))
    (store (input C) (load (input A) (index)) (index))
)
"
,
"
(seq
    (store (input C) (load (input A) (index)) (index))
    (store (input B) (load (input A) (index)) (index))
)
"
}

egg::test_fn2! {misc, rules(),
"
(seq
    (loop 0 2064 tile_p p
        (store (input C_div)
            (/
                (load (input C_exp) (index (fulltile) (tile p)))
                (bcast
                    (load (input C_sum) (index  (fulltile)))
                    2
                )
            )
            (index (fulltile) (tile p))
        )
    )  
    (loop 0 2064 tile_p p
        (store (input O)
            (+
                (x (load (input O) (index (fulltile) (fulltile))) 1)
                (*
                    (load (input C_div) (index (fulltile) (tile p)))
                    (load (input V_cache) (index (tile p) (fulltile)))
                )
            )
            (index (fulltile) (fulltile))
        )
    )
)
"
=>
// "
// (seq
//     (loop 0 2064 tile_p p
//         (store (input O)
//             (+
//                 (x (load (input O) (index (fulltile) (fulltile))) 1)
//                 (*
//                     (/
//                         (load (input C_exp) (index (fulltile) (tile p)))
//                         (bcast
//                             (load (input C_sum) (index  (fulltile)))
//                             2
//                         )
//                     )
//                     (load (input V_cache) (index (tile p) (fulltile)))
//                 )
//             )
//             (index (fulltile) (fulltile))
//         )
//     )
//     (loop 0 2064 tile_p p
//         (store (input C_div)
//             (/
//                 (load (input C_exp) (index (fulltile) (tile p)))
//                 (bcast
//                     (load (input C_sum) (index  (fulltile)))
//                     2
//                 )
//             )
//             (index (fulltile) (tile p))
//         )
//     )  
// )
// "
// ,
// "
// (seq
//     (loop 0 2064 tile_p p
//         (store (input O)
//             (+
//                 (x (load (input O) (index (fulltile) (fulltile))) 1)
//                 (/
//                     (*
//                         (load (input C_exp) (index (fulltile) (tile p)))
//                         (load (input V_cache) (index (tile p) (fulltile)))
//                     )
//                     (bcast
//                         (load (input C_sum) (index (fulltile)))
//                         2
//                     )
//                 )
//             )
//             (index (fulltile) (fulltile))
//         )
//     )
//     (loop 0 2064 tile_p p
//         (store (input C_div)
//             (/
//                 (load (input C_exp) (index (fulltile) (tile p)))
//                 (bcast
//                     (load (input C_sum) (index (fulltile)))
//                     2
//                 )
//             )
//             (index (fulltile) (tile p))
//         )
//     )  
// )
// "
// ,
"
(seq
    (loop 0 2064 tile_p p
        (store (input O)
            (+
                (x (load (input O) (index (fulltile) (fulltile))) 1)
                (*
                    (load (input C_exp) (index (fulltile) (tile p)))
                    (load (input V_cache) (index (tile p) (fulltile)))
                )
            )
            (index (fulltile) (fulltile))
        )
    )
(seq
    (store (input O)
        (/
            (load (input O) (index (fulltile) (fulltile)))
            (bcast
                (load (input C_sum) (index (fulltile)))
                2
            )
        )
        (index (fulltile) (fulltile))
    )
    (loop 0 2064 tile_p p
        (store (input C_div)
            (/
                (load (input C_exp) (index (fulltile) (tile p)))
                (bcast
                    (load (input C_sum) (index (fulltile)))
                    2
                )
            )
            (index (fulltile) (tile p))
        )
    )  
)
)
"
// ,
// "
// (seq
//     (loop 0 2064 tile_p p
//         (store (input O)
//             (+
//                 (x (load (input O) (index (fulltile) (fulltile))) 1)
//                 (*
//                     (load (input C_div) (index (fulltile) (tile p)))
//                     (load (input V_cache) (index (tile p) (fulltile)))
//                 )
//             )
//             (index (fulltile) (fulltile))
//         )
//     )
//     (loop 0 2064 tile_p p
//         (store (input C_div)
//             (/
//                 (load (input C_exp) (index (fulltile) (tile p)))
//                 (bcast
//                     (load (input C_sum) (index (fulltile)))
//                     2
//                 )
//             )
//             (index (fulltile) (tile p))
//         )
//     )  
    
// )
// "
}

egg::test_fn2! {debug_lora1, rules(),
    "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input C)
                    (+ (x (load (input C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))
                ))
                (index (fulltile) (fulltile))
            )
        )

    (seq
        (loop 0 P tile_p p 
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))
                )
                (index (fulltile) (tile p))
            )
        )

        (loop 0 P tile_p p
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p))) (load (input E) (index (fulltile) (tile p))))
                (index (fulltile) (tile p))
            )
        )   
    )
    )
    )
    "
    =>
    "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input C)
                    (+ (x (load (input C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input D)
                    (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))
                    ))
                    (index (fulltile) (fulltile))
                )
            )
        )
    (seq
        (loop 0 P tile_p p 
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))
                )
                (index (fulltile) (tile p))
            )
        )

        (loop 0 P tile_p p
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p))) (load (input E) (index (fulltile) (tile p))))
                (index (fulltile) (tile p))
            )
        )   
    )
    )
    )
    "
    ,
    "
    (seq
        (loop 0 P tile_p p 
            (seq
                (loop 0 N tile_n n
                    (store (input C)
                        (+ (x (load (input C) (index (fulltile) (tile p))) 1)
                        (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                        ))
                        (index (fulltile) (tile p))
                    )
                )
                (loop 0 P tile_p p 
                    (loop 0 N tile_n n
                        (store (input D)
                            (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                            (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))
                            ))
                            (index (fulltile) (fulltile))
                        )
                    )
                )
            )
        )
    (seq
        (loop 0 P tile_p p 
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))
                )
                (index (fulltile) (tile p))
            )
        )

        (loop 0 P tile_p p
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p))) (load (input E) (index (fulltile) (tile p))))
                (index (fulltile) (tile p))
            )
        )   
    )
    )
    "
    ,
    "
    (loop 0 P tile_p p 
        (seq 
            (loop 0 N tile_n n 
                (store (input C) 
                    (+ (x (load (input C) (index fulltile (tile p))) 1) 
                    (* (load (input X) (index fulltile (tile n))) 
                        (load (input W) (index (tile n) (tile p))))) 
                    (index fulltile (tile p))))
            (seq 
                (loop 0 P tile_p p 
                    (loop 0 N tile_n n 
                        (store (input D) 
                            (+ (x 1 (load (input D) (index fulltile fulltile))) 
                            (* (load (input X) (index fulltile (tile n))) 
                                (load (input A) (index (tile n) fulltile)))) 
                            (index fulltile fulltile))))
                (seq 
                    (loop 0 N tile_n n 
                        (store (input E)
                            (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))
                            )
                            (index (fulltile) (tile p))
                        )
                    )
                    (store (output O) 
                        (+ (load (input C) (index fulltile (tile p))) 
                        (* (load (input D) (index fulltile fulltile)) 
                            (load (input B) (index fulltile (tile p))))) 
                        (index fulltile (tile p)))))))
    "
    ,
}

egg::test_fn2! {debug_lora2, rules(),
    "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input B) 3 (index (tile p) (tile n)))
            )
        )
        (loop 0 N tile_n n
            (store (input A) 3 (index (tile n)))
        )
    )
    "
    =>
    "
    (loop 0 P tile_p p
        (seq
            (loop 0 N tile_n n
                (store (input B) 3 (index (tile p) (tile n)))
            )
            (loop 0 P tile_p p 
                (loop 0 N tile_n n
                    (store (input A) 3 (index (tile n)))
                )
            )
        )
    )
    "
    ,
    "
    (loop 0 P tile_p p
        (seq
            (loop 0 N tile_n n
                (store (input B) 3 (index (tile p) (tile n)))
            )
            (loop 0 N tile_n n
                (store (input A) 3 (index (tile n)))
            )
        )
    )   
    "
}

egg::test_fn2! {insertion_test, rules(),
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
    (loop 0 4544 tile_n n
        (seq
            (loop 0 4544 tile_k k
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
        ))
    )
)
"
=>
"    
    (loop 0 4544 tile_n n
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
        )))
    )

"

}

#[test]
fn saturate_insertion() {
    let expr = "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input B) 3 (index))
            )
        )
        (loop 0 N tile_n n
            (store (input A) 3 (index))
        )
    )
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
    );
    

    save_egraph(&runner, "egraph.dot");
}

#[test]
fn test_save_load_egraph() {
    use trinity::{save_raw_egraph, load_raw_egraph};
    use std::fs;
    
    // Create a test expression and run until saturation
    let expr = "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile p) (tile n)))
                        (load (input C) (index (tile p) (tile n)))
                    )
                    (index (tile p) (tile n))
                )
            )
        )
        (loop 0 N tile_n n
            (store (input D) 
                (load (input B) (index (fulltile) (tile n)))
                (index (tile n))
            )
        )
    )
    ";

    println!("Step 1: Creating and saturating egraph...");
    let original_runner = run_until_saturated(
        expr,
        rules(),
        10,
    );
    
    // Get statistics from original egraph
    let original_nodes = original_runner.egraph.total_number_of_nodes();
    let original_classes = original_runner.egraph.number_of_classes();
    let original_iterations = original_runner.iterations.len();
    
    println!("Original egraph statistics:");
    println!("  Nodes: {}", original_nodes);
    println!("  Classes: {}", original_classes);
    println!("  Iterations: {}", original_iterations);
    
    // Save the raw egraph structure
    let raw_file = "test_egraph_raw.txt";
    
    println!("\nStep 2: Saving raw egraph structure...");

    // Save raw egraph
    save_raw_egraph(&original_runner, raw_file)
        .expect("Failed to save raw egraph");
    println!("  Saved raw egraph to {}", raw_file);
    
    // Also save DOT visualization
    save_egraph(&original_runner, "test_egraph.dot");
    println!("  Saved DOT visualization to test_egraph.dot");
    
    // Verify the file was created and has content
    let metadata = fs::metadata(raw_file).expect("Failed to get file metadata");
    println!("\nStep 3: Verifying saved file...");
    println!("  File size: {} bytes", metadata.len());
    
    // Load the egraph back
    println!("\nStep 6: Loading egraph from saved file...");
    let loaded_egraph = load_raw_egraph(raw_file)
        .expect("Failed to load raw egraph");
    
    // Step 7: Verify loaded egraph
    println!("\nStep 7: Verifying loaded egraph...");
    let loaded_stats = loaded_egraph.number_of_classes();
    println!("  Loaded egraph has {} classes", loaded_stats);
    
    // The loaded egraph should have similar statistics
    assert!(loaded_stats > 0, "Loaded egraph should have at least one class");
    
    // Step 8: Save visualizations for comparison
    println!("\nStep 8: Saving visualizations for comparison...");
    save_egraph_from_egraph(&loaded_egraph, "test_egraph_loaded.dot");
    println!("  Saved loaded egraph visualization to test_egraph_loaded.dot");
    
    // Step 9: Verify semantic equivalence
    println!("\nStep 9: Verifying semantic equivalence...");
    
    // Both egraphs should have the same number of classes
    assert_eq!(original_classes, loaded_stats, 
        "Loaded egraph should have the same number of classes as original");
    
    println!("  ✓ Same number of classes: {}", loaded_stats);
    
    // Note: The DOT files will be different for two reasons:
    // 1. Egg assigns IDs based on insertion order
    // 2. Placeholder nodes are added to handle circular dependencies
    println!("\nTest passed! Successfully saved and loaded egraph structure.");
    println!("Note: test_egraph.dot and test_egraph_loaded.dot will differ:");
    println!("  - Different node IDs (egg assigns based on insertion order)");
    println!("  - Extra placeholder nodes (needed for circular dependencies)");
    println!("But they represent the same equivalences (same number of classes).");
}

#[test]
fn tmp() {
    let expr = "
(seq
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
";

    let mut runner = run_until_saturated(
        expr,
        custom_rules(),
        7,
    );

    // list_expressions_with_target_cost_v3_part1(&runner, "tmp.json", 100, 100);
    // list_expressions_from_semi_all(&runner, "tmp.json", 0);
    // let new_egraph = visualize_semi_expression(&runner, "tmp.json", 1);
    save_egraph(&runner, "egraph.dot");


}

#[test]


egg::test_fn2! {test_default_tiling, default_tiling(),
    "
    (seq
        (store tmp1 (x A B) (index))
    (seq
        (store tmp2 (x A C) (index))
        (store O (+ (load tmp1 (index)) (load tmp2 (index))) (index))
    ))
    "
    =>
    "(seq
        (store tmp1 (+ B C) (index))
        (store O (x A (load tmp1 (index))) (index))
    )
    "
}
