use egg::{test_fn2, test_fn_not2};
use trinity::*;

egg::test_fn2! {test_whole_hacked, rules(),
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
            (store (tensor QKV1)
                (+
                    (x (load (tensor QKV1) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQKV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor QKV2)
            (unsqueeze (load (tensor QKV1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor QKV)
            (permute3
                (load (tensor QKV2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input KV_cache)
            (load (tensor KV) (index (tile h) (fulltile) (fulltile)))
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
)))))))))))))
"
=>
"
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

(seq
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
                (store (tensor QKV1)
                    (+
                        (x (load (tensor QKV1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (tensor X_norm) (index (fulltile) (tile k)))
                            (load (input WQKV) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor QKV2)
            (unsqueeze (load (tensor QKV1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (seq
            (store (tensor QKV)
                (permute3
                    (load (tensor QKV2) (index (fulltile) (tile h) (fulltile)))
                    1 0 2
                )
                (index (tile h) (fulltile) (fulltile))
            )
            (store (input KV_cache)
                (load (tensor KV) (index (tile h) (fulltile) (fulltile)))
                (index (tile h) (const_tile 4096 16) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
            (seq
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
            (seq
                (store (tensor C_exp)
                    (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                    (index (tile h) (fulltile) (tile p))
                )
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
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
            (seq
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
)))))))
"
,
"
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

(seq
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
                (store (tensor QKV1)
                    (+
                        (x (load (tensor QKV1) (index (fulltile) (tile n))) 1)
                        (/
                            (*
                                (load (input X) (index (fulltile) (tile k)))
                                (load (input WQKV) (index (tile k) (tile n)))
                            )
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
                    )
                    (index (fulltile) (tile n))
                )
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor QKV2)
            (unsqueeze (load (tensor QKV1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (seq
            (store (tensor QKV)
                (permute3
                    (load (tensor QKV2) (index (fulltile) (tile h) (fulltile)))
                    1 0 2
                )
                (index (tile h) (fulltile) (fulltile))
            )
            (store (input KV_cache)
                (load (tensor KV) (index (tile h) (fulltile) (fulltile)))
                (index (tile h) (const_tile 4096 16) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
            (seq
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
            (seq
                (store (tensor C_exp)
                    (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                    (index (tile h) (fulltile) (tile p))
                )
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
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
            (seq
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
)))))))
"
,
"
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

(seq
    (loop 0 4096 tile_n n
        (seq
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
                    (store (tensor QKV1)
                        (+
                            (x (load (tensor QKV1) (index (fulltile) (tile n))) 1)
                            (*
                                (load (input X) (index (fulltile) (tile k)))
                                (load (input WQKV) (index (tile k) (tile n)))
                            )
                        )
                        (index (fulltile) (tile n))
                    )
                )
            )
            (store (tensor QKV1) 
                (/
                    (load (tensor QKV1) (index (fulltile) (tile n)))
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
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor QKV2)
            (unsqueeze (load (tensor QKV1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (seq
            (store (tensor QKV)
                (permute3
                    (load (tensor QKV2) (index (fulltile) (tile h) (fulltile)))
                    1 0 2
                )
                (index (tile h) (fulltile) (fulltile))
            )
            (store (input KV_cache)
                (load (tensor KV) (index (tile h) (fulltile) (fulltile)))
                (index (tile h) (const_tile 4096 16) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
            (seq
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
            (seq
                (store (tensor C_exp)
                    (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                    (index (tile h) (fulltile) (tile p))
                )
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
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 528 tile_p p
            (seq
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
)))))))
"
}

egg::test_fn2! {test_rms_norm_all, rules(),
// "
// (seq
//     (loop 0 4096 tile_k k
//         (seq
//             (store (tensor X1)
//                 (sqr (load (input X) (index (fulltile) (tile k))))
//                 (index (fulltile) (tile k))
//             )
//             (store (tensor X2)
//                 (+
//                     (x (load (tensor X2) (index (fulltile))) 1)
//                     (rsum
//                         (load (tensor X1) (index (fulltile) (tile k)))
//                         1
//                     )
//                 )
//                 (index (fulltile))
//             )
//         )
//     )
// (seq
//     (loop 0 4096 tile_k k
//         (store (tensor X_norm)
//             (/
//                 (load (input X) (index (fulltile) (tile k)))
//                 (bcast
//                     (sqrt
//                         (/
//                             (load (tensor X2) (fulltile))
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
//     (loop 0 4096 tile_n n
//         (loop 0 4096 tile_k k
//             (store (tensor Q1)
//                 (+
//                     (x (load (tensor Q1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (tensor X_norm) (index (fulltile) (tile k)))
//                         (load (input WQ) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 4096 tile_n n
//         (loop 0 4096 tile_k k
//             (store (tensor K1)
//                 (+
//                     (x (load (tensor K1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (tensor X_norm) (index (fulltile) (tile k)))
//                         (load (input WK) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
//     (loop 0 4096 tile_n n
//         (loop 0 4096 tile_k k
//             (store (tensor V1)
//                 (+
//                     (x (load (tensor V1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (tensor X_norm) (index (fulltile) (tile k)))
//                         (load (input WV) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
// ))))
// "
// =>
// "
// (seq
//     (loop 0 4096 tile_k k
//         (seq
//             (store (tensor X1)
//                 (sqr (load (input X) (index (fulltile) (tile k))))
//                 (index (fulltile) (tile k))
//             )
//             (store (tensor X2)
//                 (+
//                     (x (load (tensor X2) (index (fulltile))) 1)
//                     (rsum
//                         (load (tensor X1) (index (fulltile) (tile k)))
//                         1
//                     )
//                 )
//                 (index (fulltile))
//             )
//         )
//     )    
//     (loop 0 4096 tile_n n
//         (loop 0 4096 tile_k k
//             (seq
//                 (store (tensor X_norm)
//                     (/
//                         (load (input X) (index (fulltile) (tile k)))
//                         (bcast
//                             (sqrt
//                                 (/
//                                     (load (tensor X2) (fulltile))
//                                     4096
//                                 )
//                             )
//                             1
//                         )
//                     )
//                     (index (fulltile) (tile k))
//                 )
//             (seq
//                 (store (tensor Q1)
//                     (+
//                         (x (load (tensor Q1) (index (fulltile) (tile n))) 1)
//                         (*
//                             (load (tensor X_norm) (index (fulltile) (tile k)))
//                             (load (input WQ) (index (tile k) (tile n)))
//                         )
//                     )
//                     (index (fulltile) (tile n))
//                 )
//             (seq
//                 (store (tensor K1)
//                     (+
//                         (x (load (tensor K1) (index (fulltile) (tile n))) 1)
//                         (*
//                             (load (tensor X_norm) (index (fulltile) (tile k)))
//                             (load (input WK) (index (tile k) (tile n)))
//                         )
//                     )
//                     (index (fulltile) (tile n))
//                 )
//                 (store (tensor V1)
//                     (+
//                         (x (load (tensor V1) (index (fulltile) (tile n))) 1)
//                         (*
//                             (load (tensor X_norm) (index (fulltile) (tile k)))
//                             (load (input WV) (index (tile k) (tile n)))
//                         )
//                     )
//                     (index (fulltile) (tile n))
//                 )
            
//             )
            
//             )
//             )
//         )
//     )
// )
// "
"
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
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (seq
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
            (seq
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
        )
    )
)
)
"
=>
"
(loop 0 4096 tile_n n
    (seq
        (loop 0 4096 tile_k k
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
            (seq
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
            (seq
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
        (store (tensor Q1)
            (/
                (load (tensor Q1) (index (fulltile) (tile n)))
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
            (index (fulltile) (tile n))
        )
    (seq
        (store (tensor K1)
            (/
                (load (tensor K1) (index (fulltile) (tile n)))
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
            (index (fulltile) (tile n))
        )
        (store (tensor V1)
            (/
                (load (tensor V1) (index (fulltile) (tile n)))
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
            (index (fulltile) (tile n))
        )
    )
    )
    )
    )
)
"
}


egg::test_fn2_v2! {test_attacc_skip_ft, 
    rules1 = rules_wo_seqcomm(),
    rules2 = only_seqcomm_rules(),
    n = 3,
    m = 1,
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
(seq
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
(seq
    (loop 0 32 tile_h h
        (store (input O1)
            (permute3
                (load (input O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 2048 64 n
        (store (input O2)
            (squeeze
                (load (input O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
)))))))))))))))))
"
=>
// "
// (seq
//     (loop 0 2048 tile_n n
//         (loop 0 2048 tile_k k
//             (store (input Q1)
//                 (+
//                     (x (load (input Q1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (input X) (index (fulltile) (tile k)))
//                         (load (input WQ) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 2048 tile_n n
//         (loop 0 2048 tile_k k
//             (store (input K1)
//                 (+
//                     (x (load (input K1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (input X) (index (fulltile) (tile k)))
//                         (load (input WK) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 2048 tile_n n
//         (loop 0 2048 tile_k k
//             (store (input V1)
//                 (+
//                     (x (load (input V1) (index (fulltile) (tile n))) 1)
//                     (*
//                         (load (input X) (index (fulltile) (tile k)))
//                         (load (input WV) (index (tile k) (tile n)))
//                     )
//                 )
//                 (index (fulltile) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 2048 64 n
//         (seq
//             (store (input Q2)
//                 (unsqueeze (load (input Q1) (index (fulltile) (tile n))) 1)
//                 (index (fulltile) (elem n) (fulltile))
//             )
//         (seq
//             (store (input K2)
//                 (unsqueeze (load (input K1) (index (fulltile) (tile n))) 1)
//                 (index (fulltile) (elem n) (fulltile))
//             )
//             (store (input V2)
//                 (unsqueeze (load (input V1) (index (fulltile) (tile n))) 1)
//                 (index (fulltile) (elem n) (fulltile))
//             )
//         )
//         )
//     )
// (seq
//     (loop 0 32 tile_h h
//         (seq
//             (store (input Q)
//                 (permute3
//                     (load (input Q2) (index (fulltile) (tile h) (fulltile)))
//                     1 0 2
//                 )
//                 (index (tile h) (fulltile) (fulltile))
//             )
//         (seq
//             (store (input K)
//                 (permute3
//                     (load (input K2) (index (fulltile) (tile h) (fulltile)))
//                     1 0 2
//                 )
//                 (index (tile h) (fulltile) (fulltile))
//             )
//         (seq
//             (store (input V)
//                 (permute3
//                     (load (input V2) (index (fulltile) (tile h) (fulltile)))
//                     1 0 2
//                 )
//                 (index (tile h) (fulltile) (fulltile))
//             )
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
//                 (store (input C)
//                     (*
//                         (load (input Q) (index (tile h) (fulltile) (fulltile)))
//                         (permute3
//                             (load (input K_cache) (index (tile h) (tile p) (fulltile)))
//                             0 2 1
//                         )
//                     )
//                     (index (tile h) (fulltile) (tile p))
//                 )
//             )
//         (seq
//             (loop 0 2064 tile_p p
//                 (store (input C_exp)
//                     (exp (load (input C) (index (tile h) (fulltile) (tile p))))
//                     (index (tile h) (fulltile) (tile p))
//                 )
//             )
//         (seq
//             (loop 0 2064 tile_p p
//                 (store (input C_sum)
//                     (+
//                         (x (load (input C_sum) (index (tile h) (fulltile))) 1)
//                         (rsum
//                             (load (input C_exp) (index (tile h) (fulltile) (tile p)))
//                             2
//                         )
//                     )
//                     (index (tile h) (fulltile))
//                 )
//             )
//         (seq
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
//             (store (input O1)
//                 (permute3
//                     (load (input O) (index (tile h) (fulltile) (fulltile)))
//                     1 0 2
//                 )
//                 (index (fulltile) (tile h) (fulltile))
//             )
//         ))))))))))
//     )
//     (loop 0 2048 64 n
//         (store (input O2)
//             (squeeze
//                 (load (input O1) (index (fulltile) (elem n) (fulltile)))
//                 1
//             )
//             (index (fulltile) (tile n))
//         )
//     )
// )))))
// "
// =>
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
"
,
}

egg::test_fn2! {test_flashdecoding_skip_ft, rules(),
    "
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
                (store (input O)
                    (+
                        (x (load (input O) (index (tile m) (fulltile))) 1)
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
    "
    =>
    "
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
                        (store (input C_exp)
                            (exp 
                                (load (input C) (index (tile m) (tile n)))
                            )
                            (index (tile m) (tile n))
                        )
                    )
                )
            (seq
                (dloop 0 N new_tile_n new_n
                    (dloop new_n (+ new_n new_tile_n) tile_n n
                        (store (input new_C_sum)
                            (+
                                (x (load (input new_C_sum) (index (elem new_n) (index (tile m)))) 1)
                                (rsum
                                    (exp
                                        (*
                                            (load (input Q) (index (tile m) (fulltile)))
                                            (load (input K) (index (fulltile) (tile n)))
                                        )
                                    )
                                    1
                                )
                            )
                            (index (elem new_n) (index (tile m)))
                        )
                    )
                )
            (seq
                (store (input C_sum)
                    (rsum
                        (load (input new_C_sum) (index (fulltile) (index (tile m))))
                        0
                    )
                    (index (tile m))
                )

            (seq
                (dloop 0 N new_tile_n new_n
                    (dloop new_n (+ new_n new_tile_n) tile_n n
                        (store (input new_O)
                            (+
                                (x (load (input new_O) (index (elem new_n) (index (tile m) (fulltile)))) 1)
                                (*
                                    (exp
                                        (*
                                            (load (input Q) (index (tile m) (fulltile)))
                                            (load (input K) (index (fulltile) (tile n)))
                                        )
                                    )
                                    (load (input V) (index (tile n) (fulltile)))
                                )
                            )
                            (index (elem new_n) (index (tile m) (fulltile)))
                        )
                    )
                )
            
            
            (seq
                (store (input O)
                    (rsum
                        (load (input new_O) (index (fulltile) (index (tile m) (fulltile))))
                        0
                    )
                    (index (tile m) (fulltile))
                )
            (seq
                (store (input O)
                    (/
                        (load (input O) (index (tile m) (fulltile)))
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
            )
            )
            )
        )
    "
    ,
    "
    (seq
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
                        (store (input C_exp)
                            (exp 
                                (load (input C) (index (tile m) (tile n)))
                            )
                            (index (tile m) (tile n))
                        )
                    )
                )
                (dloop 0 N new_tile_n new_n
                    (dloop new_n (+ new_n new_tile_n) tile_n n
                        (seq
                            (store (input new_C_sum)
                                (+
                                    (x (load (input new_C_sum) (index (elem new_n) (index (tile m)))) 1)
                                    (rsum
                                        (exp
                                            (*
                                                (load (input Q) (index (tile m) (fulltile)))
                                                (load (input K) (index (fulltile) (tile n)))
                                            )
                                        )
                                        1
                                    )
                                )
                                (index (elem new_n) (index (tile m)))
                            )

                            (store (input new_O)
                                (+
                                    (x (load (input new_O) (index (elem new_n) (index (tile m) (fulltile)))) 1)
                                    (*
                                        (exp
                                            (*
                                                (load (input Q) (index (tile m) (fulltile)))
                                                (load (input K) (index (fulltile) (tile n)))
                                            )
                                        )
                                        (load (input V) (index (tile n) (fulltile)))
                                    )
                                )
                                (index (elem new_n) (index (tile m) (fulltile)))
                            )
                        )
                    )
                )
            )
        )     
        (loop 0 M tile_m m
            (seq
                (store (input C_sum)
                    (rsum
                        (load (input new_C_sum) (index (fulltile) (index (tile m))))
                        0
                    )
                    (index (tile m))
                )
            
            (seq
                (store (input O)
                    (rsum
                        (load (input new_O) (index (fulltile) (index (tile m) (fulltile))))
                        0
                    )
                    (index (tile m) (fulltile))
                )
            (seq
                (store (input O)
                    (/
                        (load (input O) (index (tile m) (fulltile)))
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
        )
    )
    "
    ,
    // "
    // (seq
    //     (loop 0 M tile_m m
    //         (seq
    //             (loop 0 N tile_n n
    //                 (seq
    //                     (store (input C) 
    //                         (*
    //                             (load (input Q) (index (tile m) (fulltile)))
    //                             (load (input K) (index (fulltile) (tile n)))
    //                         )
    //                         (index (tile m) (tile n))
    //                     )
    //                     (store (input C_exp)
    //                         (exp 
    //                             (load (input C) (index (tile m) (tile n)))
    //                         )
    //                         (index (tile m) (tile n))
    //                     )
    //                 )
    //             )
    //             (loop 0 N new_tile_n new_n
    //                 (loop new_n (+ new_n new_tile_n) tile_n n
    //                     (seq
    //                         (store (input new_C_sum)
    //                             (+
    //                                 (x (load (input new_C_sum) (index (elem new_n) (index (tile m)))) 1)
    //                                 (rsum
    //                                     (exp
    //                                         (*
    //                                             (load (input Q) (index (tile m) (fulltile)))
    //                                             (load (input K) (index (fulltile) (tile n)))
    //                                         )
    //                                     )
    //                                     1
    //                                 )
    //                             )
    //                             (index (elem new_n) (index (tile m)))
    //                         )
    //                         (store (input new_O)
    //                             (+
    //                                 (x (load (input new_O) (index (elem new_n) (index (tile m) (fulltile)))) 1)
    //                                 (*
    //                                     (exp
    //                                         (*
    //                                             (load (input Q) (index (tile m) (fulltile)))
    //                                             (load (input K) (index (fulltile) (tile n)))
    //                                         )
    //                                     )
    //                                     (load (input V) (index (tile n) (fulltile)))
    //                                 )
    //                             )
    //                             (index (elem new_n) (index (tile m) (fulltile)))
    //                         )
    //                     )
    //                 )
    //             )
    //         )
    //     )
    //     (loop 0 M tile_m m
    //         (seq
    //             (store (input C_sum)
    //                 (rsum
    //                     (load (input new_C_sum) (index (fulltile) (index (tile m))))
    //                     0
    //                 )
    //                 (index (tile m))
    //             )
    //         (seq
    //             (store (output O)
    //                 (rsum
    //                     (load (input new_O) (index (fulltile) (index (tile m) (fulltile))))
    //                     0
    //                 )
    //                 (index (tile m) (fulltile))
    //             )
    //         (seq
    //             (store (output O)
    //                 (/
    //                     (load (output O) (index (tile m) (fulltile)))
    //                     (bcast (load (input C_sum) (index (tile m))) 1)
    //                 )
    //                 (index (tile m) (fulltile))
    //             )
    //             (loop 0 N tile_n n
    //                 (store (input C_div)
    //                     (/
    //                         (load (input C_exp) (index (tile m) (tile n)))
    //                         (bcast (load (input C_sum) (index (tile m))) 1)
    //                     )
    //                     (index (tile m) (tile n))
    //                 )
    //             )
    //         )
    //         )
    //         )
    //     )
    // )
    // "
}

egg::test_fn2! {test_flashattn2_skip_ft, rules(),
    "
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
    "
    =>
    "
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
    "
    ,
    "
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
    "
    ,
    "
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
                )
                )
            )
        (seq
            (loop 0 N tile_n n
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
        (seq
            (loop 0 N tile_n n
                (store (input C_div)
                    (/
                        (load (input C_exp) (index (tile m) (tile n)))
                        (bcast (load (input C_sum) (index (tile m))) 1)
                    )
                    (index (tile m) (tile n))
                )
            )
            (store (output O)
                (/
                    (load (output O) (index (tile m) (fulltile)))
                    (bcast (load (input C_sum) (index (tile m))) 1)
                )
                (index (tile m) (fulltile))
            )
        )
        )
        )
    )
    "
    // "
    // (seq
    //     (loop 0 M tile_m m
    //         (loop 0 N tile_n n 
    //             (seq
    //                 (dummy)
    //             (seq
    //                 (dummy)
    //                 (store (input C_sum)
    //                     (+
    //                         (x (load (input C_sum) (index (tile m))) 1)
    //                         (rsum 
    //                             (exp 
    //                                 (*
    //                                     (load (input Q) (index (tile m) (fulltile)))
    //                                     (load (input K) (index (fulltile) (tile n)))
    //                                 )
    //                             )
    //                             1
    //                         )
    //                     )
    //                     (index (tile m))
    //                 )
    //             )
    //             )
    //         )
    //     )
    //     (loop 0 M tile_m m
    //         (loop 0 N tile_n n 
    //             (seq
    //                 (dummy)
    //                 (store (output O)
    //                     (+
    //                         (x (load (output O) (index (tile m) (fulltile))) 1)
    //                         (*
    //                             (/
    //                                 (exp 
    //                                     (*
    //                                         (load (input Q) (index (tile m) (fulltile)))
    //                                         (load (input K) (index (fulltile) (tile n)))
    //                                     )
    //                                 )
    //                                 (bcast (load (input C_sum) (index (tile m))) 1)
    //                             )
    //                             (load (input V) (index (tile n) (fulltile)))
    //                         )
    //                     )
    //                     (index (tile m) (fulltile))
    //                 )
    //             )
    //         )
    //     )
    // )
    // "
    // ,
    // "
    // (loop 0 M tile_m m
    //     (seq
    //         (loop 0 N tile_n n
    //             (seq
    //                 (dummy)
    //             (seq
    //                 (dummy)
    //             (seq
    //                 (store (input C_sum)
    //                     (+
    //                         (x (load (input C_sum) (index (tile m))) 1)
    //                         (rsum 
    //                             (exp 
    //                                 (*
    //                                     (load (input Q) (index (tile m) (fulltile)))
    //                                     (load (input K) (index (fulltile) (tile n)))
    //                                 )
    //                             )
    //                             1
    //                         )
    //                     )
    //                     (index (tile m))
    //                 )
    //                 (store (output O)
    //                     (+
    //                         (x (load (output O) (index (tile m) (fulltile))) 1)
    //                         (*
    //                             (exp 
    //                                 (*
    //                                     (load (input Q) (index (tile m) (fulltile)))
    //                                     (load (input K) (index (fulltile) (tile n)))
    //                                 )
    //                             )
    //                             (load (input V) (index (tile n) (fulltile)))
    //                         )
    //                     )
    //                     (index (tile m) (fulltile))
    //                 )
    //             )
    //             )
    //             )
    //         )
    //     (seq
    //         (store (output O)
    //             (/
    //                 (load (output O) (index (tile m) (fulltile)))
    //                 (bcast (load (input C_sum) (index (tile m))) 1)
    //             )
    //             (index (tile m) (fulltile))
    //         )
    //         (loop 0 N tile_n n
    //             (dummy)
    //         )
    //     )
    //     )
    // )
    // "
}

egg::test_fn2! {test_gated_mlp, rules(),
    "
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (input C1) 
                        (+
                            (x (load (input C1) (index (tile m) (tile n))) 1)
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
                (store (input C1_exp) 
                    (exp
                        (load (input C1) (index (tile m) (tile n)))
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
    "
    =>
    // "
    // (loop 0 M tile_m m 
    //     (loop 0 N tile_n n
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (seq
    //                     (store (input C1) 
    //                         (+
    //                             (x (load (input C1) (index (tile m) (tile n))) 1)
    //                             (*
    //                                 (load (input X) (index (tile m) (tile k)))
    //                                 (load (input W1) (index (tile k) (tile n)))
    //                             )
    //                         )
    //                     (index (tile m) (tile n))
    //                     )
    //                     (store (input C2) 
    //                         (+
    //                             (x (load (input C2) (index (tile m) (tile n))) 1)
    //                             (*
    //                                 (load (input X) (index (tile m) (tile k)))
    //                                 (load (input W2) (index (tile k) (tile n)))
    //                             )
    //                         )
    //                     (index (tile m) (tile n))
    //                     )
    //                 )
    //             )
    //         (seq
    //             (dummy)
    //             (store (output O) 
    //                 (x
    //                     (exp
    //                         (load (input C1) (index (tile m) (tile n)))
    //                     )
    //                     (load (input C2) (index (tile m) (tile n)))
    //                 )
    //                 (index (tile m) (tile n))
    //             )
    //         )
    //         )
    //     )
    // )
    // "
    // ,
    // "
    // (loop 0 M tile_m m 
    //     (loop 0 N tile_n n
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (store (input C1) 
    //                     (+
    //                         (x (load (input C1) (index (tile m) (tile n))) 1)
    //                         (*
    //                             (load (input X) (index (tile m) (tile k)))
    //                             (load (input W1) (index (tile k) (tile n)))
    //                         )
    //                     )
    //                     (index (tile m) (tile n))
    //                 )
    //             )
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (store (input C2) 
    //                     (+
    //                         (x (load (input C2) (index (tile m) (tile n))) 1)
    //                         (*
    //                             (load (input X) (index (tile m) (tile k)))
    //                             (load (input W2) (index (tile k) (tile n)))
    //                         )
    //                     )
    //                     (index (tile m) (tile n))
    //                 )
    //             )
            
    //         (seq
    //             (store (output O) 
    //                 (x
    //                     (exp
    //                         (load (input C1) (index (tile m) (tile n)))
    //                     )
    //                     (load (input C2) (index (tile m) (tile n)))
    //                 )
    //                 (index (tile m) (tile n))
    //             )
    //             (store (input C1_exp) 
    //                 (exp
    //                     (load (input C1) (index (tile m) (tile n)))
    //                 )
    //                 (index (tile m) (tile n))
    //             )
    //         )
    //         )
    //         )
    //     )
    // )
    // "
    // ,
    // "
    // (loop 0 M tile_m m 
    //     (loop 0 N tile_n n
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (store (input C1) 
    //                     (+
    //                         (x (load (input C1) (index (tile m) (tile n))) 1)
    //                         (*
    //                             (load (input X) (index (tile m) (tile k)))
    //                             (load (input W1) (index (tile k) (tile n)))
    //                         )
    //                     )
    //                     (index (tile m) (tile n))
    //                 )
    //             )
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (store (output O) 
    //                     (+
    //                         (x (load (output O) (index (tile m) (tile n))) 1)
    //                         (x
    //                             (*
    //                                 (load (input X) (index (tile m) (tile k)))
    //                                 (load (input W2) (index (tile k) (tile n)))
    //                             )
    //                             (exp
    //                                 (load (input C1) (index (tile m) (tile n)))
    //                             )
    //                         )
    //                     )
    //                     (index (tile m) (tile n))
    //                 )
    //             )
    //             (store (input C1_exp) 
    //                 (exp
    //                     (load (input C1) (index (tile m) (tile n)))
    //                 )
    //                 (index (tile m) (tile n))
    //             )
    //         )
    //         )
    //     )
    // )
    // "
    // ,
    "
    (loop 0 M tile_m m 
        (loop 0 N tile_n n
            (seq
                (loop 0 K tile_k k 
                    (seq
                        (store (input C1) 
                            (+
                                (x (load (input C1) (index (tile m) (tile n))) 1)
                                (*
                                    (load (input X) (index (tile m) (tile k)))
                                    (load (input W1) (index (tile k) (tile n)))
                                )
                            )
                        (index (tile m) (tile n))
                        )
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
            (seq
                (store (input C1_exp) 
                    (exp
                        (load (input C1) (index (tile m) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
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
    "
}

egg::test_fn2! {test_lora_skip_ft, rules(),
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
    ,
    "
    (loop 0 P tile_p p 
        (seq
            (loop 0 N tile_n n
                (seq
                    (store (input C)
                        (+ (x (load (input C) (index (fulltile) (tile p))) 1)
                        (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                        ))
                        (index (fulltile) (tile p))
                    )
                    (store (input D)
                        (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                        (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))
                        ))
                        (index (fulltile) (fulltile))
                    )
                )
            )
        (seq
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p))))
                (index (fulltile) (tile p))
            )
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p))) (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))))
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
                (seq
                    (store (input C)
                        (+ (x (load (input C) (index (fulltile) (tile p))) 1)
                        (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                        ))
                        (index (fulltile) (tile p))
                    )
                    (store (input E)
                        (+ (x (load (input E) (index (fulltile) (tile p))) 1)
                            (* (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile)))) (load (input B) (index (fulltile) (tile p))))
                        )
                        (index (fulltile) (tile p))
                    )
                )
            )

            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p))) (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p)))))
                (index (fulltile) (tile p))
            )
        )
    )
    "
    ,
    "
    (loop 0 P tile_p p
        (loop 0 N tile_n n
            (store (output O)
                (+
                    (x (load (output O) (index (fulltile) (tile p))) 1)
                    (*
                        (concat
                            (load (input X) (index (fulltile) (tile n)))
                            (* (load (input X) (index (fulltile) (tile n))) (load (input A) (index (tile n) (fulltile))))
                            1
                        )
                        (concat
                            (load (input W) (index (tile n) (tile p)))
                            (load (input B) (index (fulltile) (tile p)))
                            0
                        )
                    )
                )
                (index (fulltile) (tile p))
            )
        )   
    )
    "
    ,
}