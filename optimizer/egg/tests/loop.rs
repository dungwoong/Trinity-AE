use egg::{rewrite as rw, *};
use std::collections::VecDeque;
use std::collections::HashSet;
use std::collections::HashMap;
use std::fmt::Display;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

define_language! {
    enum TileLang {
    
        "loop" = Loop([Id; 5]), // loop start end tile_n loop_var body
        "dloop" = DLoop([Id; 5]), // dloop start end tile_n loop_var body => loop that only fusion rule can be applied
        "tmp_loop" = TLoop([Id; 6]), // tmp_loop start end tile_n loop_var1 loop_var2 body => we need to rebuild this node during modify()
        "input" = Input(Id),    // name of tensor
        "output" = Output(Id), // name of final output tensor of the tensor program
        "tile" = Tile(Id), // tile n = [n:n+tile_n], tile_n is the tile size of loop that is related to n
        "fulltile" = FullTile,  // fulltile = [:]. Full tile of axis
        "elem" = Elem(Id), // elem n = [n//tile_n:n//tile_n+1], tile_n is the tile size of loop that is related to n
        "index" = Index(Box<[Id]>), // index (tile n) (tile m) ... = [m:m+tile_m, n:n+tile_n, ...]
        "load" = Load([Id; 2]), // (load A index) => A
        "store" = Store([Id; 3]), // store A val index ...
        "seq" = Seq([Id; 2]),   // seq body1 body2 ...

        "const" = Const(Id), // (const N) meaning that the N is a constant, not variable

        "+" = Add([Id; 2]), // a + b
        "-" = Sub([Id; 2]), // a - b
        "*" = Mul([Id; 2]), // a * b (elementwise)
        "/" = Div([Id; 2]), // a / b
        "exp" = Exp(Id), // exp(a)
        "@" = Matmul([Id; 2]), // a @ b (matrix multiplication)
        "rsum" = ReduceSum([Id; 2]), // reduce_sum(a, axis)

        "concat" = Concat([Id; 3]), // concat(a, b, axis)
        "bcast" = Broadcast([Id; 2]), // broadcast(a, axis)

        "permute3" = Permute3([Id; 4]), // permute(A, 0, 2, 1)
        "squeeze" = Squeeze([Id; 2]), // squeeze(A, axis)
        "unsqueeze" = Unsqueeze([Id; 2]), // unsqueeze(A, axis)
        "dummy" = Dummy,

        "sloop" = SLoop([Id; 5]), // Sequential loop => not used in rewriting phase
        "ploop" = PLoop([Id; 5]), // Parallel loop => not used in rewriting phase

        Num(i32),
        Var(egg::Symbol),
    }
}



type EGraph = egg::EGraph<TileLang, LoopAnalysis>;
static max_iter: usize = 10;

fn custom_rules() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![

        // rw!("dloop-fusion-tail"; 
        //     "(seq (dloop ?start ?n ?tile1 ?loop_var ?body1) (seq (dloop ?start ?n ?tile2 ?loop_var ?body2) ?others))" => 
        //     "(seq (dloop ?start ?n ?tile1 ?loop_var (seq ?body1 ?body2)) ?others)" 
        //     if and_all!(
        //         no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
        //         is_not_num(var("?tile1")),
        //         is_not_num(var("?tile2")),
        //     )
        // ),
        // rw!("dloop-fusion"; 
        //     "(seq (dloop ?start ?n ?tile1 ?loop_var ?body1) (dloop ?start ?n ?tile2 ?loop_var ?body2))" => 
        //     "(dloop ?start ?n ?tile1 ?loop_var (seq ?body1 ?body2))" 
        //     if and_all!(
        //         no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
        //         is_not_num(var("?tile1")),
        //         is_not_num(var("?tile2")),
        //     )
        // ),

        rw!("loop-fusion-unified";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (loop ?start ?m ?tile2 ?loop_var2 ?body2))" =>
            {
                LoopFusion {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2")
                }
            }
        ),
        rw!("loop-fusion-unified-tail";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (seq (loop ?start ?m ?tile2 ?loop_var2 ?body2) ?others))" =>
            {
                LoopFusionTail {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2"), others: var("?others")
                }
            }
        ),

        rw!("seq-comm-tail";
            "(seq ?a (seq ?b ?body))" => "(seq ?b (seq ?a ?body))"
            if and_all! (
                no_all_dependency(var("?a"), var("?b")),
                no_seq(var("?a")),
                no_seq(var("?b")),
            ) 
        ),
        // rw!("seq-comm";
        //     "(seq ?a ?b)" => "(seq ?b ?a)"
        //     if and_all!(
        //         no_all_dependency(var("?a"), var("?b")),
        //         no_seq(var("?a")),
        //         no_seq(var("?b")),
        //     )
        // ),
    ]
}
fn default_tiling() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![
        rw!("factor-mul-add";
            "(seq
                (store ?tmp1 (* ?a ?b) ?idx)
            (seq
                (store ?tmp2 (* ?a ?c) ?idx)
                (store ?output (+ (load ?tmp1 ?idx) (load ?tmp2 ?idx)) ?idx)
            ))" =>
            "(seq
                (store ?tmp1 (+ ?b ?c) ?idx)
                (store ?output (* ?a (load ?tmp1 ?idx)) ?idx)
            )"
        ),
        rw!("tiling-mul1";
            "(store ?tmp1 (* A B) ?idx)" => "(tile AB)"
        ),
        rw!("tiling-mul2";
            "(store ?tmp1 (* A C) ?idx)" => "(tile AC)"
        ),
    ]
}

fn rules() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![
        // rw!("seq-assoc1"; "(seq ?a (seq ?b ?c))" => "(seq (seq ?a ?b) ?c)"),
        // rw!("seq-assoc2"; "(seq (seq ?a ?b) ?c)" => "(seq ?a (seq ?b ?c))"),
        // loop transformation rules
        rw!("seq-comm-tail";
            "(seq ?a (seq ?b ?body))" => "(seq ?b (seq ?a ?body))"
            if and_all! (
                no_all_dependency(var("?a"), var("?b")),
                no_seq(var("?a")),
                no_seq(var("?b")),
            ) 
        ),
        rw!("seq-comm";
            "(seq ?a ?b)" => "(seq ?b ?a)"
            if and_all!(
                no_all_dependency(var("?a"), var("?b")),
                no_seq(var("?a")),
                no_seq(var("?b")),
            )
        ),
        rw!("dloop-fusion-tail"; 
            "(seq (dloop ?start ?n ?tile1 ?loop_var ?body1) (seq (dloop ?start ?n ?tile2 ?loop_var ?body2) ?others))" => 
            "(seq (dloop ?start ?n ?tile1 ?loop_var (seq ?body1 ?body2)) ?others)" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
                is_not_num(var("?tile1")),
                is_not_num(var("?tile2")),
            )
        ),
        rw!("dloop-fusion"; 
            "(seq (dloop ?start ?n ?tile1 ?loop_var ?body1) (dloop ?start ?n ?tile2 ?loop_var ?body2))" => 
            "(dloop ?start ?n ?tile1 ?loop_var (seq ?body1 ?body2))" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
                is_not_num(var("?tile1")),
                is_not_num(var("?tile2")),
            )
        ),

        rw!("loop-fusion-unified";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (loop ?start ?m ?tile2 ?loop_var2 ?body2))" =>
            {
                LoopFusion {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2")
                }
            }
        ),
        rw!("loop-fusion-unified-tail";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (seq (loop ?start ?m ?tile2 ?loop_var2 ?body2) ?others))" =>
            {
                LoopFusionTail {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2"), others: var("?others")
                }
            }
        ),

        rw!("loop-fission-tail"; 
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (seq (loop 0 ?n ?tile_n ?loop_var ?body2) ?others))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),
        rw!("loop-fission"; 
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (loop 0 ?n ?tile_n ?loop_var ?body2))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),

        rw!("loop-insertion1-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (seq ?body2 ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?body2"), var("?loop_var")),
                not_same_loop(var("?body2"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion1";
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) ?body2)" =>
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))"
            if and_all!(
                no_dependency_with_loopvar(var("?body2"), var("?loop_var")),
                not_same_loop(var("?body2"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_seq(var("?body2")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion2-tail";
            "(seq ?body1 (seq (loop 0 ?n ?tile_n ?loop_var ?body2) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?body1"), var("?loop_var")),
                not_same_loop(var("?body1"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion2";
            "(seq ?body1 (loop 0 ?n ?tile_n ?loop_var ?body2))" =>
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))"
            if and_all!(
                no_dependency_with_loopvar(var("?body1"), var("?loop_var")),
                not_same_loop(var("?body1"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-deletion";
            "(loop 0 ?n ?tile_n ?loop_var ?body)" =>
            "?body"
            if and_all! (
                no_dependency_with_loopvar(var("?body"), var("?loop_var")),
                // no_dummy(var("?body")),
            )
        ),
        rw!("loop-comm-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq
                        (store ?a (+ (* (load ?a ?idx) 1) ?val1) ?idx)
                        (store ?b (+ (* (load ?b ?idx) 1) ?val2) ?idx)))
            (seq (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx) ?others))"
            =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) 1) (+ ?val1 ?val2)) ?idx)) ?others)"
        ),
        rw!("loop-comm";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq
                        (store ?a (+ (* (load ?a ?idx) 1) ?val1) ?idx)
                        (store ?b (+ (* (load ?b ?idx) 1) ?val2) ?idx)))
                (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx))"
            =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) 1) (+ ?val1 ?val2)) ?idx))"
        ),

        // Index bug. b의 idx가 matmul을 하기 전과 후에 달라진다.
        // rw!("loop-factor-matmul";
        //     "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (@ ?val1 ?val2)) ?idx))" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //           (store ?b (* (load ?b ?idx) ?val2) ?idx))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-factor-matmul-tail";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (@ ?val1 ?val2)) ?idx)) ?others)" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //           (seq (store ?b (* (load ?b ?idx) ?val2) ?idx) ?others))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),

        rw!("loop-factor-div";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (/ ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
                  (store ?b (/ (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-div-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (/ ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
                  (seq (store ?b (/ (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-mul";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (* ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
                  (store ?b (* (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),
        rw!("loop-factor-mul-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (* ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
                  (seq (store ?b (* (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),

        rw!("loop-dist-matmul-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
             (seq (store ?c (* (load ?b ?idx) ?val2) ?idx2) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx2) ?accm) (@ ?val1 ?val2)) ?idx2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-matmul";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
              (store ?c (* (load ?b ?idx) ?val2) ?idx2))" =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx2) ?accm) (@ ?val1 ?val2)) ?idx2))" 
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-div-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
             (seq (store ?c (/ (load ?b ?idx) ?val2) ?idx) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (/ ?val1 ?val2)) ?idx)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-div";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
              (store ?c (/ (load ?b ?idx) ?val2) ?idx))" =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (/ ?val1 ?val2)) ?idx))" 
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-mul-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
             (seq (store ?c (* (load ?b ?idx) ?val2) ?idx) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (* ?val1 ?val2)) ?idx)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),
        rw!("loop-dist-mul";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
              (store ?c (* (load ?b ?idx) ?val2) ?idx))" =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (* ?val1 ?val2)) ?idx))" 
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),

        // rw!("loop-split";
        //     "(loop 0 ?end ?tile ?loop_var (store (input ?a) (+ (* (load (input ?a) ?idx) 1) ?body) ?idx))" =>
        //     { LoopSplit{
        //         end: var("?end"), tile: var("?tile"), loop_var: var("?loop_var"), a: var("?a"), idx: var("?idx"), body: var("?body"),
        //         new_tile: var("?new_tile"), new_loop_var: var("?new_loop_var"), new_a: var("?new_a"), others: var("?others"),
        //         rhs: "(seq 
        //         (dloop 0 ?end ?new_tile ?new_loop_var 
        //           (dloop ?new_loop_var (+ ?new_loop_var ?new_tile) ?tile ?loop_var (store (input ?new_a) (+ (* (load (input ?new_a) (index (elem ?new_loop_var) ?idx)) 1) ?body) (index (elem ?new_loop_var) ?idx))))
        //         (store (input ?a) (rsum (load (input ?new_a) (index (fulltile) ?idx)) 0) ?idx))".parse().unwrap(),
        //     }}
        //     if and_all!(
        //         has_dependency_with_loopvar(var("?body"),var("?loop_var"))
        //     )
        // ),
        // rw!("loop-split-tail";
        //     "(seq (loop 0 ?end ?tile ?loop_var (store (input ?a) (+ (* (load (input ?a) ?idx) 1) ?body) ?idx)) ?others)" =>
        //     { LoopSplitTail{
        //         end: var("?end"), tile: var("?tile"), loop_var: var("?loop_var"), a: var("?a"), idx: var("?idx"), body: var("?body"),
        //         new_tile: var("?new_tile"), new_loop_var: var("?new_loop_var"), new_a: var("?new_a"), others: var("?others"),
        //         rhs: "(seq 
        //         (loop 0 ?end ?new_tile ?new_loop_var 
        //           (loop ?new_loop_var (+ ?new_loop_var ?new_tile) ?tile ?loop_var (store (input ?new_a) (+ (* (load (input ?new_a) (index (elem ?new_loop_var) ?idx)) 1) ?body) (index (elem ?new_loop_var) ?idx))))
        //         (seq (store (input ?a) (rsum (load (input ?new_a) (index (fulltile) ?idx)) 0) ?idx) ?others))".parse().unwrap(),
        //     }}
        //     if and_all!(
        //         has_dependency_with_loopvar(var("?body"),var("?loop_var"))
        //     )
        // ),

        // algebraic transformation rules
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add2"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),

        rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rw!("assoc-mul2"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),

        rw!("assoc-matmul"; "(@ ?a (@ ?b ?c))" => "(@(@ ?a ?b) ?c)"),
        rw!("assoc-matmul2"; "(@(@ ?a ?b) ?c)" => "(@ ?a (@ ?b ?c))"),
        rw!("assoc-div-matmul"; "(@(/ ?a (bcast ?b ?axis)) ?c)" => "(/ (@ ?a ?c) (bcast ?b ?axis))"),

        rw!("dist-mul-add"; "(* ?a (+ ?b ?c))"        => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("dist-mul-sub"; "(* ?a (- ?b ?c))"        => "(- (* ?a ?b) (* ?a ?c))"),
        
        rw!("dist-matmul-add"; "(@ ?a (+ ?b ?c))"        => "(+ (@ ?a ?b) (@ ?a ?c))"),
        rw!("dist-matmul-sub"; "(@ ?a (- ?b ?c))"        => "(- (@ ?a ?b) (@ ?a ?c))"),

        rw!("factor-mul-add"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rw!("factor-mul-sub"    ; "(- (* ?a ?b) (* ?a ?c))" => "(* ?a (- ?b ?c))"),
        rw!("factor-matmul-add"    ; "(+ (@ ?a ?b) (@ ?a ?c))" => "(@ ?a (+ ?b ?c))"),
        rw!("factor-matmul-sub"    ; "(- (@ ?a ?b) (@ ?a ?c))" => "(@ ?a (- ?b ?c))"),
        
        rw!("exp-mul"; "(* (exp ?a) (exp ?b))" => "(exp (+ ?a ?b))"),
        rw!("exp-div"; "(/ (exp ?a) (exp ?b))" => "(exp (- ?a ?b))"),
        rw!("exp0"; "(exp 0)" => "1"),
        rw!("recip-mul-div"; "(* ?x (/ 1 ?x))" => "1" if is_not_zero(var("?x"))),

        rw!("geometry-of-concat"; "(concat (concat ?x ?z 0) (concat ?y ?w 0) 1)" => "(concat (concat ?x ?y 1) (concat ?z ?w 1) 0)"),
        rw!("geometry-of-concat-inv"; "(concat (concat ?x ?y 1) (concat ?z ?w 1) 0)" => "(concat (concat ?x ?z 0) (concat ?y ?w 0) 1)"),

        rw!("operator-comm6"; "(+ (concat ?x ?z ?a) (concat ?y ?w ?a))" => "(concat (+ ?x ?y) (+ ?z ?w) ?a)"),
        rw!("operator-comm6-inv"; "(concat (+ ?x ?y) (+ ?z ?w) ?a)" => "(+ (concat ?x ?z ?a) (concat ?y ?w ?a))"),

        rw!("operator-comm7"; "(* (concat ?x ?z ?a) (concat ?y ?w ?a))" => "(concat (* ?x ?y) (* ?z ?w) ?a)"),
        rw!("operator-comm7-inv"; "(concat (* ?x ?y) (* ?z ?w) ?a)" => "(* (concat ?x ?z ?a) (concat ?y ?w ?a))"),

        rw!("concat-and-matmul0"; "(@ ?x (concat ?y ?z 1))" => "(concat (@ ?x ?y) (@ ?x ?z) 1)"),
        rw!("concat-and-matmul0-inv"; "(concat (@ ?x ?y) (@ ?x ?z) 1)" => "(@ ?x (concat ?y ?z 1))"),

        rw!("concat-and-matmul1"; "(+ (@ ?a ?b) (@ ?c ?d))" => "(@(concat ?a ?c 1) (concat ?b ?d 0))"),
        rw!("concat-and-matmul1-inv"; "(@(concat ?a ?c 1) (concat ?b ?d 0))" => "(+ (@ ?a ?b) (@ ?c ?d))"),
        
    ]
}

struct LoopSplit {
    end: Var,
    tile: Var,
    loop_var: Var,
    a: Var,
    idx: Var,
    body: Var,
    new_tile: Var,
    new_loop_var: Var,
    new_a: Var,
    others: Var,
    rhs: Pattern<TileLang>,
}

impl Applier<TileLang, LoopAnalysis> for LoopSplit {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<TileLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut subst = subst.clone();

        // Generate new variable names based on existing ones
        fn rename_var(egraph: &mut EGraph, id: Id, prefix: &str) -> Id {
            for node in &egraph[id].nodes {
                if let TileLang::Var(sym) = node {
                    let new_sym = format!("{}_{}", prefix, sym);
                    return egraph.add(TileLang::Var(new_sym.into()));
                }
            }
            // Fallback if not a variable
            egraph.add(TileLang::Var(format!("{}_{}", prefix, id.as_usize()).into()))
        }

        let new_tile_id = rename_var(egraph, subst[self.tile], "new");
        let new_loop_var_id = rename_var(egraph, subst[self.loop_var], "new");
        let new_a_id = rename_var(egraph, subst[self.a], "new");

        // Insert the new variable bindings into the substitution
        subst.insert(self.new_tile, new_tile_id);
        subst.insert(self.new_loop_var, new_loop_var_id);
        subst.insert(self.new_a, new_a_id);

        // Apply the RHS pattern using the updated substitution
        self.rhs.apply_one(egraph, eclass, &subst, searcher_ast, rule_name)
    }
}

struct LoopSplitTail {
    end: Var,
    tile: Var,
    loop_var: Var,
    a: Var,
    idx: Var,
    body: Var,
    new_tile: Var,
    new_loop_var: Var,
    new_a: Var,
    others: Var,
    rhs: Pattern<TileLang>,
}

impl Applier<TileLang, LoopAnalysis> for LoopSplitTail {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<TileLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut subst = subst.clone();

        // Generate new variable names based on existing ones
        fn rename_var(egraph: &mut EGraph, id: Id, prefix: &str) -> Id {
            for node in &egraph[id].nodes {
                if let TileLang::Var(sym) = node {
                    let new_sym = format!("{}_{}", prefix, sym);
                    return egraph.add(TileLang::Var(new_sym.into()));
                }
            }
            // Fallback if not a variable
            egraph.add(TileLang::Var(format!("{}_{}", prefix, id.as_usize()).into()))
        }

        let new_tile_id = rename_var(egraph, subst[self.tile], "new");
        let new_loop_var_id = rename_var(egraph, subst[self.loop_var], "new");
        let new_a_id = rename_var(egraph, subst[self.a], "new");

        // Insert the new variable bindings into the substitution
        subst.insert(self.new_tile, new_tile_id);
        subst.insert(self.new_loop_var, new_loop_var_id);
        subst.insert(self.new_a, new_a_id);

        // Apply the RHS pattern using the updated substitution
        self.rhs.apply_one(egraph, eclass, &subst, searcher_ast, rule_name)
    }
}

struct LoopFusion {
    tile1: Var,
    tile2: Var,
    n: Var,
    m: Var,
    loop_var1: Var,
    loop_var2: Var,
    body1: Var,
    body2: Var,
    start: Var,
}

impl Applier<TileLang, LoopAnalysis> for LoopFusion {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<TileLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let start = subst[self.start];
        let n = subst[self.n];
        let m = subst[self.m];
        let tile1 = subst[self.tile1];
        let tile2 = subst[self.tile2];
        let loop_var1 = subst[self.loop_var1];
        let loop_var2 = subst[self.loop_var2];
        let body1 = subst[self.body1];
        let body2 = subst[self.body2];

        let no_dep = no_raw_dependency(self.body1, self.body2, self.loop_var1)(egraph, eclass, subst);

        let is_num_tile1 = is_num(self.tile1)(egraph, eclass, subst);
        let is_num_tile2 = is_num(self.tile2)(egraph, eclass, subst);

        let cond1_nm = cond1(self.n, self.tile1, self.m)(egraph, eclass, subst);
        let cond1_mn = cond1(self.m, self.tile2, self.n)(egraph, eclass, subst);

        let mut results = vec![];

        if !no_dep {
            return vec![];
        }

        if loop_var1 == loop_var2 && n == m {
            // same loop var, same range
            if !is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                results.push(egraph.add(fused));
            } else if is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                results.push(egraph.add(fused));
            } else if !is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile2, loop_var1, new_body]);
                results.push(egraph.add(fused));
            } else if is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                results.push(egraph.add(fused));
            }
        } else {
            // different loop vars or range
            if is_num_tile1 && !is_num_tile2 && cond1_nm {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, n, tile1, loop_var1, loop_var2, new_body]);
                results.push(egraph.add(fused));
            } else if !is_num_tile1 && is_num_tile2 && cond1_mn {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, m, tile2, loop_var2, loop_var1, new_body]);
                results.push(egraph.add(fused));
            }
        }

        for id in &results {
            egraph.union(eclass, *id);
        }

        results
    }
}

pub struct LoopFusionTail {
    start: Var,
    n: Var,
    m: Var,
    tile1: Var,
    tile2: Var,
    loop_var1: Var,
    loop_var2: Var,
    body1: Var,
    body2: Var,
    others: Var,
}
impl Applier<TileLang, LoopAnalysis> for LoopFusionTail {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<TileLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let start = subst[self.start];
        let n = subst[self.n];
        let m = subst[self.m];
        let tile1 = subst[self.tile1];
        let tile2 = subst[self.tile2];
        let loop_var1 = subst[self.loop_var1];
        let loop_var2 = subst[self.loop_var2];
        let body1 = subst[self.body1];
        let body2 = subst[self.body2];
        let others = subst[self.others];

        let no_dep = no_raw_dependency(self.body1, self.body2, self.loop_var1)(egraph, eclass, subst);
        let is_num_tile1 = is_num(self.tile1)(egraph, eclass, subst);
        let is_num_tile2 = is_num(self.tile2)(egraph, eclass, subst);
        let cond1_nm = cond1(self.n, self.tile1, self.m)(egraph, eclass, subst);
        let cond1_mn = cond1(self.m, self.tile2, self.n)(egraph, eclass, subst);

        let mut results = vec![];

        if !no_dep {
            return vec![];
        }

        if loop_var1 == loop_var2 && n == m {
            // same loop var, same range
            if !is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if is_num_tile1 && !is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if !is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile2, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if is_num_tile1 && is_num_tile2 {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::Loop([start, n, tile1, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            }
        } else {
            // different loop vars or range
            if is_num_tile1 && !is_num_tile2 && cond1_nm {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, n, tile1, loop_var1, loop_var2, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            } else if !is_num_tile1 && is_num_tile2 && cond1_mn {
                let seq = TileLang::Seq([body1, body2]);
                let new_body = egraph.add(seq);
                let fused = TileLang::TLoop([start, m, tile2, loop_var2, loop_var1, new_body]);
                let fused_with_tail = TileLang::Seq([egraph.add(fused), others]);
                results.push(egraph.add(fused_with_tail));
            }
        }

        for id in &results {
            egraph.union(eclass, *id);
        }

        results
    }
}

fn collect_access_sets(egraph: &EGraph, root: Id, need_source: bool) -> (Vec<Access>, Vec<Access>) {
    let mut read_set = vec![];
    let mut write_set = vec![];
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        if !visited.insert(id) {
            continue;
        }

        let data = &egraph[id].data;
        for enode in &egraph[id].nodes {
            if data.is_deleted.contains(enode) {
                continue;
            }

            match enode {
                TileLang::Load([base, idx]) => {
                    for base_node in &egraph[*base].nodes {
                        match base_node {
                            TileLang::Input(tensor_id) => {
                                for tensor_node in &egraph[*tensor_id].nodes {
                                    if let TileLang::Var(sym) = tensor_node {
                                        let base_name = sym.as_str().to_string();
                                        let idx_expr = extract_expr(egraph, *idx);
                                        let src = if need_source { Some(enode.clone()) } else { None };
                                        read_set.push(Access {
                                            base: Some(base_name),
                                            index: idx_expr,
                                        });
                                    }
                                }
                            }
                            TileLang::Output(tensor_id) => {
                                for tensor_node in &egraph[*tensor_id].nodes {
                                    if let TileLang::Var(sym) = tensor_node {
                                        let base_name = sym.as_str().to_string();
                                        let idx_expr = extract_expr(egraph, *idx);
                                        let src = if need_source { Some(enode.clone()) } else { None };
                                        read_set.push(Access {
                                            base: Some(base_name),
                                            index: idx_expr,
                                        });
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }

                TileLang::Store([base, _val, idx]) => {
                    for base_node in &egraph[*base].nodes {
                        match base_node {
                            TileLang::Input(tensor_id) => {
                                for tensor_node in &egraph[*tensor_id].nodes {
                                    if let TileLang::Var(sym) = tensor_node {
                                        let base_name = sym.as_str().to_string();
                                        let idx_expr = extract_expr(egraph, *idx);
                                        let src = if need_source { Some(enode.clone()) } else { None };
                                        write_set.push(Access {
                                            base: Some(base_name),
                                            index: idx_expr,
                                        });
                                    }
                                }
                            }
                            TileLang::Output(tensor_id) => {
                                for tensor_node in &egraph[*tensor_id].nodes {
                                    if let TileLang::Var(sym) = tensor_node {
                                        let base_name = sym.as_str().to_string();
                                        let idx_expr = extract_expr(egraph, *idx);
                                        let src = if need_source { Some(enode.clone()) } else { None };
                                        write_set.push(Access {
                                            base: Some(base_name),
                                            index: idx_expr,
                                        });
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }

                _ => {}
            }

            for &child in enode.children() {
                queue.push_back(child);
            }
        }
    }

    (read_set, write_set)
}

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

fn is_not_zero(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(0)))
    }
}
fn is_not_one(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(1)))
    }
}
fn is_not_num(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(_)))
    }
}
fn is_num(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Num(_)))
    }
}
fn cond1(n: Var, tile: Var, m: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        // Helper to extract a Num(_) value
        fn get_num(egraph: &EGraph, id: Id) -> Option<i32> {
            for node in &egraph[id].nodes {
                if let TileLang::Num(val) = node {
                    return Some(*val);
                }
            }
            None
        }

        let n_val = get_num(egraph, subst[n]);
        let tile_val = get_num(egraph, subst[tile]);
        let m_val = get_num(egraph, subst[m]);

        match (n_val, tile_val, m_val) {
            (Some(n), Some(tile), Some(m)) if tile != 0 => n / tile == m,
            _ => false,
        }
    }
}
fn no_const(var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let id = subst[var];
        let data = &egraph[id].data;

        !egraph[id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Const(_)))
    }
}

fn not_same_loop(
    body2: Var,
    n: Var,
    tile_n: Var,
    loop_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body2_id = subst[body2];
        let data = &egraph[body2_id].data;

        let expected_header = (
            subst[n],
            subst[tile_n],
            subst[loop_var],
        );

        for node in &egraph[body2_id].nodes {
            if data.is_deleted.contains(node) {
                continue; // Skip deleted nodes
            }

            if let TileLang::Loop([_, loop_n, loop_tile_n, loop_var_id, _body]) = node {
                let header = (*loop_n, *loop_tile_n, *loop_var_id);
                if header == expected_header {
                    return false;
                }
            }
        }

        true
    }
}

fn is_same_base(
    a_var: Var,
    b_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let a_id = subst[a_var];
        let b_id = subst[b_var];

        // Helper to extract the base symbol from Input(_) or Output(_)
        fn get_base_name(egraph: &EGraph, id: Id) -> Option<String> {
            for node in &egraph[id].nodes {
                match node {
                    TileLang::Input(base_id) | TileLang::Output(base_id) => {
                        for base_node in &egraph[*base_id].nodes {
                            if let TileLang::Var(sym) = base_node {
                                return Some(sym.as_str().to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            None
        }

        let a_base = get_base_name(egraph, a_id);
        let b_base = get_base_name(egraph, b_id);

        match (a_base, b_base) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

// body1과 body2가 전혀 관계 없음
fn no_all_dependency(
    body1_var: Var,
    body2_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body1_id = subst[body1_var];
        let body2_id = subst[body2_var];

        let (d1_reads, d1_writes) = collect_access_sets(egraph, body1_id, false);
        let (d2_reads, d2_writes) = collect_access_sets(egraph, body2_id, false);

        for r2 in &d2_reads {
            for w1 in &d1_writes {
                if r2.base == w1.base {
                    return false;
                }
            }
        }
        for w2 in &d2_writes {
            for r1 in &d1_reads {
                if w2.base == r1.base {
                    return false;
                }
            }
        }
        for w2 in &d2_writes {
            for w1 in &d1_writes {
                if w2.base == w1.base {
                    return false;
                }
            }
        }

        true
    }
}

// body의 read/write set이 loop_var의 영향을 전혀 받지 않음
fn no_dependency_with_loopvar(body: Var, loop_var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let loop_var_id = subst[loop_var];

        // extract the variable name string (e.g., "n")
        let loop_var_str = egraph[loop_var_id]
            .nodes
            .iter()
            .find_map(|n| {
                if let TileLang::Var(sym) = n {
                    Some(sym.as_str().to_string())
                } else {
                    None
                }
            })
            .expect("loop_var must be a Var");

        let (read_set, write_set) = collect_access_sets(egraph, body_id, false);

        for access in read_set.iter().chain(write_set.iter()) {
            if let Some(index) = &access.index {
                if index_depends_on(index, egraph, &loop_var_str) {
                    return false;
                }
            }
        }

        true
    }
}
fn has_dependency_with_loopvar(body: Var, loop_var: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let loop_var_id = subst[loop_var];

        // extract the variable name string (e.g., "n")
        let loop_var_str = egraph[loop_var_id]
            .nodes
            .iter()
            .find_map(|n| {
                if let TileLang::Var(sym) = n {
                    Some(sym.as_str().to_string())
                } else {
                    None
                }
            })
            .expect("loop_var must be a Var");

        let (read_set, write_set) = collect_access_sets(egraph, body_id, false);

        for access in read_set.iter().chain(write_set.iter()) {
            if let Some(index) = &access.index {
                if index_depends_on(index, egraph, &loop_var_str) {
                    return true;
                }
            }
        }

        false
    }
}

/// Returns true if the e-class has no Seq operator
fn no_seq(body: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let data = &egraph[body_id].data;

        !egraph[body_id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::Seq([_, _])))
    }
}
// fn no_dummy(body: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
//     move |egraph, _eclass, subst| {
//         let body_id = subst[body];
//         let data = &egraph[body_id].data;

//         !egraph[body_id]
//             .nodes
//             .iter()
//             // .filter(|n| !data.is_deleted.contains(n))
//             .any(|n| matches!(n, TileLang::NoneOp))
//     }
// }

// body1과 body2가 loopvar에 대해서 raw hazard가 없음
fn no_raw_dependency(
    body1_var: Var,
    body2_var: Var,
    loop_var: Var,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body1_id = subst[body1_var];
        let body2_id = subst[body2_var];

        let loop_var_str = egraph[subst[loop_var]]
            .nodes
            .iter()
            .find_map(|n| {
                if let TileLang::Var(sym) = n {
                    Some(sym.as_str().to_string())
                } else {
                    None
                }
            })
            .expect("loop_var should be a Var");

        let (d1_reads, d1_writes) = collect_access_sets(egraph, body1_id, false);
        let (d2_reads, _d2_writes) = collect_access_sets(egraph, body2_id, false);
        // println!("{:?}", d2_reads);
        // println!("{:?}", d1_writes);
        // println!("=================");

        for r2 in &d2_reads {
            for w1 in &d1_writes {
                if r2.base == w1.base {
                    if has_cross_iteration_dependency(w1, &d1_reads, &loop_var_str, egraph) {
                        // println!("Dependency!");
                        // println!("{:?}", w1);
                        // println!("{:?}", loop_var_str);
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn has_cross_iteration_dependency(
    write: &Access,
    read_set: &[Access],
    loop_var: &str,
    egraph: &EGraph,
) -> bool {

    if let Some(write_idx) = &write.index {
        if index_depends_on(write_idx, egraph, loop_var) {
            return false;
        }
    }

    for read in read_set {
        if let Some(read_idx) = &read.index {
            if index_depends_on(read_idx, egraph, loop_var) {
                return true;
            }
        }
    }

    false
}

fn index_depends_on(index: &TileLang, egraph: &EGraph, loop_var: &str) -> bool {
    match index {
        TileLang::Index(args) => {
            args.iter().any(|id| {
                egraph[*id].nodes.iter().any(|n| match n {
                    TileLang::FullTile => false, // no dependency
                    TileLang::Tile(tile_idx) => depends_on_id(egraph, *tile_idx, loop_var),
                    TileLang::Elem(tile_idx) => depends_on_id(egraph, *tile_idx, loop_var),
                    TileLang::Index(inner_args) => index_depends_on(n, egraph, loop_var),
                    _ => false, // not a tile structure
                })
            })
        }
        _ => false, // not an Index node
    }
}

fn expr_depends_on(egraph: &EGraph, expr: &TileLang, loop_var: &str) -> bool {
    match expr {
        TileLang::Var(sym) => sym.as_str() == loop_var,
        TileLang::Num(_) => false,
        TileLang::Add([a, b])
        | TileLang::Sub([a, b])
        | TileLang::Mul([a, b])
        | TileLang::Div([a, b])
        | TileLang::Matmul([a, b]) => {
            depends_on_id(egraph, *a, loop_var) || depends_on_id(egraph, *b, loop_var)
        }
        _ => false,
    }
}

fn depends_on_id(egraph: &EGraph, id: Id, loop_var: &str) -> bool {
    egraph[id]
        .nodes
        .iter()
        .any(|n| expr_depends_on(egraph, n, loop_var))
}

/// Flatten a nested seq structure into a list of elements 
fn flatten_seq(egraph: &EGraph, id: Id, out: &mut Vec<Id>) {
    let data = &egraph[id].data;

    for node in &egraph[id].nodes {
        if data.is_deleted.contains(node) {
            continue; // Skip deleted enodes
        }
        if !is_legal_seq_node(egraph, &node) {
            continue;
        }
        match node {
            TileLang::Seq([left, right]) => {
                flatten_seq(egraph, *left, out);
                flatten_seq(egraph, *right, out);
                return; // Done for this seq path
            }
            _ => {}
        }
    }

    out.push(id); // Not a seq: treat as leaf
}


// Returns all possible sequence arrangements from an e-class
fn flatten_seq_all_branch(egraph: &EGraph, id: Id, should_legal: bool) -> Vec<Vec<Id>> {
    let data = &egraph[id].data;
    let mut all_sequences = Vec::new();

    for node in &egraph[id].nodes {
        if data.is_deleted.contains(node) {
            continue;
        }
        
        if let TileLang::Seq([left, right]) = node {
            if should_legal && !is_legal_seq_node(egraph, node) {
                continue; // Skip illegal sequences
            }
            
            let left_seqs = flatten_seq_all_branch(egraph, *left, true);
            let right_seqs = flatten_seq_all_branch(egraph, *right, true);
            
            // Cartesian product of left and right sequences
            for left_seq in &left_seqs {
                for right_seq in &right_seqs {
                    let mut combined = left_seq.clone();
                    combined.extend(right_seq.clone());
                    all_sequences.push(combined);
                }
            }
        }
    }
    
    // If no sequences found, treat as leaf
    if all_sequences.is_empty() {
        all_sequences.push(vec![id]);
    }
    
    all_sequences
}

// fn retag_accesses(enode: &TileLang, accesses: &[Access]) -> Vec<Access> {
//     accesses
//         .iter()
//         .cloned()
//         .map(|mut a| {
//             a.source_node = enode.clone();
//             a
//         })
//         .collect()
// }

fn print_eclass(egraph: &EGraph, id: Id) {
    let class = &egraph[id];
    let data = &egraph[id].data;
    println!("EClass {} has {} enodes:", id, class.nodes.len());
    for node in &class.nodes {
        if data.is_deleted.contains(node) {
            continue;
        }
        println!("  {}", node);
    }
}

#[derive(Default)]
struct LoopAnalysis;


#[derive(Debug, Clone, Default)]
pub struct LoopData {
    read_set: Vec<Access>,
    write_set: Vec<Access>,
    pub is_tensor: bool, // None if not tensor, Some(shape) if tensor
    pub is_deleted: HashSet<TileLang>, // track deleted terms per eclass
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Access {
    pub base: Option<String>,
    pub index: Option<TileLang>,
}

impl Analysis<TileLang> for LoopAnalysis {
    type Data = LoopData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        assert_eq!(to.is_tensor, from.is_tensor, "Mismatched is_tensor flags during merge");
        let old_deleted_len = to.is_deleted.len();
        to.is_deleted.extend(from.is_deleted);
        to.read_set.extend(from.read_set);
        to.write_set.extend(from.write_set);

        DidMerge(
            // to.read_set.len() > old_read_len ||
            // to.write_set.len() > old_write_len ||
            // to.is_deleted.len() > old_deleted_len,
            // true,
            to.is_deleted.len() > old_deleted_len, true
            // true, true
        )
    }

    fn make(egraph: &mut EGraph, enode: &TileLang) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        match enode {
            TileLang::Tile(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            },
            TileLang::FullTile => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Dummy => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Elem(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Const(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Input(tensor) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Output(tensor) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Index(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Load([base, idx]) => {
                /*
                dim = get_dimension_from_index(base, idx)
                tensor_shape = dim
                */
                Self::Data {
                    is_tensor: true,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Store([base, val, idx]) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Loop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::TLoop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::DLoop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::SLoop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::PLoop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Seq(args) => {
            
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Add(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Sub(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Div(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Mul(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Matmul(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Concat(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Exp(arg) => {
                Self::Data {
                    is_tensor: true,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::ReduceSum(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Broadcast(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Permute3(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Squeeze(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Unsqueeze(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Var(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }
            TileLang::Num(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                    read_set: Vec::new(),
                    write_set: Vec::new(),
                }
            }

        }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        // ====================================
        // (1) Sequence flattening
        // ====================================
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            if egraph[id].data.is_deleted.contains(&node) {
                continue;
            }
        
            if is_legal_seq_node(egraph, &node) {
                continue; // there already exists legal sequence node
            }
            
            let mut new_forms = vec![];
            if let TileLang::Seq([left, right]) = node {
                // let all_sequences = flatten_seq_all_branch(egraph, id, false);
                // for seq_elements in all_sequences {
                //     if seq_elements.len() > 1 {
                //         let mut iter = seq_elements.into_iter().rev();
                //         let mut current = iter.next().unwrap();
                //         while let Some(prev) = iter.next() {
                //             current = egraph.add(TileLang::Seq([prev, current]));
                //         }
                //         new_forms.push(current);
                //     }
                // }
                // for form in new_forms {
                //     egraph.union(id, form);
                // }
                // // egraph[id].data.is_deleted.insert(node);

                
                // Step 1: Flatten recursively
                let mut seq_elements = vec![];
                flatten_seq(egraph, left, &mut seq_elements);
                flatten_seq(egraph, right, &mut seq_elements);

                if seq_elements.is_empty() {
                    continue;
                }
    
                // Step 2: Rebuild canonical nested structure
                let mut iter = seq_elements.into_iter().rev();
    
                let mut current = iter.next().unwrap(); // last element (no seq_end)
    
                while let Some(prev) = iter.next() {
                    current = egraph.add(TileLang::Seq([prev, current]));
                }
                new_forms.push(current);

                for form in new_forms {
                    egraph.union(id, form);
                }

            }
        }

        // ====================================
        // (2) Value Forwarding
        // ====================================
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            if egraph[id].data.is_deleted.contains(&node) {
                continue;
            }

            if !is_legal_seq_node(egraph, &node) {
                continue;
            }
            let TileLang::Seq([left, right]) = node else {
                continue; // If not a Seq node, skip
            };

            // if contains_loop(egraph, left) || is_reduction_op(egraph, left){
            //     continue; // If the first op is loop, skip
            // }
            if is_reduction_op(egraph, left) {
                continue; // If the first op is loop, skip
            }
            // flatten the sequences
            let mut seq_elements = vec![];
            flatten_seq(egraph, left, &mut seq_elements);
            flatten_seq(egraph, right, &mut seq_elements);
            
            if seq_elements.len() < 2 {
                continue;
            }
            
            // println!("            try memory forwarding at {:?}", seq_elements);

            let mut iter = seq_elements.into_iter();
            let mut op1 = iter.next().unwrap();

            // op1 must be (store A, val, idx)
            let Some((base_name, idx_expr, val_id)) = extract_store_info(egraph, op1) else {
                continue;
            };

            // println!("{:?}", base_name);
            // println!("{:?}", idx_expr);
            
            for op_i in iter {
                // if op_i is loop or reduction op, skip
                if contains_loop(egraph, op_i) {
                    continue; 
                }

                // If op_i.readset.base != basename, skip
                let (op_read_set, _) = collect_access_sets(egraph, op_i, false);
                let dependent = op_read_set.iter().any(|access| {
                    access.base.as_ref() == Some(&base_name)
                });
                if !dependent {
                    // println!("                 No dependency");
                    continue;
                }
                
                // println!("            try memory forwarding at {:?}", op_i);

                // Walk subgraph rooted at op_i to find all reachable e-classes
                let subgraph = collect_reachable_eclasses(egraph, op_i);

                // Find all (load A idx) eclass from the subgraph
                let mut load_targets = vec![];
                for &eclass_id in &subgraph {
                    let data = &egraph[eclass_id].data;
                    for enode in &egraph[eclass_id].nodes {
                        if data.is_deleted.contains(&enode) {
                            continue;
                        }
                        if let TileLang::Load([base_id, load_idx_id]) = enode {

                            let base_match = egraph[*base_id].nodes.iter().any(|n| {
                                match n {
                                    TileLang::Input(tensor_id) | TileLang::Output(tensor_id) => {
                                        egraph[*tensor_id].nodes.iter().any(|tn| {
                                            matches!(tn, TileLang::Var(sym) if sym.as_str() == base_name)
                                        })
                                    }
                                    _ => false,
                                }
                            });
                            
                            let idx_match = extract_expr(egraph, *load_idx_id).as_ref() == Some(&idx_expr);
                            
                            // println!("            find load node {:?}, {:?}", base_id, load_idx_id);
                            // println!("            base match: {:?}, idx match: {:?}", base_match, idx_match);
    
                            if base_match && idx_match {
                                load_targets.push(eclass_id);
                            }
                        }
                    }
                }

                // println!("            find load target {:?}", load_targets);

                // Replace all parent nodes that use the load with the value
                for &eclass_id in &subgraph {
                    let enodes = egraph[eclass_id].nodes.clone();
                    for enode in enodes {
                        if egraph[eclass_id].data.is_deleted.contains(&enode) {
                            continue;
                        }
                        let mut new_children = enode.children().to_vec();
                        let mut changed = false;
    
                        for i in 0..new_children.len() {
                            if load_targets.contains(&new_children[i]) {
                                new_children[i] = val_id;
                                changed = true;
                            }
                        }
    
                        if changed {
                            if let Ok(new_node) = TileLang::from_op(&enode.to_string(), new_children) {
                                let new_id = egraph.add(new_node);
                                egraph.union(eclass_id, new_id); // ✅ merge into same eclass
                                // print_eclass(egraph, val_id);
                                // println!("{:?}", extract_expr(egraph, val_id).as_ref());
                                // egraph[eclass_id].data.is_deleted.insert(enode.clone()); // ✅ Mark the old enode as deleted
                            }
                        }
                    }
                }
            }
        }

        // ====================================
        // (3) Resolve tmp_loop
        // ====================================
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            // if egraph[id].data.is_deleted.contains(&node) {
            //     continue;
            // }

            if let TileLang::TLoop([start, end, tile, loop_var1_id, loop_var2_id, body_id]) = node {
                let has_loop_already = egraph[id]
                    .nodes
                    .iter()
                    .any(|n| matches!(n, TileLang::Loop([s, e, t, lv, b])
                        if *s == start && *e == end && *t == tile && *lv == loop_var1_id && *b == body_id
                    ));

                if !has_loop_already {
                    // Insert new Loop(...) node into the current eclass
                    let loop_node = TileLang::Loop([start, end, tile, loop_var1_id, body_id]);
                    let loop_id = egraph.add(loop_node);
                    egraph.union(id, loop_id);
                } else {
                    continue;
                }

                let reachable = collect_reachable_eclasses(egraph, body_id);

                for &eclass_id in &reachable {
                    let enodes = egraph[eclass_id].nodes.clone();
                    for enode in enodes {
                        if let TileLang::Tile(arg_id) = enode {
                            let loop_var2_sym = get_var_symbol(egraph, loop_var2_id);

                            let is_loop_var2 = egraph[arg_id]
                                .nodes
                                .iter()
                                .any(|n| match (n, &loop_var2_sym) {
                                    (TileLang::Var(sym), Some(expected)) => sym == *expected,
                                    _ => false,
                                });
                            if is_loop_var2 {
                                let Some(loop_var1_sym) = get_var_symbol(egraph, loop_var1_id) else {
                                    continue;
                                };
                                let loop_var1_node = TileLang::Var(loop_var1_sym.clone());
                                let loop_var1_eclass = egraph.add(loop_var1_node);

                                let elem_node = TileLang::Elem(loop_var1_eclass);
                                let elem_id = egraph.add(elem_node);

                                egraph.union(eclass_id, elem_id);
                            }
                        }
                    }
                }

                // egraph[id].data.is_deleted.insert(node);
            }
        }

    }
}

fn postprocess_egraph(egraph: &mut EGraph) {
    /*
    Step1) Finally resolve all illegal sequences
    */
    let class_ids: Vec<Id> = egraph.classes().map(|class| class.id).collect();
    
    for id in class_ids {
        let nodes = egraph[id].nodes.clone();
        for node in nodes {
            if is_legal_seq_node(egraph, &node) {
                continue; // there already exists legal sequence node
            }
            
            let mut new_forms = vec![];
            if let TileLang::Seq([left, right]) = node {
                let mut seq_elements = vec![];
    
                // Step 1: Flatten recursively
                flatten_seq(egraph, left, &mut seq_elements);
                flatten_seq(egraph, right, &mut seq_elements);

                if seq_elements.is_empty() {
                    continue;
                }
    
                // Step 2: Rebuild canonical nested structure
                let mut iter = seq_elements.into_iter().rev();
    
                let mut current = iter.next().unwrap(); // last element (no seq_end)
    
                while let Some(prev) = iter.next() {
                    current = egraph.add(TileLang::Seq([prev, current]));
                }
                new_forms.push(current);

                for form in new_forms {
                    egraph.union(id, form);
                }

                // egraph[id].data.is_deleted.insert(node);
            }
        }
    }

    let class_ids: Vec<Id> = egraph.classes().map(|class| class.id).collect();
    for id in class_ids {
        let nodes = egraph[id].nodes.clone();
        let mut legal = false;
        for node in nodes {
            if let TileLang::Seq([left, right]) = node {
                if is_legal_seq_node(egraph, &node) {
                    legal = true;
                }
            } else {
                legal = true;
            }
        }
        if !legal {
            println!("Still illegal, {}", id);
            print_eclass(egraph, id);
        }
    }
    
    egraph.rebuild();
}

fn postprocess(input: &str) -> RecExpr<TileLang> {
    /*
        1. Parse input AST string into RecExpr format
        2. Deadcode elimination
            2-1) From the root, traverse entire AST
            2-2) When meeting Load(base, _) operator, insert base to read_set
            2-3) Traverse the entire AST from root againg
            2-4) When meeting Store(base, _, _) operator, do the following
                if base doesn't exist in read_set and base is not Output(_) operator,
                Substitute the Store() operator to Dummy() operator
        3. Decide parallel/sequential loop
            3-1) From the root, traverse entire AST
            3-2) for all Loop() operator, do the following
                if loop body has cross iteration dependency, substitute to SLoop()
                else, substitute to PLoop()

                if loop body is Seq() operator, substitute all reachable Loop() operator from the loop body to SLoop()
            
    */
    let mut expr: RecExpr<TileLang> = input.parse().unwrap();
    expr = deadcode_elimination(expr);
    expr = decide_loop_types(expr);
    expr
}

fn deadcode_elimination(mut expr: RecExpr<TileLang>) -> RecExpr<TileLang> {
    let mut read_set = HashSet::new();
    let root_idx = expr.as_ref().len() - 1;
    collect_read_set(&expr, root_idx, &mut read_set);

    eliminate_dead_stores(&mut expr, root_idx, &read_set);

    expr
}

fn collect_read_set(expr: &RecExpr<TileLang>, node_idx: usize, read_set: &mut HashSet<String>) {
    if node_idx >= expr.as_ref().len() {
        return;
    }
    
    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Load([base_id, _]) => {
            if let Some(base_name) = get_base_name(expr, usize::from(*base_id)) {
                read_set.insert(base_name);
            }
        }
        _ => {
            for &child_id in node.children() {
                collect_read_set(expr, usize::from(child_id), read_set);
            }
        }
    }
}

fn eliminate_dead_stores(expr: &mut RecExpr<TileLang>, node_idx: usize, read_set: &HashSet<String>) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = expr.as_ref()[node_idx].clone();

    match node {
        TileLang::Store([base_id, val_id, idx_id]) => {
            if let Some(base_name) = get_base_name(expr, usize::from(base_id)) {
                let is_read = read_set.contains(&base_name);
                let is_output = is_output_base(expr, usize::from(base_id));

                if !is_read && !is_output {
                    expr.as_mut()[node_idx] = TileLang::Dummy;
                    return;
                }
            }
        }
        _ => {}
    }
    
    for &child_id in node.children() {
        eliminate_dead_stores(expr, usize::from(child_id), read_set);
    }
}

fn decide_loop_types(mut expr: RecExpr<TileLang>) -> RecExpr<TileLang> {
    let root_idx = expr.as_ref().len() - 1;
    decide_loop_types_recursive(&mut expr, root_idx, false);
    expr
}


fn decide_loop_types_recursive(expr: &mut RecExpr<TileLang>, node_idx: usize, force_sequential: bool) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = expr.as_ref()[node_idx].clone();

    match node {
        TileLang::Loop([start, end, tile, loop_var, body]) => {
            let has_loop_carried_dep = has_loop_carried_dependency(expr, usize::from(body), usize::from(loop_var));

            let body_has_seq = contains_seq_operator(expr, usize::from(body));

            if has_loop_carried_dep || force_sequential {
                expr.as_mut()[node_idx] = TileLang::SLoop([start, end, tile, loop_var, body]);
            } else {
                expr.as_mut()[node_idx] = TileLang::PLoop([start, end, tile, loop_var, body]);
            }

            if body_has_seq {
                decide_loop_types_recursive(expr, usize::from(body), true);
            } else {
                decide_loop_types_recursive(expr, usize::from(body), false);
            }
        }
        _ => {
            for &child_id in node.children() {
                decide_loop_types_recursive(expr, usize::from(child_id), force_sequential);
            }
        }
    }
}

fn contains_seq_operator(expr: &RecExpr<TileLang>, node_idx: usize) -> bool {
    if node_idx >= expr.as_ref().len() {
        return false;
    }
    
    let node = &expr.as_ref()[node_idx];
    
    match node {
        TileLang::Seq(_) => true,
        _ => {
            // Recursively check children
            for &child_id in node.children() {
                if contains_seq_operator(expr, usize::from(child_id)) {
                    return true;
                }
            }
            false
        }
    }
}

fn has_loop_carried_dependency(expr: &RecExpr<TileLang>, body_idx: usize, loop_var_idx: usize) -> bool {
    let loop_var_name = get_var_name(expr, loop_var_idx);
    if loop_var_name.is_none() {
        return false;
    }

    // 1. collect all write access from the body
    let mut read_accesses = Vec::new();
    let mut write_accesses = Vec::new();
    collect_memory_accesses(expr, body_idx, &mut read_accesses, &mut write_accesses);

    // 2. check whether the index of write access contains loop_var or not.
    for write_access in &write_accesses {
        if let Some(index) = &write_access.index {
            if !index_involves_loop_var(expr, index, &loop_var_name.as_ref().unwrap()) {
                // 3. if contain loop_var, return false (meaning has dependency -> sloop)
                return true;
            }
        }
    }

    // if no write access contains loop_var, return true (meaning no dependency -> ploop)
    false
}

fn involves_loop_variable_dependency(index1: &Option<TileLang>, index2: &Option<TileLang>, loop_var: &str, expr: &RecExpr<TileLang>) -> bool {
    match (index1, index2) {
        (Some(idx1), Some(idx2)) => {
            // Check if either index involves the loop variable in a dependency-creating way
            index_involves_loop_var(expr, idx1, loop_var) && index_involves_loop_var(expr, idx2, loop_var)
        }
        _ => false,
    }
}

fn index_involves_loop_var(expr: &RecExpr<TileLang>, index: &TileLang, loop_var: &str) -> bool {
    match index {
        TileLang::Index(args) => {
            args.iter().any(|id| {
                let node_idx = usize::from(*id);
                if node_idx < expr.as_ref().len() {
                    let node = &expr.as_ref()[node_idx];
                    match node {
                        TileLang::FullTile => false, // no dependency
                        TileLang::Tile(tile_idx) => depends_on_id_recexpr(expr, *tile_idx, loop_var),
                        TileLang::Elem(tile_idx) => depends_on_id_recexpr(expr, *tile_idx, loop_var),
                        TileLang::Index(_) => index_involves_loop_var(expr, node, loop_var),
                        _ => false, // not a tile structure
                    }
                } else {
                    false
                }
            })
        }
        _ => false, // not an Index node
    }
}

fn expr_depends_on_recexpr(expr: &RecExpr<TileLang>, node: &TileLang, loop_var: &str) -> bool {
    match node {
        TileLang::Var(sym) => sym.as_str() == loop_var,
        TileLang::Num(_) => false,
        TileLang::Add([a, b])
        | TileLang::Sub([a, b])
        | TileLang::Mul([a, b])
        | TileLang::Div([a, b])
        | TileLang::Matmul([a, b]) => {
            depends_on_id_recexpr(expr, *a, loop_var) || depends_on_id_recexpr(expr, *b, loop_var)
        }
        TileLang::Exp(a) => depends_on_id_recexpr(expr, *a, loop_var),
        TileLang::ReduceSum([a, b]) => {
            depends_on_id_recexpr(expr, *a, loop_var) || depends_on_id_recexpr(expr, *b, loop_var)
        }
        TileLang::Tile(tile_idx) => depends_on_id_recexpr(expr, *tile_idx, loop_var),
        TileLang::Elem(tile_idx) => depends_on_id_recexpr(expr, *tile_idx, loop_var),
        TileLang::Index(_) => index_involves_loop_var(expr, node, loop_var),
        _ => false,
    }
}

fn depends_on_id_recexpr(expr: &RecExpr<TileLang>, id: Id, loop_var: &str) -> bool {
    let node_idx = usize::from(id);
    if node_idx < expr.as_ref().len() {
        let node = &expr.as_ref()[node_idx];
        expr_depends_on_recexpr(expr, node, loop_var)
    } else {
        false
    }
}

fn collect_memory_accesses(
    expr: &RecExpr<TileLang>,
    node_idx: usize,
    reads: &mut Vec<Access>,
    writes: &mut Vec<Access>,
) {
    if node_idx >= expr.as_ref().len() {
        return;
    }

    let node = &expr.as_ref()[node_idx];

    match node {
        TileLang::Load([base_id, idx_id]) => {
            let base = get_base_name(expr, usize::from(*base_id));
            let index = get_index_node(expr, usize::from(*idx_id));
            reads.push(Access{base, index});
        },
        TileLang::Store([base_id, _, idx_id]) => {
            let base = get_base_name(expr, usize::from(*base_id));
            let index = get_index_node(expr, usize::from(*idx_id));
            writes.push(Access{base, index});
        }
        _ => {}
    }

    for &child_id in node.children() {
        collect_memory_accesses(expr, usize::from(child_id), reads, writes);
    }
}

fn base_overlap(access1: &Access, access2: &Access) -> bool {
    match (&access1.base, &access2.base) {
        (Some(base1), Some(base2)) => base1 == base2,
        _ => false,
    }
}

fn get_base_name(expr: &RecExpr<TileLang>, node_idx: usize) -> Option<String> {
    if node_idx >= expr.as_ref().len() {
        return None;
    }

    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Input(tensor_id) | TileLang::Output(tensor_id) => {
            get_var_name(expr, usize::from(*tensor_id))
        },
        TileLang::Var(symbol) => Some(symbol.to_string()),
        _ => None,
    }
}

fn get_index_node(expr: &RecExpr<TileLang>, node_idx: usize) -> Option<TileLang> {
    if node_idx >= expr.as_ref().len() {
        return None;
    }
    
    Some(expr.as_ref()[node_idx].clone())
}


fn get_var_name(expr: &RecExpr<TileLang>, node_idx: usize) -> Option<String> {
    if node_idx >= expr.as_ref().len() {
        return None;
    }

    let node = &expr.as_ref()[node_idx];
    match node {
        TileLang::Var(symbol) => Some(symbol.to_string()),
        _ => None,
    }
}

fn is_output_base(expr: &RecExpr<TileLang>, node_idx: usize) -> bool {
    if node_idx >= expr.as_ref().len() {
        return false;
    }

    let node = &expr.as_ref()[node_idx];
    matches!(node, TileLang::Output(_))
}


fn get_var_symbol(egraph: &EGraph, id: Id) -> Option<&egg::Symbol> {
    for node in &egraph[id].nodes {
        if let TileLang::Var(sym) = node {
            return Some(sym);
        }
    }
    None
}


/// Check if there exists a legal Seq tree in this eclass
// fn has_legal_seq(egraph: &EGraph, id: Id) -> bool {
//     for node in &egraph[id].nodes {
//         if egraph[id].data.is_deleted.contains(&node) {
//             continue;
//         }    
//         if let TileLang::Seq([left, right]) = node {
//             if is_legal_seq_tree_strict(egraph, *left, *right) {
//                 return true;
//             }
//         }
//     }
//     false
// }

/// Checks whether a single Seq enode is a legal sequence tree
fn is_legal_seq_node(egraph: &EGraph, node: &TileLang) -> bool {
    if let TileLang::Seq([left, right]) = node {
        is_legal_seq_tree_strict(egraph, *left, *right)
    } else {
        false
    }
}
fn is_legal_seq_node_soft(egraph: &EGraph, node: &TileLang) -> bool {
    if let TileLang::Seq([left, right]) = node {
        is_legal_seq_tree_soft(egraph, *left, *right)
    } else {
        false
    }
}

/// Recursively check if a given (left, right) sequence is legal
fn is_legal_seq_tree_strict(egraph: &EGraph, left: Id, right: Id) -> bool {
    // Left must NOT be a Seq
    let left_data = &egraph[left].data;
    let left_is_seq = egraph[left]
        .nodes
        .iter()
        .filter(|n| !left_data.is_deleted.contains(n))
        .any(|n| matches!(n, TileLang::Seq([_, _])));
    if left_is_seq {
        // println!("Left is seq!");
        return false;
    }

    // Right must be either a Seq (legal) or a leaf
    for node in &egraph[right].nodes {
        if egraph[right].data.is_deleted.contains(&node) {
            continue;
        }
        match node {
            TileLang::Seq([rleft, rright]) => {
                return is_legal_seq_tree_strict(egraph, *rleft, *rright);
            }
            _ => {} // leaf, fine
        }
    }

    true
}

fn is_legal_seq_tree_soft(egraph: &EGraph, left: Id, right: Id) -> bool {
    // Left must have at least one NON-Seq node
    let left_data = &egraph[left].data;
    let left_has_non_seq = egraph[left]
        .nodes
        .iter()
        .filter(|n| !left_data.is_deleted.contains(n))
        .any(|n| !matches!(n, TileLang::Seq([_, _])));
    
    if !left_has_non_seq {
        return false;
    }

    // Right must be either a Seq (legal) or a leaf
    for node in &egraph[right].nodes {
        if egraph[right].data.is_deleted.contains(&node) {
            continue;
        }
        match node {
            TileLang::Seq([rleft, rright]) => {
                return is_legal_seq_tree_soft(egraph, *rleft, *rright);
            }
            _ => {} // leaf, fine
        }
    }

    true
}

/// Check whether there exists same access in readset and writeset
fn is_reduction_op(egraph: &EGraph, id: Id) -> bool {
    let (reads, writes) = collect_access_sets(egraph, id, false);

    for read in &reads {
        for write in &writes {
            if read.base == write.base {
                return true;
            }
        }
    }

    false
}
use egg::{RecExpr, Id};


fn collect_reachable_eclasses(egraph: &EGraph, root: Id) -> HashSet<Id> {
    let mut visited = HashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        if visited.insert(id) {
            let data = &egraph[id].data;
            for enode in &egraph[id].nodes {
                if data.is_deleted.contains(&enode) {
                    continue;
                }
                for &child in enode.children() {
                    queue.push_back(child);
                }
            }
        }
    }
    visited
}
fn contains_loop(egraph: &EGraph, id: Id) -> bool {
    let data = &egraph[id].data;
    egraph[id]
        .nodes
        .iter()
        .filter(|n| !data.is_deleted.contains(n))
        .any(|n| matches!(n, TileLang::Loop(_)))
}
fn extract_store_info(
    egraph: &EGraph,
    id: Id,
) -> Option<(String, TileLang, Id)> {
    let data = &egraph[id].data;
    for enode in &egraph[id].nodes {
        if data.is_deleted.contains(&enode) {
            continue;
        }
        if let TileLang::Store([base_id, val_id, idx_id]) = enode {
            // base_id should point to TileLang::Input(Var(sym))
            for base_node in &egraph[*base_id].nodes {
                match base_node {
                    TileLang::Input(tensor_id) | TileLang::Output(tensor_id) => {
                        for tensor_node in &egraph[*tensor_id].nodes {
                            if let TileLang::Var(sym) = tensor_node {
                                let base_name = sym.as_str().to_string();
                                let idx_expr = extract_expr(egraph, *idx_id)?;
                                return Some((base_name, idx_expr, *val_id));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    None
}
// fn extract_expr(egraph: &EGraph, id: Id) -> Option<RecExpr<TileLang>> {
//     let extractor = Extractor::new(egraph, AstSize);
//     let (best_cost, best_expr) = extractor.find_best(id);
//     Some(best_expr)
// }
fn extract_expr(egraph: &EGraph, id: Id) -> Option<TileLang> {
    egraph[id].nodes.iter().next().cloned()
}

// === Custom Searcher & Rewrite Macro ===
use std::sync::Arc;

/// A wrapper searcher that skips deleted nodes
// pub struct DeletedFilteredSearcher {
//     pattern: Pattern<TileLang>,
// }

// impl DeletedFilteredSearcher {
//     pub fn new(pattern: Pattern<TileLang>) -> Self {
//         Self { pattern }
//     }
// }

// impl Searcher<TileLang, LoopAnalysis> for DeletedFilteredSearcher {
//     fn search(&self, egraph: &EGraph) -> Vec<SearchMatches<TileLang>> {
//         self.pattern
//             .search(egraph)
//             .into_iter()
//             .map(|mut matches| {
//                 matches.substs.retain(|subst| {
//                     self.vars().iter().all(|&var| {
//                         let id = subst[var];
//                         let data = &egraph[id].data;
//                         egraph[id]
//                             .nodes
//                             .iter()
//                             .any(|n| !data.is_deleted.contains(n))
//                     })
//                 });
//                 matches
//             })
//             .filter(|matches| !matches.substs.is_empty())
//             .collect()
//     }

//     fn search_eclass_with_limit(
//         &self,
//         egraph: &EGraph,
//         eclass: Id,
//         limit: usize,
//     ) -> Option<SearchMatches<TileLang>> {
//         let mut matches = self.pattern.search_eclass_with_limit(egraph, eclass, limit)?;

//         matches.substs.retain(|subst| {
//             self.vars().iter().all(|&var| {
//                 let id = subst[var];
//                 let data = &egraph[id].data;
//                 egraph[id]
//                     .nodes
//                     .iter()
//                     .any(|n| !data.is_deleted.contains(n))
//             })
//         });

//         if matches.substs.is_empty() {
//             None
//         } else {
//             Some(matches)
//         }
//     }

//     fn vars(&self) -> Vec<egg::Var> {
//         self.pattern.vars()
//     }
// }


// #[macro_export]
// macro_rules! rw {
//     (
//         $name:expr;
//         $lhs:tt => $rhs:tt
//         $(if $cond:expr)*
//     ) => {{
//         // Use your DeletedFilteredSearcher wrapper
//         let searcher = $crate::DeletedFilteredSearcher::new($crate::__rewrite!(@parse Pattern $lhs));
//         let core_applier = $crate::__rewrite!(@parse Pattern $rhs);
//         let applier = $crate::__rewrite!(@applier core_applier; $($cond,)*);
//         $crate::Rewrite::new($name.to_string(), searcher, applier).unwrap()
//     }};
// }

// #[doc(hidden)]
// #[macro_export]
// macro_rules! __rewrite {
//     (@parse $t:ident $expr:literal) => {
//         $expr.parse::<$crate::$t<_>>().unwrap()
//     };
//     (@parse $t:ident $expr:expr) => { $expr };

//     (@applier $applier:expr;) => { $applier };
//     (@applier $applier:expr; $cond:expr, $($conds:expr,)*) => {
//         $crate::ConditionalApplier {
//             condition: $cond,
//             applier: $crate::__rewrite!(@applier $applier; $($conds,)*)
//         }
//     };
// }

#[macro_export]
macro_rules! and_all {
    ($($cond:expr),+ $(,)?) => {{
        move |egraph: &mut EGraph, id: Id, subst: &Subst| {
            true $( && $cond(egraph, id, subst) )+
        }
    }};
}

// fn contains_expr(egraph: &EGraph, root: Id, expr: &RecExpr<TileLang>) -> bool {
//     lookup_expr_filtered(egraph, expr).map_or(false, |id| {
//         // Ensure the id is reachable from root
//         is_reachable_from(egraph, root, id)
//     })
// }

fn is_reachable_from(egraph: &EGraph, root: Id, target: Id) -> bool {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(root);
    while let Some(id) = queue.pop_front() {
        if id == target {
            return true;
        }
        if !visited.insert(id) {
            continue;
        }
        for enode in &egraph[id].nodes {
            for &child in enode.children() {
                queue.push_back(child);
            }
        }
    }
    false
}

// fn dead_code_elimination_pass(egraph: &mut EGraph, root: Id) {
//     use std::collections::{HashSet, VecDeque};

//     let mut visited = HashSet::new();
//     let mut queue = VecDeque::new();
//     let mut read_bases = HashSet::new();
//     let mut write_info = Vec::new();

//     queue.push_back(root);

//     while let Some(id) = queue.pop_front() {
//         if !visited.insert(id) {
//             continue;
//         }

//         let eclass = &egraph[id];
//         for enode in &eclass.nodes {
//             // if eclass.data.is_deleted.contains(enode) {
//             //     // println!("{:?}", eclass.data.is_deleted);
//             //     continue;
//             // }

//             match enode {
//                 TileLang::Load([base, _]) => {
//                     read_bases.insert(*base);
//                 }
//                 TileLang::Store([base, _, _]) => {
//                     write_info.push((*base, id, enode.clone()));
//                 }
//                 _ => {}
//             }

//             for &child in enode.children() {
//                 queue.push_back(child);
//             }
//         }
//     }

//     // println!("Read set");
//     // for &read_base in &read_bases {
//     //     print_eclass(egraph, egraph[read_base].nodes[0].children()[0]);
//     // }
//     // println!("Write set");
//     // for (wbase, _, _) in &write_info {
//     //     print_eclass(egraph, egraph[*wbase].nodes[0].children()[0]);
//     // }
//     // println!("-------------------------");

//     // println!("{:?}", read_bases.clone());
//     // println!("{:?}", write_info.clone());

//     // Mark dead stores
//     for (wbase, eclass_id, enode) in write_info {
//         let is_output = egraph[wbase].nodes.iter().any(|n| matches!(n, TileLang::Output(_)));
//         if is_output {
//             continue;
//         }
//         if !read_bases.contains(&wbase) {
//             // egraph[eclass_id].data.is_deleted.insert(enode.clone());
//             // print_eclass(egraph, egraph[wbase].nodes[0].children()[0]);
//             // println!("{:?}", enode);

//             // let all_deleted = egraph[eclass_id].nodes.iter().all(|n| egraph[eclass_id].data.is_deleted.contains(n));
//             let all_deleted = true;
//             if all_deleted {
//                 let dummy = TileLang::NoneOp;
//                 let new_id = egraph.add(dummy);
//                 egraph.union(eclass_id, new_id);
//             }
//         }
//     }

//     egraph.rebuild();
// }

fn run_until_saturated(
    expr: &str,
    rules: Vec<Rewrite<TileLang, LoopAnalysis>>,
) -> Runner<TileLang, LoopAnalysis> {
    let parsed_expr: RecExpr<TileLang> = expr.parse().unwrap();

    let mut runner = Runner::default()
        .with_expr(&parsed_expr)
        .with_iter_limit(max_iter)
        .with_node_limit(1000000)
        .with_time_limit(std::time::Duration::from_secs(1000))
        // .with_hook(|runner| {
        //     for &root in &runner.roots {
        //         dead_code_elimination_pass(&mut runner.egraph, root);
        //     }
        //     Ok(())
        // })
        .run(&rules);
    // for &root in &runner.roots {
    //     dead_code_elimination_pass(&mut runner.egraph, root);
    // }
    // runner.egraph.rebuild();

    println!("📦 E-classes: {}", runner.egraph.number_of_classes());
    println!("🧱 E-nodes:   {}", runner.egraph.total_size());
    println!(" Iterations: {:?}", runner.iterations.len());

    runner
}


// Your TileLang definition would go here
// (I'm assuming it's already defined in your codebase)

// fn enumerate_all_expressions(
//     egraph: &EGraph,
//     eclass_id: Id,
// ) -> Vec<String> {
//     let mut visited = HashSet::new();
//     enumerate_recursive(egraph, eclass_id, &mut visited, 0)
// }

fn enumerate_all_expressions(
    egraph: &EGraph,
    eclass_id: Id,
) -> Vec<String> {
    let mut visited = HashSet::new();
    let extractor = Extractor::new(egraph, AstSize);
    enumerate_recursive_with_parent(egraph, eclass_id, &mut visited, 0, None, 0, &extractor)
}

fn enumerate_recursive_with_parent(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    extractor: &Extractor<AstSize, TileLang, LoopAnalysis>,
) -> Vec<String> {
    // Constraint 2: Handle cycles - allow only once
    if visited.contains(&eclass_id) {
        let best_expr = extractor.find_best(eclass_id);
        return vec![format!("{}", best_expr.1)]; // .1 is the extracted expression
    }
    // Depth limit (separate from cycle detection)
    if depth > 100 {
        return vec![format!("depth_limit")];
    }

    visited.insert(eclass_id);
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    for enode in &eclass.nodes {
        // NEW FILTER: Skip Seq nodes that are left children of parent Seq nodes
        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 { // Left child (index 0)
                    // println!("Skipping Seq as left child of parent Seq at depth: {:?}", depth);
                    continue;
                }
            }
        }

        if let TileLang::TLoop(_) = enode {
            continue;
        }
        
        
        let children = enode.children();
        
        if children.is_empty() {
            // Leaf node
            results.push(format!("{}", enode));
        } else {
            // Get expressions for each child
            let mut child_expressions = Vec::new();
            for (index, &child_id) in children.iter().enumerate() {
                let child_exprs = enumerate_recursive_with_parent(
                    egraph, 
                    child_id, 
                    visited, 
                    depth + 1, 
                    Some(enode),  // Pass current node as parent
                    index,         // Pass child index
                    extractor
                );
                child_expressions.push(child_exprs);
            }
            
            // Generate cartesian product
            let combinations = cartesian_product(&child_expressions);
            for combo in combinations {
                let expr_str = format_enode_with_children(enode, &combo);
                results.push(expr_str);
            }
        }
    }

    visited.remove(&eclass_id);
    results
}

// Alternative version with BigInt support for very large counts
use num_bigint::BigUint;
use std::str::FromStr;

fn count_all_expressions_bigint(egraph: &EGraph, eclass_id: Id) -> BigUint {
    let mut memo = HashMap::new();
    let mut visited = HashSet::new();
    count_recursive_with_parent_bigint(
        egraph, 
        eclass_id, 
        &mut visited, 
        0, 
        None, 
        0, 
        &mut memo
    )
}

fn count_recursive_with_parent_bigint(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
    parent_node: Option<&TileLang>,
    child_index: usize,
    memo: &mut HashMap<Id, BigUint>,
) -> BigUint {
    // Check memoization first
    if !visited.contains(&eclass_id) {
        if let Some(cached_count) = memo.get(&eclass_id) {
            return cached_count.clone();
        }
    }

    // Handle cycles
    if visited.contains(&eclass_id) {
        return BigUint::from(1u32);
    }

    // Depth limit
    if depth > 100 {
        return BigUint::from(1u32);
    }

    visited.insert(eclass_id);
    let mut total_count = BigUint::from(0u32);
    let eclass = &egraph[eclass_id];

    for enode in &eclass.nodes {
        // Apply filters
        if let TileLang::Seq(_) = enode {
            if let Some(TileLang::Seq(_)) = parent_node {
                if child_index == 0 {
                    continue;
                }
            }
        }

        if let TileLang::TLoop(_) = enode {
            continue;
        }
        
        let children = enode.children();
        
        if children.is_empty() {
            total_count += BigUint::from(1u32);
        } else {
            let mut node_count = BigUint::from(1u32);
            
            for (index, &child_id) in children.iter().enumerate() {
                let child_count = count_recursive_with_parent_bigint(
                    egraph, 
                    child_id, 
                    visited, 
                    depth + 1, 
                    Some(enode),
                    index,
                    memo
                );
                
                node_count *= child_count;
            }
            
            total_count += node_count;
        }
    }

    visited.remove(&eclass_id);
    memo.insert(eclass_id, total_count.clone());
    total_count
}

fn enumerate_recursive(
    egraph: &EGraph,
    eclass_id: Id,
    visited: &mut HashSet<Id>,
    depth: usize,
) -> Vec<String> {
    // Constraint 2: Handle cycles - allow only once
    if visited.contains(&eclass_id) {
        return vec![format!("#{}", eclass_id.as_usize())];
    }
    
    // Depth limit (separate from cycle detection)
    if depth > 100 {
        return vec![format!("depth_limit")];
    }

    visited.insert(eclass_id);
    let mut results = Vec::new();
    let eclass = &egraph[eclass_id];

    for enode in &eclass.nodes {
        // Constraint 1: Filter seq nodes
        if let TileLang::Seq(_) = enode {
            if !is_legal_seq_node(egraph, enode) {
                println!("---------------------");
                println!("Ilegal, depth: {:?}", depth);
                let children = enode.children();
                for &child_id in children {
                    print_eclass(egraph, child_id);
                }
                println!("---------------------");
                continue; // Skip this seq node if it's not legal
            } else {
                // println!("Legal: {:?}", depth);
            }
        }
        if let TileLang::TLoop(_) = enode {
            continue;
        }
        
        let children = enode.children();
        
        if children.is_empty() {
            // Leaf node
            results.push(format!("{}", enode));
        } else {
            // Get expressions for each child
            let mut child_expressions = Vec::new();
            for &child_id in children {
                let child_exprs = enumerate_recursive(egraph, child_id, visited, depth + 1);
                child_expressions.push(child_exprs);
            }
            
            // Generate cartesian product
            let combinations = cartesian_product(&child_expressions);
            for combo in combinations {
                let expr_str = format_enode_with_children(enode, &combo);
                results.push(expr_str);
            }
        }
    }

    visited.remove(&eclass_id);
    results
}

fn format_enode_with_children(enode: &TileLang, children: &[String]) -> String {
    match enode {
        // Loop constructs
        TileLang::Loop(_) => format!("(loop {} {} {} {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string())),
        
        TileLang::DLoop(_) => format!("(loop {} {} {} {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string())),
        
        TileLang::TLoop(_) => format!("(tmp_loop {} {} {} {} {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string()),
            children.get(4).unwrap_or(&"?".to_string()),
            children.get(5).unwrap_or(&"?".to_string())),

        // Tensor operations
        TileLang::Input(_) => format!("(input {})", 
            children.get(0).unwrap_or(&"?".to_string())),
        
        TileLang::Output(_) => format!("(output {})", 
            children.get(0).unwrap_or(&"?".to_string())),

        // Indexing
        TileLang::Tile(_) => format!("(tile {})", 
            children.get(0).unwrap_or(&"?".to_string())),
        
        TileLang::FullTile => "(fulltile)".to_string(),
        TileLang::Dummy => "(dummy)".to_string(),
        
        TileLang::Elem(_) => format!("(elem {})", 
            children.get(0).unwrap_or(&"?".to_string())),
        
        TileLang::Index(_) => {
            if children.is_empty() {
                "(index)".to_string()
            } else {
                format!("(index {})", children.join(" "))
            }
        },

        // Memory operations
        TileLang::Load(_) => format!("(load {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Store(_) => format!("(store {} {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string())),
        
        TileLang::Seq(_) => format!("(seq {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),

        // Constants
        TileLang::Const(_) => format!("(const {})", 
            children.get(0).unwrap_or(&"?".to_string())),

        // Arithmetic operations
        TileLang::Add(_) => format!("(+ {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Sub(_) => format!("(- {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Mul(_) => format!("(* {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Div(_) => format!("(/ {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Exp(_) => format!("(exp {})", 
            children.get(0).unwrap_or(&"?".to_string())),
        
        TileLang::Matmul(_) => format!("(@{} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::ReduceSum(_) => format!("(rsum {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),

        // Tensor manipulation
        TileLang::Concat(_) => format!("(concat {} {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string())),
        
        TileLang::Broadcast(_) => format!("(bcast {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Permute3(_) => format!("(permute3 {} {} {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string()),
            children.get(2).unwrap_or(&"?".to_string()),
            children.get(3).unwrap_or(&"?".to_string())),
        
        TileLang::Squeeze(_) => format!("(squeeze {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),
        
        TileLang::Unsqueeze(_) => format!("(unsqueeze {} {})", 
            children.get(0).unwrap_or(&"?".to_string()),
            children.get(1).unwrap_or(&"?".to_string())),

        // Leaf nodes
        TileLang::Num(n) => n.to_string(),
        TileLang::Var(s) => s.to_string(),
        _ => "".to_string(),
    }
}
// Helper function to compute cartesian product
fn cartesian_product<T: Clone>(lists: &[Vec<T>]) -> Vec<Vec<T>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    
    if lists.len() == 1 {
        return lists[0].iter().map(|item| vec![item.clone()]).collect();
    }
    
    let mut result = Vec::new();
    let first = &lists[0];
    let rest_product = cartesian_product(&lists[1..]);
    
    for item in first {
        for rest in &rest_product {
            let mut combination = vec![item.clone()];
            combination.extend(rest.clone());
            result.push(combination);
        }
    }
    
    result
}

// Function to return expressions as strings for easier reading
fn enumerate_all_expressions_as_strings(
    egraph: &EGraph,
    eclass_id: Id,
) -> Vec<String> {
    let expressions = enumerate_all_expressions(egraph, eclass_id);
    expressions.iter().map(|expr| expr.to_string()).collect()
}

// Function to get expressions for the root e-class (your original expression)
fn list_expressions_for_root(
    runner: &egg::Runner<TileLang, LoopAnalysis>
) -> Vec<String> {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression
    enumerate_all_expressions_as_strings(egraph, root_id)
}
// Function to get expressions for the root e-class (your original expression)
fn count_expressions_for_root(
    runner: &egg::Runner<TileLang, LoopAnalysis>
) -> BigUint {
    let egraph = &runner.egraph;
    let root_id = runner.roots[0]; // Assuming single root expression
    count_all_expressions_bigint(egraph, root_id)
}



// pub fn lookup_expr_filtered(egraph: &EGraph, expr: &RecExpr<TileLang>) -> Option<Id> {
//     fn helper(
//         egraph: &EGraph,
//         expr: &RecExpr<TileLang>,
//         memo: &mut HashMap<Id, Id>,
//         node_id: Id,
//     ) -> Option<Id> {
//         if let Some(&cached) = memo.get(&node_id) {
//             return Some(cached);
//         }

//         let node = &expr[node_id];
//         let child_ids: Vec<Id> = node
//             .children()
//             .iter()
//             .map(|&i| helper(egraph, expr, memo, i))
//             .collect::<Option<_>>()?;

//         let new_node = TileLang::from_op(&node.to_string(), child_ids).ok()?;

//         for class in egraph.classes() {
//             let class_id = class.id;

//             if class.data.is_deleted.len() > 0 {
//                 println!("{:?}", class.data.is_deleted)
//             }

//             if class.data.is_deleted.contains(&new_node) {
//                 continue;
//             }

//             if class.nodes.contains(&new_node) {
//                 memo.insert(node_id, class_id);
//                 return Some(class_id);
//             }
//         }
//         None
//     }

//     let mut memo = HashMap::new();
//     let root_id = Id::from(expr.as_ref().len() - 1);
//     helper(egraph, expr, &mut memo, root_id)
// }

// #[macro_export]
// macro_rules! test_fn_not {
//     (
//         $(#[$meta:meta])*
//         $name:ident, $rules:expr,
//         $(runner = $runner:expr,)?
//         $start:literal
//         !=
//         $($goal:literal),+ $(,)?
//         $(@check $check_fn:expr)?
//     ) => {
//         $(#[$meta])*
//         #[test]
//         pub fn $name() {
//             use egg::{Runner, RecExpr, Rewrite};
//             use crate::lookup_expr_filtered;

//             let start_expr: RecExpr<_> = $start.parse().unwrap();
//             let goal_exprs: Vec<RecExpr<_>> = vec![$($goal.parse().unwrap()),+];

//             let mut runner = Runner::default()
//                 .with_expr(&start_expr)
//                 .with_iter_limit(max_iter)
//                 .with_node_limit(1000000)
//                 .with_time_limit(std::time::Duration::from_secs(1000))
//                 .with_hook(|runner| {
//                     for &root in &runner.roots {
//                         dead_code_elimination_pass(&mut runner.egraph, root);
//                     }
//                     Ok(())
//                 })
//                 .run(&$rules);
//             for &root in &runner.roots {
//                 dead_code_elimination_pass(&mut runner.egraph, root);
//             }
//             runner.egraph.rebuild();

//             for goal_expr in &goal_exprs {
//                 assert!(
//                     lookup_expr_filtered(&runner.egraph, goal_expr).is_none(),
//                     "Expected goal expr to be found, but it was not:\n{}",
//                     goal_expr,
//                 );
//             }
//         }
//     };
// }

// #[macro_export]
// macro_rules! test_fn {
//     (
//         $(#[$meta:meta])*
//         $name:ident, $rules:expr,
//         $(runner = $runner:expr,)?
//         $start:literal
//         =>
//         $($goal:literal),+ $(,)?
//     ) => {
//         $(#[$meta])*
//         #[test]
//         pub fn $name() {
//             use egg::{Runner, RecExpr, Rewrite};
//             use crate::lookup_expr_filtered;

//             let start_expr: RecExpr<_> = $start.parse().unwrap();
//             let goal_exprs: Vec<RecExpr<_>> = vec![$($goal.parse().unwrap()),+];
        
//             let mut runner = Runner::default()
//                 .with_expr(&start_expr)
//                 .with_iter_limit(max_iter)
//                 .with_node_limit(1000000)
//                 .with_time_limit(std::time::Duration::from_secs(1000))
//                 // .with_hook(|runner| {
//                 //     for &root in &runner.roots {
//                 //         dead_code_elimination_pass(&mut runner.egraph, root);
//                 //     }
//                 //     Ok(())
//                 // })
//                 .run(&$rules);
//             // for &root in &runner.roots {
//             //     dead_code_elimination_pass(&mut runner.egraph, root);
//             // }
//             // runner.egraph.rebuild();

//             for goal_expr in &goal_exprs {
//                 assert!(
//                     lookup_expr_filtered(&runner.egraph, goal_expr).is_some(),
//                     "Expected goal expr to be found, but it was not:\n{}",
//                     goal_expr,
//                 );
//             }
//         }
//     };
// }

// #[macro_export]
// macro_rules! test_fn_without_deadcode {
//     (
//         $(#[$meta:meta])*
//         $name:ident, $rules:expr,
//         $(runner = $runner:expr,)?
//         $start:literal
//         =>
//         $($goal:literal),+ $(,)?
//         $(@check $check_fn:expr)?
//     ) => {
//         $(#[$meta])*
//         #[test]
//         pub fn $name() {
//             use egg::{Runner, RecExpr, Rewrite};
//             use crate::lookup_expr_filtered;

//             let start_expr: RecExpr<_> = $start.parse().unwrap();
//             let goal_exprs: Vec<RecExpr<_>> = vec![$($goal.parse().unwrap()),+];
        
//             let mut runner = {
//                 let mut r = Runner::default()
//                     .with_expr(&start_expr)
//                     .with_node_limit(1000000000)
//                     .with_iter_limit(max_iter)
//                     .with_time_limit(std::time::Duration::from_secs(1000))
//                     .run(&$rules);

//                 $( $check_fn(&r); )?

//                 r
//             };

//             for goal_expr in &goal_exprs {
//                 assert!(
//                     lookup_expr_filtered(&runner.egraph, goal_expr).is_some(),
//                     "Expected goal expr to be found, but it was not:\n{}",
//                     goal_expr,
//                 );
//             }
//         }
//     };
// }

egg::test_fn2! {bug_loop_split, rules(),
"
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
            (store (input Q1)
                (+
                    (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input K1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input C_sum) (index (tile h) (fulltile))) 1)
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
                    (* (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
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
                        (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input K1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WK) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )

                (store (input V1)
                    (+
                        (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input C_sum) (index (elem n) (fulltile))) 1)
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
                        (* (load (input O) (index (elem n) (fulltile) (fulltile))) 1)
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

egg::test_fn2! {test_attacc_skip_ft, rules(),
"
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
            (store (input Q1)
                (+
                    (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input K1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input C_sum) (index (tile h) (fulltile))) 1)
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
                    (* (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
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
"
(loop 0 2048 64 n
    (seq
        (loop 0 2048 tile_k k
            (seq
                (store (input Q1)
                    (+
                        (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input K1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WK) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )

                (store (input V1)
                    (+
                        (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input C_sum) (index (elem n) (fulltile))) 1)
                        (rsum
                            (load (input C_exp) (index (elem n) (fulltile) (tile p)))
                            2
                        )
                    )
                    (index (elem n) (fulltile))
                )
                (store (input O)
                    (+
                        (* (load (input O) (index (elem n) (fulltile) (fulltile))) 1)
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
                        (* (load (input C_sum) (index (tile m))) 1)
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
                        (* (load (input O) (index (tile m) (fulltile))) 1)
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
                                (* (load (input new_C_sum) (index (elem new_n) (index (tile m)))) 1)
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
                                (* (load (input new_O) (index (elem new_n) (index (tile m) (fulltile)))) 1)
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
                                    (* (load (input new_C_sum) (index (elem new_n) (index (tile m)))) 1)
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
                                    (* (load (input new_O) (index (elem new_n) (index (tile m) (fulltile)))) 1)
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
    //                                 (* (load (input new_C_sum) (index (elem new_n) (index (tile m)))) 1)
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
    //                                 (* (load (input new_O) (index (elem new_n) (index (tile m) (fulltile)))) 1)
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
                        (* (load (input C_sum) (index (tile m))) 1)
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
                        (* (load (output O) (index (tile m) (fulltile))) 1)
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
                        (* (load (input C_sum) (index (tile m))) 1)
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
                        (* (load (output O) (index (tile m) (fulltile))) 1)
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
                            (* (load (input C_sum) (index (tile m))) 1)
                            (rsum 
                                (load (input C_exp) (index (tile m) (tile n)))
                                1
                            )
                        )
                        (index (tile m))
                    )
                    (store (output O)
                        (+
                            (* (load (output O) (index (tile m) (fulltile))) 1)
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
                            (* (load (input C_sum) (index (tile m))) 1)
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
                        (* (load (output O) (index (tile m) (fulltile))) 1)
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
    //                         (* (load (input C_sum) (index (tile m))) 1)
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
    //                         (* (load (output O) (index (tile m) (fulltile))) 1)
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
    //                         (* (load (input C_sum) (index (tile m))) 1)
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
    //                         (* (load (output O) (index (tile m) (fulltile))) 1)
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
                            (* (load (input C1) (index (tile m) (tile n))) 1)
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
                            (* (load (input C2) (index (tile m) (tile n))) 1)
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
                    (*
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
    //                             (* (load (input C1) (index (tile m) (tile n))) 1)
    //                             (*
    //                                 (load (input X) (index (tile m) (tile k)))
    //                                 (load (input W1) (index (tile k) (tile n)))
    //                             )
    //                         )
    //                     (index (tile m) (tile n))
    //                     )
    //                     (store (input C2) 
    //                         (+
    //                             (* (load (input C2) (index (tile m) (tile n))) 1)
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
    //                 (*
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
    //                         (* (load (input C1) (index (tile m) (tile n))) 1)
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
    //                         (* (load (input C2) (index (tile m) (tile n))) 1)
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
    //                 (*
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
    //                         (* (load (input C1) (index (tile m) (tile n))) 1)
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
    //                         (* (load (output O) (index (tile m) (tile n))) 1)
    //                         (*
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
                                (* (load (input C1) (index (tile m) (tile n))) 1)
                                (*
                                    (load (input X) (index (tile m) (tile k)))
                                    (load (input W1) (index (tile k) (tile n)))
                                )
                            )
                        (index (tile m) (tile n))
                        )
                        (store (input C2) 
                            (+
                                (* (load (input C2) (index (tile m) (tile n))) 1)
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
                    (*
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
                    (+ (* (load (input C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
                    (+ (* (load (input C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
                        (+ (* (load (input C) (index (fulltile) (tile p))) 1)
                        (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                        ))
                        (index (fulltile) (tile p))
                    )
                    (store (input D)
                        (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
                        (+ (* (load (input C) (index (fulltile) (tile p))) 1)
                        (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                        ))
                        (index (fulltile) (tile p))
                    )
                    (store (input E)
                        (+ (* (load (input E) (index (fulltile) (tile p))) 1)
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
                    (* (load (output O) (index (fulltile) (tile p))) 1)
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
                (store (output B) (+ (* (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input A) (index (tile n))) (load (output C) (index (tile n)))) (index (tile n)))
            )
            )
        )
        (store (output E) (* (load (output B) (index)) 10) (index))
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
            (seq
                (store (output B) (+ (* (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
            )
        )
        (store (output E) (* (load (output B) (index)) 10) (index))
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output B) (+ (* (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
            )
        )
        (store (output E) (* (load (output B) (index)) 10) (index))
    )
    "
    ,
    "
    (seq
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (* (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
        )
    )
        (store (output E) (* (load (output B) (index)) 10) (index))
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (* (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
        )
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
        )
        (store (output E) (* (load (output B) (index)) 10) (index))
    ))
    "
    ,
    "
    (loop 0 N tile_n n 
        (seq
            (store (output C) (+ (load (input A) (index (tile n))) 1) (index (tile n)))
        (seq
            (store (output E) (+ (* (load (output E) (index)) 3) (* 10 (load (input A) (index (tile n))))) (index))
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
            (store (output B) (+ (* (load (output B) (index)) 3) (* 10 (load (input A) (index (tile n))))) (index))
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
                (store (output B) (+ (* (load (output B) (index)) 3) (load (input A) (index (tile n)))) (index))
                (store (output D) (* (load (input A) (index (tile n))) (+ (load (input A) (index (tile n))) 1)) (index (tile n)))
            )
            )
        )
        (store (output B) (* (load (output B) (index)) 10) (index))
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
                        (* (load (input B) (index (tile n))) 1)
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
                            (* (load (input E) (index (tile n) (fulltile))) 1)
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
                            (* (load (input B) (index (tile n))) 1)
                            (rsum (load (input A) (index (tile n) (tile m))) 1)
                        )
                        (index (tile n))
                    )
                    (store (input E)
                        (+
                            (* (load (input E) (index (tile n) (fulltile))) 1)
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
                            (* (load (input B) (index (tile n))) 1)
                            (rsum (load (input A) (index (tile n) (tile m))) 1)
                        )
                        (index (tile n))
                    )
                    (store (input E)
                        (+
                            (* (load (input E) (index (tile n) (fulltile))) 1)
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
            (store (input B) (+ (* (load (input B) (index)) 1) (* 10 (rsum (load (input A) (index (tile n))) 1))) (index))
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
                        (* (load (input new_B) (index (elem new_n) (index))) 1)
                        (* 10 (rsum (load (input A) (index (tile n))) 1))
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
                    (* (* 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
                    (* (* 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
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
    //             (+ (* (load (output D) (index)) 1)
    //                 (* (* 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
    //             )
    //             (index)
    //         )
    //     )
    //     (loop 0 N tile_n n 
    //         (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
    //     )
    // )
    // "
}
egg::test_fn2! {seq_comm3, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
                    (* (* 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
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
                (+ (* (load (output D) (index)) 1)
                    (* (* 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
                )
                (index)
            )
        )
        (loop 0 N tile_n n 
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
                    (* (* 2 (load (input A) (index (tile n)))) (load (input C) (index (tile n))))
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
                    (* (* (load (input A) (index (tile n))) (load (input C) (index (tile n)))) 2)
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (* (load (output D) (index)) 1)
                    (* (* (load (input A) (index (tile n))) (load (input C) (index (tile n)))) 2)
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
            (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
        )
        (seq
            (loop 0 N tile_n n 
                (store (output D) 
                    (+ (* (load (output D) (index)) 1)
                        (* (load (input A) (index (tile n))) (load (input C) (index (tile n))))
                    )
                    (index)
                )
            )
            (store (output D) (* (load (output D) (index)) 2) (index))
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output B) (* 2 (load (input A) (index (tile n)))) (index (tile n)))
                (store (output D) 
                    (+ (* (load (output D) (index)) 1)
                        (* (load (input A) (index (tile n))) (load (input C) (index (tile n))))
                    )
                    (index)
                )
            )
        )
        (store (output D) (* (load (output D) (index)) 2) (index))
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
            (store (output B) (+ (* (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 N tile_n n
            (store (output D) (+ (* (load (output D) (index)) 1) (/ (load (input C) (index (tile n))) (load (output B) (index)))) (index))
        )
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (* (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
        )
    (seq
        (loop 0 N tile_n n
            (store (output D) (+ (* (load (output D) (index)) 1) (load (input C) (index (tile n)))) (index))
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
                (store (output B) (+ (* (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
                (store (output D) (+ (* (load (output D) (index)) 1) (load (input C) (index (tile n)))) (index))
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
            (store (output B) (+ (* (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
        )
        (loop 0 N tile_n n
            (store (output D) (+ (* (load (output D) (index)) 1) (/ (load (input C) (index (tile n))) (load (output B) (index)))) (index))
        )
    )
    "
    !=
    "
    (loop 0 N tile_n n
        (seq
            (store (output B) (+ (* (load (output B) (index)) 1) (load (input A) (index (tile n)))) (index))
            (store (output D) (+ (* (load (output D) (index)) 1) (/ (load (input C) (index (tile n))) (load (output B) (index)))) (index))
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
                        (* (load (input C1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input C1) (index (fulltile) (tile n))) 1)
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
                    (*
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
                    (*
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
                        (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input K1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                            (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                            (* (load (input K1) (index (fulltile) (tile n))) 1)
                            (*
                                (load (input X) (index (fulltile) (tile k)))
                                (load (input WK) (index (tile k) (tile n)))
                            )
                        )
                        (index (fulltile) (tile n))
                    )

                    (store (input V1)
                        (+
                            (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input K1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WK) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )

                (store (input V1)
                    (+
                        (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                            (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                            (* (load (input K1) (index (fulltile) (tile n))) 1)
                            (*
                                (load (input X) (index (fulltile) (tile k)))
                                (load (input WK) (index (tile k) (tile n)))
                            )
                        )
                        (index (fulltile) (tile n))
                    )

                    (store (input V1)
                        (+
                            (* (load (input V1) (index (fulltile) (tile n))) 1)
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

egg::test_fn2! {debug_lora1, rules(),
    "
    (seq
        (loop 0 P tile_p p 
            (loop 0 N tile_n n
                (store (input C)
                    (+ (* (load (input C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
    (loop 0 P tile_p p 
        (seq 
            (loop 0 N tile_n n 
                (store (input C) 
                    (+ (* (load (input C) (index fulltile (tile p))) 1) 
                    (* (load (input X) (index fulltile (tile n))) 
                        (load (input W) (index (tile n) (tile p))))) 
                    (index fulltile (tile p))))
            (seq 
                (loop 0 P tile_p p 
                    (loop 0 N tile_n n 
                        (store (input D) 
                            (+ (* 1 (load (input D) (index fulltile fulltile))) 
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
egg::test_fn2! {test_default_tiling, default_tiling(),
    "
    (seq
        (store tmp1 (* A B) (index))
    (seq
        (store tmp2 (* A C) (index))
        (store O (+ (load tmp1 (index)) (load tmp2 (index))) (index))
    ))
    "
    =>
    "(seq
        (store tmp1 (+ B C) (index))
        (store O (* A (load tmp1 (index))) (index))
    )
    "
}

// fn generate_nested_seq(n: usize, tail: &str) -> String {
//     let mut expr = tail.to_string();
//     for i in (1..=n).rev() {
//         expr = format!("(seq {} {})", i, expr);
//     }
//     expr
// }


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
    );

    save_egraph(&runner, "egraph.dot");
}

#[test]
fn saturate_gated_mlp_skip_ft() {
    let expr = "
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C1) 
                    (+
                        (* (load (input C1) (index (fulltile) (tile n))) 1)
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
            (store (input C1_exp) 
                (exp
                    (load (input C1) (index (fulltile) (tile n)))
                )
                (index (fulltile) (tile n))
            )
        )
    
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C2) 
                    (+
                        (* (load (input C2) (index (fulltile) (tile n))) 1)
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
                (*
                    (load (input C1_exp) (index (fulltile) (tile n)))
                    (load (input C2) (index (fulltile) (tile n)))
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
                    (store (input C1) 
                        (+
                            (* (load (input C1) (index (tile m) (tile n))) 1)
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
                            (* (load (input C2) (index (tile m) (tile n))) 1)
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
                    (*
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
                        (+ (* (load (input C) (index (tile m) (tile p))) 1)
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
                        (+ (* (load (input D) (index (tile m) (tile k))) 1)
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
                        (+ (* (load (input E) (index (tile m) (tile p))) 1)
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
                (store (input C)
                    (+ (* (load (input C) (index (fulltile) (tile p))) 1)
                    (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
                    ))
                    (index (fulltile) (tile p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
                    (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input K1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input C_sum) (index (tile h) (fulltile))) 1)
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
                    (* (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
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
                        (* (load (input C_sum) (index (tile m))) 1)
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
                        (* (load (output O) (index (tile m) (fulltile))) 1)
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
    //                 (+ (* (load (input C) (index (fulltile) (tile p))) 1)
    //                 (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
    //                 ))
    //                 (index (fulltile) (tile p))
    //             )
    //         )
    //     )
    // (seq
    //     (loop 0 N tile_n n
    //         (store (input D)
    //             (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
    //         (store (input B) (+ (* (load (input B) (index)) 1) (* 10 (rsum (load (input A) (index (tile n))) 1))) (index))
    //     )
    // (seq
    //     (loop 0 N tile_n n
    //         (store (input B) (+ (* (load (input B) (index)) 1) (* 10 (rsum (load (input A) (index (tile n))) 1))) (index))
    //     )
    //     (store (input B) (* (load (input B) (index)) 20) (index))
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

#[test]
fn extract_expressions() {
    // let expr = "
    // (seq
    //     (loop 0 P tile_p p 
    //         (loop 0 N tile_n n
    //             (store (input C)
    //                 (+ (* (load (input C) (index (fulltile) (tile p))) 1)
    //                 (* (load (input X) (index (fulltile) (tile n))) (load (input W) (index (tile n) (tile p)))
    //                 ))
    //                 (index (fulltile) (tile p))
    //             )
    //         )
    //     )
    // (seq
    //     (loop 0 N tile_n n
    //         (store (input D)
    //             (+ (* (load (input D) (index (fulltile) (fulltile))) 1)
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
    // let expr = "
    // (seq
    //     (loop 0 N tile_n n 
    //         (loop 0 K tile_k k 
    //             (store (input C1) 
    //                 (+
    //                     (* (load (input C1) (index (fulltile) (tile n))) 1)
    //                     (*
    //                         (load (input X) (index (fulltile) (tile k)))
    //                         (load (input W1) (index (tile k) (tile n)))
    //                     )
    //                 )
    //                 (index (fulltile) (tile n))
    //             )
    //         )
    //     )
        
    // (seq
    //     (loop 0 N tile_n n 
    //         (store (input C1_exp) 
    //             (exp
    //                 (load (input C1) (index (fulltile) (tile n)))
    //             )
    //             (index (fulltile) (tile n))
    //         )
    //     )
    
    // (seq
    //     (loop 0 N tile_n n 
    //         (loop 0 K tile_k k 
    //             (store (input C2) 
    //                 (+
    //                     (* (load (input C2) (index (fulltile) (tile n))) 1)
    //                     (*
    //                         (load (input X) (index (fulltile) (tile k)))
    //                         (load (input W2) (index (tile k) (tile n)))
    //                     )
    //                 )
    //                 (index (fulltile) (tile n))
    //             )
    //         )
    //     )
    //     (loop 0 N tile_n n 
    //         (store (output O) 
    //             (*
    //                 (load (input C1_exp) (index (fulltile) (tile n)))
    //                 (load (input C2) (index (fulltile) (tile n)))
    //             )
    //             (index (fulltile) (tile n))
    //         )
    //     )
    // )
    // )
    // )
    // ";
    let expr = "
(seq
    (loop 0 2048 tile_n n
        (loop 0 2048 tile_k k
            (store (input Q1)
                (+
                    (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input K1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                    (* (load (input C_sum) (index (tile h) (fulltile))) 1)
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
                    (* (load (input O) (index (tile h) (fulltile) (fulltile))) 1)
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
";

// let expr = "
// (seq
//     (loop 0 M tile_m m
//         (loop 0 N tile_n n 
//             (store (input C) 
//                 (*
//                     (load (input Q) (index (tile m) (fulltile)))
//                     (load (input K) (index (fulltile) (tile n)))
//                 )
//                 (index (tile m) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 M tile_m m
//         (loop 0 N tile_n n 
//             (store (input C_exp)
//                 (exp (load (input C) (index (tile m) (tile n))))
//                 (index (tile m) (tile n))
//             )
//         )
//     )
// (seq
//     (loop 0 M tile_m m
//         (loop 0 N tile_n n 
//             (store (input C_sum)
//                 (+
//                     (* (load (input C_sum) (index (tile m))) 1)
//                     (rsum (load (input C_exp) (index (tile m) (tile n))) 1)
//                 )
//                 (index (tile m))
//             )
//         )
//     )
// (seq
//     (loop 0 M tile_m m
//         (loop 0 N tile_n n 
//             (store (input C_div)
//                 (/
//                     (load (input C_exp) (index (tile m) (tile n)))
//                     (bcast (load (input C_sum) (index (tile m))) 1)
//                 )
//                 (index (tile m) (tile n))
//             )
//         )
//     )
//     (loop 0 M tile_m m
//         (loop 0 N tile_n n 
//             (store (output O)
//                 (+
//                     (* (load (output O) (index (tile m) (fulltile))) 1)
//                     (*
//                         (load (input C_div) (index (tile m) (tile n)))
//                         (load (input V) (index (tile n) (fulltile)))
//                     )
//                 )
//                 (index (tile m) (fulltile))
//             )
//         )
//     )
// ))))
// ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
    );
    // postprocess_egraph(&mut runner.egraph);

    // List all expressions for the root e-class
    println!("All equivalent expressions for your input:");
    // let root_expressions = list_expressions_for_root(&runner);
    // println!("There are {:?} expressions", root_expressions.len());

    let total_expressions = count_expressions_for_root(&runner);
    println!("There are {:?} expressions", total_expressions);

    // let file = File::create("expressions/attacc_skip_ft.txt").expect("Failed to create file");
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
}

#[test]
fn postprocess_expression() {
    let expr = "
(loop 0 2048 64 n
    (seq
        (loop 0 2048 tile_k k
            (seq
                (store (input Q1)
                    (+
                        (* (load (input Q1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input K1) (index (fulltile) (tile n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k)))
                            (load (input WK) (index (tile k) (tile n)))
                        )
                    )
                    (index (fulltile) (tile n))
                )

                (store (input V1)
                    (+
                        (* (load (input V1) (index (fulltile) (tile n))) 1)
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
                        (* (load (input C_sum) (index (elem n) (fulltile))) 1)
                        (rsum
                            (load (input C_exp) (index (elem n) (fulltile) (tile p)))
                            2
                        )
                    )
                    (index (elem n) (fulltile))
                )
                (store (input O)
                    (+
                        (* (load (input O) (index (elem n) (fulltile) (fulltile))) 1)
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
        (store (output O2)
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
    let new_expr = postprocess(expr);
    println!("{}", new_expr);
}

fn save_egraph(runner: &egg::Runner<TileLang, LoopAnalysis>, filename: &str) {
    let dot_string = runner.egraph.dot().to_string();
    let mut file = File::create(filename).expect("Failed to create dot file");
    file.write_all(dot_string.as_bytes()).expect("Failed to write dot file");
}