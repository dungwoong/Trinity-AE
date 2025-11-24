use egg::{rewrite as rw2, *};
use std::collections::VecDeque;
use std::collections::HashSet;
use std::collections::HashMap;

define_language! {
    enum TileLang {
    
        "loop" = Loop([Id; 5]), // loop start end tile_n loop_var body
        "dloop" = DLoop([Id; 5]), // loop start end tile_n loop_var body
        "input" = Input(Id),    // name of tensor => input 말고 다른 이름으로 바꾸자
        "output" = Output(Id), // name of tensor that should not be deleted -> 이것도 이름 바꾸자
        "tile" = Tile([Id; 2]), // tile n tile_n = [n:n+tile_n]
        "fulltile" = FullTile,  // fulltile = [:]
        "index" = Index(Box<[Id]>), // index (tile n tile_n) (tile m tile_m) ...
        "load" = Load([Id; 2]), // load A index ...
        "store" = Store([Id; 3]), // store A val index ...
        "seq" = Seq([Id; 2]),   // seq body1 body2 ...

        "+" = Add([Id; 2]), // a + b
        "-" = Sub([Id; 2]), // a - b
        "x" = Mul([Id; 2]), // a x b
        "/" = Div([Id; 2]), // a / b
        "exp" = Exp(Id), // exp(a)
        "*" = Matmul([Id; 2]), // a * b
        "rsum" = ReduceSum([Id; 2]), // reduce_sum(a, axis)

        "concat" = Concat([Id; 3]), // concat(a, b, axis)
        "bcast" = Broadcast([Id; 2]), // broadcast(a, axis)
        "dummy" = NoneOp, // for deadcode

        Num(i32),
        Var(egg::Symbol),
    }
}



type EGraph = egg::EGraph<TileLang, LoopAnalysis>;
static max_iter: usize = 10;



fn custom_rules() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![
        rw!("loop-fusion"; 
            "(seq (loop ?start ?n ?tile_n ?loop_var ?body1) (seq (loop ?start ?n ?tile_n ?loop_var ?body2) ?others))" => 
            "(seq (loop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
            )
        ),
        rw!("loop-fusion-tail"; 
            "(seq (loop ?start ?n ?tile_n ?loop_var ?body1) (loop ?start ?n ?tile_n ?loop_var ?body2))" => 
            "(loop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2))" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
            )
        ),
        rw!("loop-fission"; 
            "(seq (loop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" =>
            "(seq (loop ?start ?n ?tile_n ?loop_var ?body1) (seq (loop ?start ?n ?tile_n ?loop_var ?body2) ?others))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),
        rw!("loop-fission-tail"; 
            "(loop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2))" =>
            "(seq (loop ?start ?n ?tile_n ?loop_var ?body1) (loop ?start ?n ?tile_n ?loop_var ?body2))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),
        rw!("loop-deletion";
            "(loop 0 ?n ?tile_n ?loop_var ?body)" =>
            "?body"
            if and_all! (
                no_dependency_with_loopvar(var("?body"), var("?loop_var")),
            )
        ),

        rw!("loop-split";
            "(loop 0 ?end ?tile ?loop_var (store (input ?a) (+ (x (load (input ?a) ?idx) 1) ?body) ?idx))" =>
            { LoopSplit{
                end: var("?end"), tile: var("?tile"), loop_var: var("?loop_var"), a: var("?a"), idx: var("?idx"), body: var("?body"),
                new_tile: var("?new_tile"), new_loop_var: var("?new_loop_var"), new_a: var("?new_a"),
                rhs: "(seq 
                (loop 0 ?end ?new_tile ?new_loop_var 
                  (loop ?new_loop_var (+ ?new_loop_var ?new_tile) ?tile ?loop_var (store (input ?new_a) (+ (x (load (input ?new_a) (index (tile (/ ?new_loop_var ?new_tile) 1) ?idx)) 1) ?body) (index (tile (/ ?new_loop_var ?new_tile) 1) ?idx))))
                (store (input ?a) (rsum (load (input ?new_a) (index (fulltile) ?idx)) 0) ?idx))".parse().unwrap(),
            }}
        ),
    ]
}
fn rules() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![
        // loop transformation rules
        rw!("seq-comm";
            "(seq ?a (seq ?b ?body))" => "(seq ?b (seq ?a ?body))"
            if and_all! (
                no_all_dependency(var("?a"), var("?b"))
            )
        ),
        rw!("seq-comm-tail";
            "(seq ?a ?b)" => "(seq ?b ?a)"
            if and_all!(
                no_all_dependency(var("?a"), var("?b")),
                no_seq(var("?b"))
            )
        ),
        rw!("loop-fusion"; 
            "(seq (loop ?start ?n ?tile_n ?loop_var ?body1) (seq (loop ?start ?n ?tile_n ?loop_var ?body2) ?others))" => 
            "(seq (loop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
            )
        ),
        rw!("loop-fusion-tail"; 
            "(seq (loop ?start ?n ?tile_n ?loop_var ?body1) (loop ?start ?n ?tile_n ?loop_var ?body2))" => 
            "(loop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2))" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
            )
        ),
        rw!("dloop-fusion"; 
            "(seq (dloop ?start ?n ?tile_n ?loop_var ?body1) (seq (dloop ?start ?n ?tile_n ?loop_var ?body2) ?others))" => 
            "(seq (dloop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
            )
        ),
        rw!("dloop-fusion-tail"; 
            "(seq (dloop ?start ?n ?tile_n ?loop_var ?body1) (dloop ?start ?n ?tile_n ?loop_var ?body2))" => 
            "(dloop ?start ?n ?tile_n ?loop_var (seq ?body1 ?body2))" 
            if and_all!(
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
            )
        ),

        rw!("loop-fission"; 
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (seq (loop 0 ?n ?tile_n ?loop_var ?body2) ?others))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),
        rw!("loop-fission-tail"; 
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (loop 0 ?n ?tile_n ?loop_var ?body2))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),

        rw!("loop-insertion1";
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (seq ?body2 ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?body2"), var("?loop_var")),
                not_same_loop(var("?body2"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion1-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) ?body2)" =>
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))"
            if and_all!(
                no_dependency_with_loopvar(var("?body2"), var("?loop_var")),
                not_same_loop(var("?body2"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_seq(var("?body2")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion2";
            "(seq ?body1 (seq (loop 0 ?n ?tile_n ?loop_var ?body2) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?body1"), var("?loop_var")),
                not_same_loop(var("?body1"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion2-tail";
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
        rw!("loop-comm";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq
                        (store ?a (+ (x (load ?a ?idx) 1) ?val1) ?idx)
                        (store ?b (+ (x (load ?b ?idx) 1) ?val2) ?idx)))
            (seq (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx) ?others))"
            =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx) 1) (+ ?val1 ?val2)) ?idx)) ?others)"
        ),
        rw!("loop-comm-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq
                        (store ?a (+ (x (load ?a ?idx) 1) ?val1) ?idx)
                        (store ?b (+ (x (load ?b ?idx) 1) ?val2) ?idx)))
                (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx))"
            =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx) 1) (+ ?val1 ?val2)) ?idx))"
        ),

        rw!("loop-factor-matmul-tail";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) (* ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
                  (store ?b (* (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-matmul";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) (* ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
                  (seq (store ?b (* (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),

        rw!("loop-factor-div-tail";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) (/ ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
                  (store ?b (/ (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-div";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) (/ ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
                  (seq (store ?b (/ (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-mul-tail";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) (x ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
                  (store ?b (x (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),
        rw!("loop-factor-mul";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) (x ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
                  (seq (store ?b (x (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),

        rw!("loop-dist-matmul";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
             (seq (store ?c (* (load ?b ?idx) ?val2) ?idx2) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx2) ?accm) (* ?val1 ?val2)) ?idx2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-matmul-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
              (store ?c (* (load ?b ?idx) ?val2) ?idx2))" =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx2) ?accm) (* ?val1 ?val2)) ?idx2))" 
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-div";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
             (seq (store ?c (/ (load ?b ?idx) ?val2) ?idx) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx) ?accm) (/ ?val1 ?val2)) ?idx)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-div-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
              (store ?c (/ (load ?b ?idx) ?val2) ?idx))" =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx) ?accm) (/ ?val1 ?val2)) ?idx))" 
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-dist-mul";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
             (seq (store ?c (x (load ?b ?idx) ?val2) ?idx) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx) ?accm) (x ?val1 ?val2)) ?idx)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),
        rw!("loop-dist-mul-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (x (load ?b ?idx) ?accm) ?val1) ?idx))
              (store ?c (x (load ?b ?idx) ?val2) ?idx))" =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (x (load ?c ?idx) ?accm) (x ?val1 ?val2)) ?idx))" 
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),

        rw!("loop-split";
            "(loop 0 ?end ?tile ?loop_var (store (input ?a) (+ (x (load (input ?a) ?idx) 1) ?body) ?idx))" =>
            { LoopSplit{
                end: var("?end"), tile: var("?tile"), loop_var: var("?loop_var"), a: var("?a"), idx: var("?idx"), body: var("?body"),
                new_tile: var("?new_tile"), new_loop_var: var("?new_loop_var"), new_a: var("?new_a"),
                rhs: "(seq 
                (dloop 0 ?end ?new_tile ?new_loop_var 
                  (dloop ?new_loop_var (+ ?new_loop_var ?new_tile) ?tile ?loop_var (store (input ?new_a) (+ (x (load (input ?new_a) (index (tile (/ ?new_loop_var ?new_tile) 1) ?idx)) 1) ?body) (index (tile (/ ?new_loop_var ?new_tile) 1) ?idx))))
                (store (input ?a) (rsum (load (input ?new_a) (index (fulltile) ?idx)) 0) ?idx))".parse().unwrap(),
            }}
            if and_all!(
                has_dependency_with_loopvar(var("?body"),var("?loop_var"))
            )
        ),

        // algebraic transformation rules
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("comm-mul";  "(x ?a ?b)"        => "(x ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add2"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("assoc-mul"; "(x ?a (x ?b ?c))" => "(x (x ?a ?b) ?c)"),
        rw!("assoc-mul2"; "(x (x ?a ?b) ?c)" => "(x ?a (x ?b ?c))"),
        rw!("assoc-matmul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rw!("assoc-matmul2"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rw!("assoc-div-matmul"; "(* (/ ?a (bcast ?b 1)) ?c)" => "(/ (* ?a ?c) (bcast ?b 1))"),

        // rw!("cancel-sub"; "(- ?a ?a)" => "0"),
        // rw!("cancel-div"; "(/ ?a ?a)" => "1" if is_not_zero(var("?a"))),
        // rw!("multiply-one"; "?a" => "(x ?a 1)" if is_load(var("?a"))),
        
        rw!("dist-mul-add"; "(x ?a (+ ?b ?c))"        => "(+ (x ?a ?b) (x ?a ?c))"),
        rw!("dist-mul-sub"; "(x ?a (- ?b ?c))"        => "(- (x ?a ?b) (x ?a ?c))"),
        rw!("factor-add"    ; "(+ (x ?a ?b) (x ?a ?c))" => "(x ?a (+ ?b ?c))"),
        rw!("factor-sub"    ; "(- (x ?a ?b) (x ?a ?c))" => "(x ?a (- ?b ?c))"),
        
        rw!("exp-mul"; "(x (exp ?a) (exp ?b))" => "(exp (+ ?a ?b))"),
        rw!("exp-div"; "(/ (exp ?a) (exp ?b))" => "(exp (- ?a ?b))"),
        rw!("exp0"; "(exp 0)" => "1"),
        rw!("recip-mul-div"; "(x ?x (/ 1 ?x))" => "1" if is_not_zero(var("?x"))),

        rw!("matmul-concat"; "(+ (* ?a ?b) (* ?c ?d))" => "(* (concat ?a ?c 1) (concat ?b ?d 0))"),
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
                                            source_node: src,
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
                                            source_node: src,
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
                                            source_node: src,
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
                                            source_node: src,
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
fn no_dummy(body: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _eclass, subst| {
        let body_id = subst[body];
        let data = &egraph[body_id].data;

        !egraph[body_id]
            .nodes
            .iter()
            .filter(|n| !data.is_deleted.contains(n))
            .any(|n| matches!(n, TileLang::NoneOp))
    }
}

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
                    TileLang::Tile([tile_idx, _]) => depends_on_id(egraph, *tile_idx, loop_var),
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
    println!("EClass {} has {} enodes:", id, class.nodes.len());
    for node in &class.nodes {
        println!("  {}", node);
    }
}

#[derive(Default)]
struct LoopAnalysis;

#[derive(Debug, Clone, Default)]
pub struct LoopData {
    // read_set: Vec<Access>,
    // write_set: Vec<Access>,
    is_tensor: bool, // whether the output of the operator is tensor or not
    is_deleted: HashSet<TileLang>, // track deleted terms per eclass
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Access {
    pub base: Option<String>,
    pub index: Option<TileLang>,
    pub source_node: Option<TileLang>, // NEW: the enode that generated this Access
}

impl Analysis<TileLang> for LoopAnalysis {
    type Data = LoopData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        // let old_read_len = to.read_set.len();
        // let old_write_len = to.write_set.len();
        let old_deleted_len = to.is_deleted.len();

        // to.read_set.extend(from.read_set);
        // to.write_set.extend(from.write_set);
        to.is_deleted.extend(from.is_deleted);

        // to.read_set.sort();
        // to.read_set.dedup();
        // to.write_set.sort();
        // to.write_set.dedup();

        assert_eq!(to.is_tensor, from.is_tensor, "Mismatched is_tensor flags during merge");


        DidMerge(
            // to.read_set.len() > old_read_len ||
            // to.write_set.len() > old_write_len ||
            to.is_deleted.len() > old_deleted_len,
            true,
            // false, true
            // true, true
        )
    }

    fn make(egraph: &mut EGraph, enode: &TileLang) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        match enode {
            TileLang::Tile([base, _tile_size]) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            },
            TileLang::FullTile => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Input(tensor) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Output(tensor) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Index(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Load([base, idx]) => {
                Self::Data {
                    is_tensor: true,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Store([base, val, idx]) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Loop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::DLoop(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Seq(args) => {
            
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Add(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Sub(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Div(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Mul(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Matmul(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Concat(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Exp(arg) => {
                Self::Data {
                    is_tensor: true,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::ReduceSum(args) => {
                Self::Data {

                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Broadcast(args) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Var(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::Num(_) => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
            TileLang::NoneOp => {
                Self::Data {
                    is_tensor: false,
                    is_deleted: HashSet::new(),
                }
            }
        }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        // ====================================
        // (1) Sequence flattening phase
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
                let mut seq_elements = vec![];
                
                // println!("{:?}", node);
                // for &printid in node.children() {
                //     print_eclass(egraph, printid);
                // }
                // println!("------------------------");
    
                // Step 1: Flatten recursively
                flatten_seq(egraph, left, &mut seq_elements);
                flatten_seq(egraph, right, &mut seq_elements);

                // println!("{:?}",left);
                // println!("{:?}",right);
                // println!("{:?}",seq_elements);

                // print_eclass(egraph, left);
                // print_eclass(egraph, right);
                // let seqs = seq_elements.clone();
                // for &printid in seqs.iter() {
                //     println!("{:?}", extract_expr(egraph, printid).as_ref());
                // }

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

                // Delete the original illegal seq node
                // egraph[id].data.is_deleted.insert(node);
            }
        }

        // ====================================
        // (2) Value Forwarding Phase
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
                    let nodes = egraph[eclass_id].nodes.clone();
                    for enode in nodes {
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
        // (3) Deadcode Elimination Phase
        // ====================================

    }
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
pub struct DeletedFilteredSearcher {
    pattern: Pattern<TileLang>,
}

impl DeletedFilteredSearcher {
    pub fn new(pattern: Pattern<TileLang>) -> Self {
        Self { pattern }
    }
}

impl Searcher<TileLang, LoopAnalysis> for DeletedFilteredSearcher {
    fn search(&self, egraph: &EGraph) -> Vec<SearchMatches<TileLang>> {
        self.pattern
            .search(egraph)
            .into_iter()
            .map(|mut matches| {
                matches.substs.retain(|subst| {
                    self.vars().iter().all(|&var| {
                        let id = subst[var];
                        let data = &egraph[id].data;
                        egraph[id]
                            .nodes
                            .iter()
                            .any(|n| !data.is_deleted.contains(n))
                    })
                });
                matches
            })
            .filter(|matches| !matches.substs.is_empty())
            .collect()
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<TileLang>> {
        let mut matches = self.pattern.search_eclass_with_limit(egraph, eclass, limit)?;

        matches.substs.retain(|subst| {
            self.vars().iter().all(|&var| {
                let id = subst[var];
                let data = &egraph[id].data;
                egraph[id]
                    .nodes
                    .iter()
                    .any(|n| !data.is_deleted.contains(n))
            })
        });

        if matches.substs.is_empty() {
            None
        } else {
            Some(matches)
        }
    }

    fn vars(&self) -> Vec<egg::Var> {
        self.pattern.vars()
    }
}


#[macro_export]
macro_rules! rw {
    (
        $name:expr;
        $lhs:tt => $rhs:tt
        $(if $cond:expr)*
    ) => {{
        // Use your DeletedFilteredSearcher wrapper
        let searcher = $crate::DeletedFilteredSearcher::new($crate::__rewrite!(@parse Pattern $lhs));
        let core_applier = $crate::__rewrite!(@parse Pattern $rhs);
        let applier = $crate::__rewrite!(@applier core_applier; $($cond,)*);
        $crate::Rewrite::new($name.to_string(), searcher, applier).unwrap()
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __rewrite {
    (@parse $t:ident $expr:literal) => {
        $expr.parse::<$crate::$t<_>>().unwrap()
    };
    (@parse $t:ident $expr:expr) => { $expr };

    (@applier $applier:expr;) => { $applier };
    (@applier $applier:expr; $cond:expr, $($conds:expr,)*) => {
        $crate::ConditionalApplier {
            condition: $cond,
            applier: $crate::__rewrite!(@applier $applier; $($conds,)*)
        }
    };
}

#[macro_export]
macro_rules! and_all {
    ($($cond:expr),+ $(,)?) => {{
        move |egraph: &mut EGraph, id: Id, subst: &Subst| {
            true $( && $cond(egraph, id, subst) )+
        }
    }};
}

fn contains_expr(egraph: &EGraph, root: Id, expr: &RecExpr<TileLang>) -> bool {
    lookup_expr_filtered(egraph, expr).map_or(false, |id| {
        // Ensure the id is reachable from root
        is_reachable_from(egraph, root, id)
    })
}

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

fn dead_code_elimination_pass(egraph: &mut EGraph, root: Id) {
    use std::collections::{HashSet, VecDeque};

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut read_bases = HashSet::new();
    let mut write_info = Vec::new();

    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        if !visited.insert(id) {
            continue;
        }

        let eclass = &egraph[id];
        for enode in &eclass.nodes {
            if eclass.data.is_deleted.contains(enode) {
                // println!("{:?}", eclass.data.is_deleted);
                continue;
            }

            match enode {
                TileLang::Load([base, _]) => {
                    read_bases.insert(*base);
                }
                TileLang::Store([base, _, _]) => {
                    write_info.push((*base, id, enode.clone()));
                }
                _ => {}
            }

            for &child in enode.children() {
                queue.push_back(child);
            }
        }
    }

    // println!("Read set");
    // for &read_base in &read_bases {
    //     print_eclass(egraph, egraph[read_base].nodes[0].children()[0]);
    // }
    // println!("Write set");
    // for (wbase, _, _) in &write_info {
    //     print_eclass(egraph, egraph[*wbase].nodes[0].children()[0]);
    // }
    // println!("-------------------------");

    // println!("{:?}", read_bases.clone());
    // println!("{:?}", write_info.clone());

    // Mark dead stores
    for (wbase, eclass_id, enode) in write_info {
        let is_output = egraph[wbase].nodes.iter().any(|n| matches!(n, TileLang::Output(_)));
        if is_output {
            continue;
        }
        if !read_bases.contains(&wbase) {
            egraph[eclass_id].data.is_deleted.insert(enode.clone());
            // print_eclass(egraph, egraph[wbase].nodes[0].children()[0]);
            // println!("{:?}", enode);

            // let all_deleted = egraph[eclass_id].nodes.iter().all(|n| egraph[eclass_id].data.is_deleted.contains(n));
            let all_deleted = true;
            if all_deleted {
                let dummy = TileLang::NoneOp;
                let new_id = egraph.add(dummy);
                egraph.union(eclass_id, new_id);
            }
        }
    }

    egraph.rebuild();
}

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

pub fn lookup_expr_filtered(egraph: &EGraph, expr: &RecExpr<TileLang>) -> Option<Id> {
    fn helper(
        egraph: &EGraph,
        expr: &RecExpr<TileLang>,
        memo: &mut HashMap<Id, Id>,
        node_id: Id,
    ) -> Option<Id> {
        if let Some(&cached) = memo.get(&node_id) {
            return Some(cached);
        }

        let node = &expr[node_id];
        let child_ids: Vec<Id> = node
            .children()
            .iter()
            .map(|&i| helper(egraph, expr, memo, i))
            .collect::<Option<_>>()?;

        let new_node = TileLang::from_op(&node.to_string(), child_ids).ok()?;

        for class in egraph.classes() {
            let class_id = class.id;

            if class.data.is_deleted.len() > 0 {
                println!("{:?}", class.data.is_deleted)
            }

            if class.data.is_deleted.contains(&new_node) {
                continue;
            }

            if class.nodes.contains(&new_node) {
                memo.insert(node_id, class_id);
                return Some(class_id);
            }
        }
        None
    }

    let mut memo = HashMap::new();
    let root_id = Id::from(expr.as_ref().len() - 1);
    helper(egraph, expr, &mut memo, root_id)
}

#[macro_export]
macro_rules! test_fn_not {
    (
        $(#[$meta:meta])*
        $name:ident, $rules:expr,
        $(runner = $runner:expr,)?
        $start:literal
        !=
        $($goal:literal),+ $(,)?
        $(@check $check_fn:expr)?
    ) => {
        $(#[$meta])*
        #[test]
        pub fn $name() {
            use egg::{Runner, RecExpr, Rewrite};
            use crate::lookup_expr_filtered;

            let start_expr: RecExpr<_> = $start.parse().unwrap();
            let goal_exprs: Vec<RecExpr<_>> = vec![$($goal.parse().unwrap()),+];

            let mut runner = Runner::default()
                .with_expr(&start_expr)
                .with_iter_limit(max_iter)
                .with_node_limit(1000000)
                .with_time_limit(std::time::Duration::from_secs(1000))
                .with_hook(|runner| {
                    for &root in &runner.roots {
                        dead_code_elimination_pass(&mut runner.egraph, root);
                    }
                    Ok(())
                })
                .run(&$rules);
            for &root in &runner.roots {
                dead_code_elimination_pass(&mut runner.egraph, root);
            }
            runner.egraph.rebuild();

            for goal_expr in &goal_exprs {
                assert!(
                    lookup_expr_filtered(&runner.egraph, goal_expr).is_none(),
                    "Expected goal expr to be found, but it was not:\n{}",
                    goal_expr,
                );
            }
        }
    };
}

#[macro_export]
macro_rules! test_fn {
    (
        $(#[$meta:meta])*
        $name:ident, $rules:expr,
        $(runner = $runner:expr,)?
        $start:literal
        =>
        $($goal:literal),+ $(,)?
    ) => {
        $(#[$meta])*
        #[test]
        pub fn $name() {
            use egg::{Runner, RecExpr, Rewrite};
            use crate::lookup_expr_filtered;

            let start_expr: RecExpr<_> = $start.parse().unwrap();
            let goal_exprs: Vec<RecExpr<_>> = vec![$($goal.parse().unwrap()),+];
        
            let mut runner = Runner::default()
                .with_expr(&start_expr)
                .with_iter_limit(max_iter)
                .with_node_limit(1000000)
                .with_time_limit(std::time::Duration::from_secs(1000))
                // .with_hook(|runner| {
                //     for &root in &runner.roots {
                //         dead_code_elimination_pass(&mut runner.egraph, root);
                //     }
                //     Ok(())
                // })
                .run(&$rules);
            // for &root in &runner.roots {
            //     dead_code_elimination_pass(&mut runner.egraph, root);
            // }
            // runner.egraph.rebuild();

            for goal_expr in &goal_exprs {
                assert!(
                    lookup_expr_filtered(&runner.egraph, goal_expr).is_some(),
                    "Expected goal expr to be found, but it was not:\n{}",
                    goal_expr,
                );
            }
        }
    };
}

#[macro_export]
macro_rules! test_fn_without_deadcode {
    (
        $(#[$meta:meta])*
        $name:ident, $rules:expr,
        $(runner = $runner:expr,)?
        $start:literal
        =>
        $($goal:literal),+ $(,)?
        $(@check $check_fn:expr)?
    ) => {
        $(#[$meta])*
        #[test]
        pub fn $name() {
            use egg::{Runner, RecExpr, Rewrite};
            use crate::lookup_expr_filtered;

            let start_expr: RecExpr<_> = $start.parse().unwrap();
            let goal_exprs: Vec<RecExpr<_>> = vec![$($goal.parse().unwrap()),+];
        
            let mut runner = {
                let mut r = Runner::default()
                    .with_expr(&start_expr)
                    .with_node_limit(1000000000)
                    .with_iter_limit(max_iter)
                    .with_time_limit(std::time::Duration::from_secs(1000))
                    .run(&$rules);

                $( $check_fn(&r); )?

                r
            };

            for goal_expr in &goal_exprs {
                assert!(
                    lookup_expr_filtered(&runner.egraph, goal_expr).is_some(),
                    "Expected goal expr to be found, but it was not:\n{}",
                    goal_expr,
                );
            }
        }
    };
}

egg::test_fn2! {test_flashdecoding_skip_ft, rules(),
    "
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C) 
                    (*
                        (load (input Q) (index (tile m tile_m) (fulltile)))
                        (load (input K) (index (fulltile) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_exp)
                    (exp (load (input C) (index (tile m tile_m) (tile n tile_n))))
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_sum)
                    (+
                        (x (load (input C_sum) (index (tile m tile_m))) 1)
                        (rsum (load (input C_exp) (index (tile m tile_m) (tile n tile_n))) 1)
                    )
                    (index (tile m tile_m))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_div)
                    (/
                        (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                        (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input O)
                    (+
                        (x (load (input O) (index (tile m tile_m) (fulltile))) 1)
                        (*
                            (load (input C_div) (index (tile m tile_m) (tile n tile_n)))
                            (load (input V) (index (tile n tile_n) (fulltile)))
                        )
                    )
                    (index (tile m tile_m) (fulltile))
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
                                (load (input Q) (index (tile m tile_m) (fulltile)))
                                (load (input K) (index (fulltile) (tile n tile_n)))
                            )
                            (index (tile m tile_m) (tile n tile_n))
                        )
                        (store (input C_exp)
                            (exp 
                                (load (input C) (index (tile m tile_m) (tile n tile_n)))
                            )
                            (index (tile m tile_m) (tile n tile_n))
                        )
                    )
                )
            (seq
                (dloop 0 N new_tile_n new_n
                    (dloop new_n (+ new_n new_tile_n) tile_n n
                        (store (input new_C_sum)
                            (+
                                (x (load (input new_C_sum) (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m)))) 1)
                                (rsum
                                    (exp
                                        (*
                                            (load (input Q) (index (tile m tile_m) (fulltile)))
                                            (load (input K) (index (fulltile) (tile n tile_n)))
                                        )
                                    )
                                    1
                                )
                            )
                            (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m)))
                        )
                    )
                )
            (seq
                (store (input C_sum)
                    (rsum
                        (load (input new_C_sum) (index (fulltile) (index (tile m tile_m))))
                        0
                    )
                    (index (tile m tile_m))
                )

            (seq
                (dloop 0 N new_tile_n new_n
                    (dloop new_n (+ new_n new_tile_n) tile_n n
                        (store (input new_O)
                            (+
                                (x (load (input new_O) (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m) (fulltile)))) 1)
                                (*
                                    (exp
                                        (*
                                            (load (input Q) (index (tile m tile_m) (fulltile)))
                                            (load (input K) (index (fulltile) (tile n tile_n)))
                                        )
                                    )
                                    (load (input V) (index (tile n tile_n) (fulltile)))
                                )
                            )
                            (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m) (fulltile)))
                        )
                    )
                )
            
            
            (seq
                (store (input O)
                    (rsum
                        (load (input new_O) (index (fulltile) (index (tile m tile_m) (fulltile))))
                        0
                    )
                    (index (tile m tile_m) (fulltile))
                )
            (seq
                (store (input O)
                    (/
                        (load (input O) (index (tile m tile_m) (fulltile)))
                        (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                    )
                    (index (tile m tile_m) (fulltile))
                )
                (loop 0 N tile_n n
                    (store (input C_div)
                        (/
                            (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                            (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                        )
                        (index (tile m tile_m) (tile n tile_n))
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
                                (load (input Q) (index (tile m tile_m) (fulltile)))
                                (load (input K) (index (fulltile) (tile n tile_n)))
                            )
                            (index (tile m tile_m) (tile n tile_n))
                        )
                        (store (input C_exp)
                            (exp 
                                (load (input C) (index (tile m tile_m) (tile n tile_n)))
                            )
                            (index (tile m tile_m) (tile n tile_n))
                        )
                    )
                )
                (dloop 0 N new_tile_n new_n
                    (dloop new_n (+ new_n new_tile_n) tile_n n
                        (seq
                            (store (input new_C_sum)
                                (+
                                    (x (load (input new_C_sum) (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m)))) 1)
                                    (rsum
                                        (exp
                                            (*
                                                (load (input Q) (index (tile m tile_m) (fulltile)))
                                                (load (input K) (index (fulltile) (tile n tile_n)))
                                            )
                                        )
                                        1
                                    )
                                )
                                (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m)))
                            )

                            (store (input new_O)
                                (+
                                    (x (load (input new_O) (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m) (fulltile)))) 1)
                                    (*
                                        (exp
                                            (*
                                                (load (input Q) (index (tile m tile_m) (fulltile)))
                                                (load (input K) (index (fulltile) (tile n tile_n)))
                                            )
                                        )
                                        (load (input V) (index (tile n tile_n) (fulltile)))
                                    )
                                )
                                (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m) (fulltile)))
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
                        (load (input new_C_sum) (index (fulltile) (index (tile m tile_m))))
                        0
                    )
                    (index (tile m tile_m))
                )
            
            (seq
                (store (input O)
                    (rsum
                        (load (input new_O) (index (fulltile) (index (tile m tile_m) (fulltile))))
                        0
                    )
                    (index (tile m tile_m) (fulltile))
                )
            (seq
                (store (input O)
                    (/
                        (load (input O) (index (tile m tile_m) (fulltile)))
                        (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                    )
                    (index (tile m tile_m) (fulltile))
                )
                (loop 0 N tile_n n
                    (store (input C_div)
                        (/
                            (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                            (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                        )
                        (index (tile m tile_m) (tile n tile_n))
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
    //                             (load (input Q) (index (tile m tile_m) (fulltile)))
    //                             (load (input K) (index (fulltile) (tile n tile_n)))
    //                         )
    //                         (index (tile m tile_m) (tile n tile_n))
    //                     )
    //                     (store (input C_exp)
    //                         (exp 
    //                             (load (input C) (index (tile m tile_m) (tile n tile_n)))
    //                         )
    //                         (index (tile m tile_m) (tile n tile_n))
    //                     )
    //                 )
    //             )
    //             (loop 0 N new_tile_n new_n
    //                 (loop new_n (+ new_n new_tile_n) tile_n n
    //                     (seq
    //                         (store (input new_C_sum)
    //                             (+
    //                                 (x (load (input new_C_sum) (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m)))) 1)
    //                                 (rsum
    //                                     (exp
    //                                         (*
    //                                             (load (input Q) (index (tile m tile_m) (fulltile)))
    //                                             (load (input K) (index (fulltile) (tile n tile_n)))
    //                                         )
    //                                     )
    //                                     1
    //                                 )
    //                             )
    //                             (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m)))
    //                         )
    //                         (store (input new_O)
    //                             (+
    //                                 (x (load (input new_O) (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m) (fulltile)))) 1)
    //                                 (*
    //                                     (exp
    //                                         (*
    //                                             (load (input Q) (index (tile m tile_m) (fulltile)))
    //                                             (load (input K) (index (fulltile) (tile n tile_n)))
    //                                         )
    //                                     )
    //                                     (load (input V) (index (tile n tile_n) (fulltile)))
    //                                 )
    //                             )
    //                             (index (tile (/ new_n new_tile_n) 1) (index (tile m tile_m) (fulltile)))
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
    //                     (load (input new_C_sum) (index (fulltile) (index (tile m tile_m))))
    //                     0
    //                 )
    //                 (index (tile m tile_m))
    //             )
    //         (seq
    //             (store (output O)
    //                 (rsum
    //                     (load (input new_O) (index (fulltile) (index (tile m tile_m) (fulltile))))
    //                     0
    //                 )
    //                 (index (tile m tile_m) (fulltile))
    //             )
    //         (seq
    //             (store (output O)
    //                 (/
    //                     (load (output O) (index (tile m tile_m) (fulltile)))
    //                     (bcast (load (input C_sum) (index (tile m tile_m))) 1)
    //                 )
    //                 (index (tile m tile_m) (fulltile))
    //             )
    //             (loop 0 N tile_n n
    //                 (store (input C_div)
    //                     (/
    //                         (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
    //                         (bcast (load (input C_sum) (index (tile m tile_m))) 1)
    //                     )
    //                     (index (tile m tile_m) (tile n tile_n))
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
                        (load (input Q) (index (tile m tile_m) (fulltile)))
                        (load (input K) (index (fulltile) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_exp)
                    (exp (load (input C) (index (tile m tile_m) (tile n tile_n))))
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_sum)
                    (+
                        (x (load (input C_sum) (index (tile m tile_m))) 1)
                        (rsum (load (input C_exp) (index (tile m tile_m) (tile n tile_n))) 1)
                    )
                    (index (tile m tile_m))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_div)
                    (/
                        (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                        (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (output O)
                    (+
                        (x (load (output O) (index (tile m tile_m) (fulltile))) 1)
                        (*
                            (load (input C_div) (index (tile m tile_m) (tile n tile_n)))
                            (load (input V) (index (tile n tile_n) (fulltile)))
                        )
                    )
                    (index (tile m tile_m) (fulltile))
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
                            (load (input Q) (index (tile m tile_m) (fulltile)))
                            (load (input K) (index (fulltile) (tile n tile_n)))
                        )
                        (index (tile m tile_m) (tile n tile_n))
                    )
                (seq
                    (store (input C_exp)
                        (exp 
                            (load (input C) (index (tile m tile_m) (tile n tile_n)))
                        )
                        (index (tile m tile_m) (tile n tile_n))
                    )
                (seq
                    (store (input C_sum)
                        (+
                            (x (load (input C_sum) (index (tile m tile_m))) 1)
                            (rsum 
                                (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                                1
                            )
                        )
                        (index (tile m tile_m))
                    )
                    (store (output O)
                        (+
                            (x (load (output O) (index (tile m tile_m) (fulltile))) 1)
                            (*
                                (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                                (load (input V) (index (tile n tile_n) (fulltile)))
                            )
                        )
                        (index (tile m tile_m) (fulltile))
                    )
                )
                )
                )
            )
        (seq
            (store (output O)
                (/
                    (load (output O) (index (tile m tile_m) (fulltile)))
                    (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                )
                (index (tile m tile_m) (fulltile))
            )
            (loop 0 N tile_n n
                (store (input C_div)
                    (/
                        (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                        (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                    )
                    (index (tile m tile_m) (tile n tile_n))
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
    //                         (x (load (input C_sum) (index (tile m tile_m))) 1)
    //                         (rsum 
    //                             (exp 
    //                                 (*
    //                                     (load (input Q) (index (tile m tile_m) (fulltile)))
    //                                     (load (input K) (index (fulltile) (tile n tile_n)))
    //                                 )
    //                             )
    //                             1
    //                         )
    //                     )
    //                     (index (tile m tile_m))
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
    //                         (x (load (output O) (index (tile m tile_m) (fulltile))) 1)
    //                         (*
    //                             (/
    //                                 (exp 
    //                                     (*
    //                                         (load (input Q) (index (tile m tile_m) (fulltile)))
    //                                         (load (input K) (index (fulltile) (tile n tile_n)))
    //                                     )
    //                                 )
    //                                 (bcast (load (input C_sum) (index (tile m tile_m))) 1)
    //                             )
    //                             (load (input V) (index (tile n tile_n) (fulltile)))
    //                         )
    //                     )
    //                     (index (tile m tile_m) (fulltile))
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
    //                         (x (load (input C_sum) (index (tile m tile_m))) 1)
    //                         (rsum 
    //                             (exp 
    //                                 (*
    //                                     (load (input Q) (index (tile m tile_m) (fulltile)))
    //                                     (load (input K) (index (fulltile) (tile n tile_n)))
    //                                 )
    //                             )
    //                             1
    //                         )
    //                     )
    //                     (index (tile m tile_m))
    //                 )
    //                 (store (output O)
    //                     (+
    //                         (x (load (output O) (index (tile m tile_m) (fulltile))) 1)
    //                         (*
    //                             (exp 
    //                                 (*
    //                                     (load (input Q) (index (tile m tile_m) (fulltile)))
    //                                     (load (input K) (index (fulltile) (tile n tile_n)))
    //                                 )
    //                             )
    //                             (load (input V) (index (tile n tile_n) (fulltile)))
    //                         )
    //                     )
    //                     (index (tile m tile_m) (fulltile))
    //                 )
    //             )
    //             )
    //             )
    //         )
    //     (seq
    //         (store (output O)
    //             (/
    //                 (load (output O) (index (tile m tile_m) (fulltile)))
    //                 (bcast (load (input C_sum) (index (tile m tile_m))) 1)
    //             )
    //             (index (tile m tile_m) (fulltile))
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
                            (x (load (input C1) (index (tile m tile_m) (tile n tile_n))) 1)
                            (*
                                (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                            )
                        )
                     (index (tile m tile_m) (tile n tile_n))
                    )
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (input C1_exp) 
                    (exp
                        (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (input C2) 
                        (+
                            (x (load (input C2) (index (tile m tile_m) (tile n tile_n))) 1)
                            (*
                                (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                (load (input W2) (index (tile k tile_k) (tile n tile_n)))
                            )
                        )
                     (index (tile m tile_m) (tile n tile_n))
                    )
                )
            )
        )
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (output O) 
                    (x
                        (load (input C1_exp) (index (tile m tile_m) (tile n tile_n)))
                        (load (input C2) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
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
    //                             (x (load (input C1) (index (tile m tile_m) (tile n tile_n))) 1)
    //                             (*
    //                                 (load (input X) (index (tile m tile_m) (tile k tile_k)))
    //                                 (load (input W1) (index (tile k tile_k) (tile n tile_n)))
    //                             )
    //                         )
    //                     (index (tile m tile_m) (tile n tile_n))
    //                     )
    //                     (store (input C2) 
    //                         (+
    //                             (x (load (input C2) (index (tile m tile_m) (tile n tile_n))) 1)
    //                             (*
    //                                 (load (input X) (index (tile m tile_m) (tile k tile_k)))
    //                                 (load (input W2) (index (tile k tile_k) (tile n tile_n)))
    //                             )
    //                         )
    //                     (index (tile m tile_m) (tile n tile_n))
    //                     )
    //                 )
    //             )
    //         (seq
    //             (dummy)
    //             (store (output O) 
    //                 (x
    //                     (exp
    //                         (load (input C1) (index (tile m tile_m) (tile n tile_n)))
    //                     )
    //                     (load (input C2) (index (tile m tile_m) (tile n tile_n)))
    //                 )
    //                 (index (tile m tile_m) (tile n tile_n))
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
    //                         (x (load (input C1) (index (tile m tile_m) (tile n tile_n))) 1)
    //                         (*
    //                             (load (input X) (index (tile m tile_m) (tile k tile_k)))
    //                             (load (input W1) (index (tile k tile_k) (tile n tile_n)))
    //                         )
    //                     )
    //                     (index (tile m tile_m) (tile n tile_n))
    //                 )
    //             )
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (store (input C2) 
    //                     (+
    //                         (x (load (input C2) (index (tile m tile_m) (tile n tile_n))) 1)
    //                         (*
    //                             (load (input X) (index (tile m tile_m) (tile k tile_k)))
    //                             (load (input W2) (index (tile k tile_k) (tile n tile_n)))
    //                         )
    //                     )
    //                     (index (tile m tile_m) (tile n tile_n))
    //                 )
    //             )
            
    //         (seq
    //             (store (output O) 
    //                 (x
    //                     (exp
    //                         (load (input C1) (index (tile m tile_m) (tile n tile_n)))
    //                     )
    //                     (load (input C2) (index (tile m tile_m) (tile n tile_n)))
    //                 )
    //                 (index (tile m tile_m) (tile n tile_n))
    //             )
    //             (store (input C1_exp) 
    //                 (exp
    //                     (load (input C1) (index (tile m tile_m) (tile n tile_n)))
    //                 )
    //                 (index (tile m tile_m) (tile n tile_n))
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
    //                         (x (load (input C1) (index (tile m tile_m) (tile n tile_n))) 1)
    //                         (*
    //                             (load (input X) (index (tile m tile_m) (tile k tile_k)))
    //                             (load (input W1) (index (tile k tile_k) (tile n tile_n)))
    //                         )
    //                     )
    //                     (index (tile m tile_m) (tile n tile_n))
    //                 )
    //             )
    //         (seq
    //             (loop 0 K tile_k k 
    //                 (store (output O) 
    //                     (+
    //                         (x (load (output O) (index (tile m tile_m) (tile n tile_n))) 1)
    //                         (x
    //                             (*
    //                                 (load (input X) (index (tile m tile_m) (tile k tile_k)))
    //                                 (load (input W2) (index (tile k tile_k) (tile n tile_n)))
    //                             )
    //                             (exp
    //                                 (load (input C1) (index (tile m tile_m) (tile n tile_n)))
    //                             )
    //                         )
    //                     )
    //                     (index (tile m tile_m) (tile n tile_n))
    //                 )
    //             )
    //             (store (input C1_exp) 
    //                 (exp
    //                     (load (input C1) (index (tile m tile_m) (tile n tile_n)))
    //                 )
    //                 (index (tile m tile_m) (tile n tile_n))
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
                                (x (load (input C1) (index (tile m tile_m) (tile n tile_n))) 1)
                                (*
                                    (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                    (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                                )
                            )
                        (index (tile m tile_m) (tile n tile_n))
                        )
                        (store (input C2) 
                            (+
                                (x (load (input C2) (index (tile m tile_m) (tile n tile_n))) 1)
                                (*
                                    (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                    (load (input W2) (index (tile k tile_k) (tile n tile_n)))
                                )
                            )
                        (index (tile m tile_m) (tile n tile_n))
                        )
                    )
                )
            (seq
                (store (input C1_exp) 
                    (exp
                        (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
                (store (output O) 
                    (x
                        (load (input C1_exp) (index (tile m tile_m) (tile n tile_n)))
                        (load (input C2) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
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
                    (+ (x (load (input C) (index (fulltile) (tile p tile_p))) 1)
                    (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input W) (index (tile n tile_n) (tile p tile_p)))
                    ))
                    (index (fulltile) (tile p tile_p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input A) (index (tile n tile_n) (fulltile)))
                ))
                (index (fulltile) (fulltile))
            )
        )

    (seq
        (loop 0 P tile_p p 
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p tile_p)))
                )
                (index (fulltile) (tile p tile_p))
            )
        )

        (loop 0 P tile_p p
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p tile_p))) (load (input E) (index (fulltile) (tile p tile_p))))
                (index (fulltile) (tile p tile_p))
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
                (seq
                    (store (input C)
                        (+ (x (load (input C) (index (fulltile) (tile p tile_p))) 1)
                        (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input W) (index (tile n tile_n) (tile p tile_p)))
                        ))
                        (index (fulltile) (tile p tile_p))
                    )
                    (store (input D)
                        (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                        (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input A) (index (tile n tile_n) (fulltile)))
                        ))
                        (index (fulltile) (fulltile))
                    )
                )
            )
        (seq
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p tile_p))))
                (index (fulltile) (tile p tile_p))
            )
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p tile_p))) (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p tile_p)))))
                (index (fulltile) (tile p tile_p))
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
                        (+ (x (load (input C) (index (fulltile) (tile p tile_p))) 1)
                        (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input W) (index (tile n tile_n) (tile p tile_p)))
                        ))
                        (index (fulltile) (tile p tile_p))
                    )
                    (store (input E)
                        (+ (x (load (input E) (index (fulltile) (tile p tile_p))) 1)
                            (* (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input A) (index (tile n tile_n) (fulltile)))) (load (input B) (index (fulltile) (tile p tile_p))))
                        )
                        (index (fulltile) (tile p tile_p))
                    )
                )
            )

            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p tile_p))) (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p tile_p)))))
                (index (fulltile) (tile p tile_p))
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
                    (x (load (output O) (index (fulltile) (tile p tile_p))) 1)
                    (*
                        (concat
                            (load (input X) (index (fulltile) (tile n tile_n)))
                            (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input A) (index (tile n tile_n) (fulltile))))
                            1
                        )
                        (concat
                            (load (input W) (index (tile n tile_n) (tile p tile_p)))
                            (load (input B) (index (fulltile) (tile p tile_p)))
                            0
                        )
                    )
                )
                (index (fulltile) (tile p tile_p))
            )
        )   
    )
    "
    ,
}
egg::test_fn2! {sequence, rules(),
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
        (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
        (loop 0 N tile_n n (store (output C) (load (input B) (index (tile n tile_n))) (index (tile n tile_n))))
    )"
    =>
    "
    (loop 0 N tile_n n
        (seq 
            (store (input B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            (store (output C) (load (input B) (index (tile n tile_n))) (index (tile n tile_n)))
        )
    )
    "
    ,
    // "
    // (loop 0 N tile_n n
    //     (seq 
    //         (dummy)  
    //         (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
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
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                        3
                    )
                    (index (tile n tile_n) (tile m tile_m))
                )
            )    
        )

        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n) (tile m tile_m)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                        3
                    )
                    (index (tile n tile_n) (tile m tile_m))
                )
                (store (output C) 
                    (+ 
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                        (+ 2 3)
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input B) (index (tile n tile_n)))
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                    )
                    (index (tile n tile_n))
                )
            )    
        )

        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input B) (index (tile n tile_n)))
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                    )
                    (index (tile n tile_n))
                )
            ) 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input B) (index (tile n tile_n)))
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                    )
                    (index (tile n tile_n))
                )
            )    
        )
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input B) (index (tile n tile_n)))
                        (load (input A) (index (tile n tile_n) (tile m tile_m)))
                    )
                    (index (tile n tile_n))
                )
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input A) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n))
                )
            )
        )
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
    //                     (load (input B) (index (tile n tile_n)))
    //                     2
    //                 )
    //                 (index (tile n tile_n) (tile m tile_m))
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
    //                 (load (input A) (index (tile n tile_n)))
    //                 2
    //             )
    //             (index (tile n tile_n))
    //         )
    //         (loop M tile_m m 
    //             (store (input C) 
    //                 (+ 
    //                     (load (input B) (index (tile n tile_n)))
    //                     2
    //                 )
    //                 (index (tile n tile_n) (tile m tile_m))
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
    //                     (load (input A) (index (tile n tile_n)))
    //                     2
    //                 )
    //                 (index (tile n tile_n))
    //             )
    //             (store (input C) 
    //                 (+ 
    //                     (load (input B) (index (tile n tile_n)))
    //                     2
    //                 )
    //                 (index (tile n tile_n) (tile m tile_m))
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
                        (load (input A) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n))
                )
                (store (output C) 
                    (+ 
                        (+ (load (input A) (index (tile n tile_n))) 2)
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                    (load (input A) (index (tile n tile_n)))
                )
                (index (tile n tile_n))
            )
        )
        (loop 0 N tile_n n 
            (loop 0 M tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n tile_n)))
                        2
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                        (load (input A) (index (tile n tile_n)))
                    )
                    (index (tile n tile_n))
                )
                (store (output C) 
                    (+ 
                        (load (input A) (index (tile n tile_n)))
                        (+ 2 2)
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                    (load (input A) (index (tile n tile_n)))
                )
                (index (tile n tile_n))
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
                (load (input A) (index (tile n tile_n)))
            )
            (index (tile n tile_n))
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
                    (load (output B) (index (tile n tile_n)))
                    (load (input A) (index (tile n tile_n) (tile m tile_m)))
                )
                (index (tile n tile_n))
            )
        )    
    )
    "
    !=
    "
    (loop 0 N tile_n n 
        (store (output B) 
            (+ 
                (load (output B) (index (tile n tile_n)))
                (load (input A) (index (tile n tile_n) (tile m tile_m)))
            )
            (index (tile n tile_n))
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
            (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
            (loop 0 N tile_n n (store (input C) (load (input B) (index (tile n tile_n))) (index (tile n tile_n))))
        )
        (loop 0 N tile_n n (store (output D) (load (input C) (index (tile n tile_n))) (index (tile n tile_n))))
    )
    "
    =>
    
    "
    (loop 0 N tile_n n 
        (seq
            (store (input B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            (seq
                (store (input C) (load (input B) (index (tile n tile_n))) (index (tile n tile_n)))
                (store (output D) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            )
        )
    )
    "
}
egg::test_fn2! {seq_comm1, rules(),
    "
    (seq
        (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
    )
    "
    =>
    "
    (seq
        (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
    )
    "
}
egg::test_fn2! {seq_comm2, rules(),
    "
    (seq
        (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        (seq
            (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            (store (output D) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        )
    )
    "
    =>
    "
    (seq
        (store (output D) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        (seq
            (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
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
                            (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                            (*
                                (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                            )
                        )
                     (index (tile m tile_m) (tile n tile_n))
                    )
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (output C1_exp) 
                    (exp
                        (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
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
                                (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                                (*
                                    (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                    (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                                )
                            )
                         (index (tile m tile_m) (tile n tile_n))
                        )
                    )
                    (store (output C1_exp) 
                        (exp
                            (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                        )
                        (index (tile m tile_m) (tile n tile_n))
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
                            (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                            (*
                                (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                            )
                        )
                     (index (tile m tile_m) (tile n tile_n))
                    )
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (output C1_exp) 
                    (exp
                        (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
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
                                (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                                (*
                                    (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                    (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                                )
                            )
                            (index (tile m tile_m) (tile n tile_n))
                        )
                        (store (output C1_exp) 
                            (exp
                                (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                            )
                            (index (tile m tile_m) (tile n tile_n))
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
        (loop 0 N tile_n n (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
    (seq
        (loop 0 N tile_n n (store (output C) (load (output B) (index (tile n tile_n))) (index (tile n tile_n))))
    (seq
        (loop 0 N tile_n n (store (output D) (load (output C) (index (tile n tile_n))) (index (tile n tile_n))))
        (loop 0 N tile_n n (store (output E) (load (output D) (index (tile n tile_n))) (index (tile n tile_n))))
    )
    )
    )"
    =>
    "
    (loop 0 N tile_n n
        (seq 
            (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        (seq
            (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
        (seq
            (store (output D) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            (store (output E) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
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
                (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
                (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            )
        )
        (loop 0 N tile_n n
            (seq
                (store (output D) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
                (store (output E) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            )
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
    (seq
        (loop 0 N tile_n n
            (seq
                (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
                (store (output D) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
            )
        )
        (loop 0 N tile_n n (store (output E) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
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
        (loop 0 N tile_n n (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
        (loop 0 M tile_m m (store (output D) (load (input C) (index (tile m tile_m))) (index (tile m tile_m))))
    )
    "
    =>
    "
    (loop 0 N tile_n n 
        (loop 0 M tile_m m
            (seq
                (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))
                (store (output D) (load (input C) (index (tile m tile_m))) (index (tile m tile_m)))
            )
        
        )
    )
    "
    ,
    "
    (seq
        (loop 0 N tile_n n 
            (loop 0 M tile_m m
                (store (output B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n)))            
            )
        )
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (output D) (load (input C) (index (tile m tile_m))) (index (tile m tile_m)))
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
            (store (output B) (+ (load (output B) (index)) (load (input A) (index (tile n tile_n)))) (index))
        )
        (loop 0 M tile_m m
            (store (output D) (* (load (input E) (index)) (load (input C) (index (tile m tile_m)))) (index (tile m tile_m)))
        )
    )
    "
    =>
    "
    (loop 0 M tile_m m
        (loop 0 N tile_n n
            (seq
                (store (output B) (+ (load (output B) (index)) (load (input A) (index (tile n tile_n)))) (index))
                (store (output D) (* (load (input E) (index)) (load (input C) (index (tile m tile_m)))) (index (tile m tile_m)))
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
        (store (output B) (+ (load (input A) (index (tile n tile_n))) 2) (index (tile n tile_n)))
        (store (output C) (* (load (output B) (index (tile n tile_n))) 3) (index (tile n tile_n)))
    )
    "
    =>
    "
    (seq
        (store (output B) (+ (load (input A) (index (tile n tile_n))) 2) (index (tile n tile_n)))
        (store (output C) (* (+ (load (input A) (index (tile n tile_n))) 2) 3) (index (tile n tile_n)))
    )
    "
    ,
    "
    (seq
        (store (output B) (+ (load (input A) (index (tile n tile_n))) 2) (index (tile n tile_n)))
        (store (output C) (* (+ (load (input A) (index (tile n tile_n))) 2) 3) (index (tile n tile_n)))
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
                (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n tile_n)))) (index))
                (store (output D) (* (load (input A) (index (tile n tile_n))) (load (output C) (index (tile n tile_n)))) (index (tile n tile_n)))
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
                (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n tile_n)))) (index))
                (store (output D) (* (load (input A) (index (tile n tile_n))) (+ (load (input A) (index (tile n tile_n))) 1)) (index (tile n tile_n)))
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
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n tile_n)))) (index))
            (seq
                (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
                (store (output D) (* (load (input A) (index (tile n tile_n))) (+ (load (input A) (index (tile n tile_n))) 1)) (index (tile n tile_n)))
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
            (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n tile_n)))) (index))
        )
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
                (store (output D) (* (load (input A) (index (tile n tile_n))) (+ (load (input A) (index (tile n tile_n))) 1)) (index (tile n tile_n)))
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
            (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n tile_n)))) (index))
        )
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
                (store (output D) (* (load (input A) (index (tile n tile_n))) (+ (load (input A) (index (tile n tile_n))) 1)) (index (tile n tile_n)))
            )
        )
        (store (output E) (x (load (output B) (index)) 10) (index))
    ))
    "
    ,
    "
    (loop 0 N tile_n n 
        (seq
            (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
        (seq
            (store (output E) (+ (x (load (output E) (index)) 3) (x 10 (load (input A) (index (tile n tile_n))))) (index))
            (store (output D) (* (load (input A) (index (tile n tile_n))) (+ (load (input A) (index (tile n tile_n))) 1)) (index (tile n tile_n)))
        )
        )
    )
    "
}

egg::test_fn2! {loop_factor1, rules(),
    "
    (loop 0 N tile_n n 
        (seq
            (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
        (seq
            (store (output B) (+ (x (load (output B) (index)) 3) (x 10 (load (input A) (index (tile n tile_n))))) (index))
            (store (output D) (* (load (input A) (index (tile n tile_n))) (load (output C) (index (tile n tile_n)))) (index (tile n tile_n)))
        )
        )
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (seq
                (store (output C) (+ (load (input A) (index (tile n tile_n))) 1) (index (tile n tile_n)))
            (seq
                (store (output B) (+ (x (load (output B) (index)) 3) (load (input A) (index (tile n tile_n)))) (index))
                (store (output D) (* (load (input A) (index (tile n tile_n))) (+ (load (input A) (index (tile n tile_n))) 1)) (index (tile n tile_n)))
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
                        (x (load (input B) (index (tile n tile_n))) 1)
                        (rsum (load (input A) (index (tile n tile_n) (tile m tile_m))) 1)
                    )
                    (index (tile n tile_n))
                )
            )
        )
        (loop 0 N tile_n n
            (loop 0 M tile_m m
                (seq
                    (store (input D)
                        (/
                            (load (input C) (index (tile n tile_n) (tile m tile_m)))
                            (bcast (load (input B) (index (tile n tile_n))) 1)
                        )
                        (index (tile n tile_n) (tile m tile_m))
                    )
                    (store (input E)
                        (+
                            (x (load (input E) (index (tile n tile_n) (fulltile))) 1)
                            (*
                                (load (input D) (index (tile n tile_n) (tile m tile_m)))
                                (load (input F) (index (tile m tile_m) (fulltile)))
                            )
                        )
                        (index (tile n tile_n) (fulltile))
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
                            (x (load (input B) (index (tile n tile_n))) 1)
                            (rsum (load (input A) (index (tile n tile_n) (tile m tile_m))) 1)
                        )
                        (index (tile n tile_n))
                    )
                    (store (input E)
                        (+
                            (x (load (input E) (index (tile n tile_n) (fulltile))) 1)
                            (*
                                (load (input C) (index (tile n tile_n) (tile m tile_m)))
                                (load (input F) (index (tile m tile_m) (fulltile)))
                            )
                        )
                        (index (tile n tile_n) (fulltile))
                    )
                )
            )
        (seq
            (store (input E)
                (/
                    (load (input E) (index (tile n tile_n) (fulltile)))
                    (bcast (load (input B) (index (tile n tile_n))) 1)
                )
                (index (tile n tile_n) (fulltile))
            )
            (loop 0 M tile_m m
                (store (input D)
                    (/
                        (load (input C) (index (tile n tile_n) (tile m tile_m)))
                        (bcast (load (input B) (index (tile n tile_n))) 1)
                    )
                    (index (tile n tile_n) (tile m tile_m))
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
                            (x (load (input B) (index (tile n tile_n))) 1)
                            (rsum (load (input A) (index (tile n tile_n) (tile m tile_m))) 1)
                        )
                        (index (tile n tile_n))
                    )
                    (store (input E)
                        (+
                            (x (load (input E) (index (tile n tile_n) (fulltile))) 1)
                            (*
                                (load (input C) (index (tile n tile_n) (tile m tile_m)))
                                (load (input F) (index (tile m tile_m) (fulltile)))
                            )
                        )
                        (index (tile n tile_n) (fulltile))
                    )
                )
            )
        (seq
            (loop 0 M tile_m m
                (store (input D)
                    (/
                        (load (input C) (index (tile n tile_n) (tile m tile_m)))
                        (bcast (load (input B) (index (tile n tile_n))) 1)
                    )
                    (index (tile n tile_n) (tile m tile_m))
                )
            )
            (store (input E)
                (/
                    (load (input E) (index (tile n tile_n) (fulltile)))
                    (bcast (load (input B) (index (tile n tile_n))) 1)
                )
                (index (tile n tile_n) (fulltile))
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
            (store (input B) (+ (x (load (input B) (index)) 1) (x 10 (rsum (load (input A) (index (tile n tile_n))) 1))) (index))
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
                        (x (load (input new_B) (index (tile (/ new_n new_tile_n) 1) (index))) 1)
                        (x 10 (rsum (load (input A) (index (tile n tile_n))) 1))
                    )
                    (index (tile (/ new_n new_tile_n) 1) (index))
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (load (output B) (index (tile n tile_n))) (load (input C) (index (tile n tile_n))))
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n tile_n)))) (load (input C) (index (tile n tile_n))))
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n tile_n)))) (load (input C) (index (tile n tile_n))))
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
    //                 (* (x 2 (load (input A) (index (tile n tile_n)))) (load (input C) (index (tile n tile_n))))
    //             )
    //             (index)
    //         )
    //     )
    //     (loop 0 N tile_n n 
    //         (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
    //     )
    // )
    // "
}
egg::test_fn2! {seq_comm3, rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (* (x 2 (load (input A) (index (tile n tile_n)))) (load (input C) (index (tile n tile_n))))
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
                    (* (x 2 (load (input A) (index (tile n tile_n)))) (load (input C) (index (tile n tile_n))))
                )
                (index)
            )
        )
        (loop 0 N tile_n n 
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (load (output B) (index (tile n tile_n))) (load (input C) (index (tile n tile_n))))
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (x 2 (load (input A) (index (tile n tile_n)))) (load (input C) (index (tile n tile_n))))
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (x (load (input A) (index (tile n tile_n))) (load (input C) (index (tile n tile_n)))) 2)
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
        )
        (loop 0 N tile_n n 
            (store (output D) 
                (+ (x (load (output D) (index)) 1)
                    (x (x (load (input A) (index (tile n tile_n))) (load (input C) (index (tile n tile_n)))) 2)
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
            (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
        )
        (seq
            (loop 0 N tile_n n 
                (store (output D) 
                    (+ (x (load (output D) (index)) 1)
                        (x (load (input A) (index (tile n tile_n))) (load (input C) (index (tile n tile_n))))
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
                (store (output B) (x 2 (load (input A) (index (tile n tile_n)))) (index (tile n tile_n)))
                (store (output D) 
                    (+ (x (load (output D) (index)) 1)
                        (x (load (input A) (index (tile n tile_n))) (load (input C) (index (tile n tile_n))))
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
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n tile_n)))) (index))
        )
        (loop 0 N tile_n n
            (store (output D) (+ (x (load (output D) (index)) 1) (/ (load (input C) (index (tile n tile_n))) (load (output B) (index)))) (index))
        )
    )
    "
    =>
    "
    (seq
        (loop 0 N tile_n n 
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n tile_n)))) (index))
        )
    (seq
        (loop 0 N tile_n n
            (store (output D) (+ (x (load (output D) (index)) 1) (load (input C) (index (tile n tile_n)))) (index))
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
                (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n tile_n)))) (index))
                (store (output D) (+ (x (load (output D) (index)) 1) (load (input C) (index (tile n tile_n)))) (index))
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
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n tile_n)))) (index))
        )
        (loop 0 N tile_n n
            (store (output D) (+ (x (load (output D) (index)) 1) (/ (load (input C) (index (tile n tile_n))) (load (output B) (index)))) (index))
        )
    )
    "
    !=
    "
    (loop 0 N tile_n n
        (seq
            (store (output B) (+ (x (load (output B) (index)) 1) (load (input A) (index (tile n tile_n)))) (index))
            (store (output D) (+ (x (load (output D) (index)) 1) (/ (load (input C) (index (tile n tile_n))) (load (output B) (index)))) (index))
        )
    )
    "
}

egg::test_fn2! {equivalent, rules(),
    "(seq 
        (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
        (loop 0 N tile_n n (store (output C) (load (input B) (index (tile n tile_n))) (index (tile n tile_n))))
    )"
    =>
    "(seq 
        (loop 0 N tile_n n (store (input B) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
        (loop 0 N tile_n n (store (output C) (load (input A) (index (tile n tile_n))) (index (tile n tile_n))))
    )"
}

egg::test_fn2! {loop_split2, custom_rules(),
    "
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C1) 
                    (+
                        (x (load (input C1) (index (fulltile) (tile n tile_n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k tile_k)))
                            (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                        )
                    )
                    (index (fulltile) (tile n tile_n))
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
                        (x (load (input C1) (index (fulltile) (tile n tile_n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k tile_k)))
                            (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                        )
                    )
                    (index (fulltile) (tile n tile_n))
                )
            )
        )
        (store (input A) (+ (load (input B) (index)) 1) (index))
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

use std::fs::File;
use std::io::Write;

#[test]
fn visualizer() {
    let expr = "
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C1) 
                    (+
                        (x (load (input C1) (index (fulltile) (tile n tile_n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k tile_k)))
                            (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                        )
                    )
                    (index (fulltile) (tile n tile_n))
                )
            )
        )
        (store (input A) (+ (load (input B) (index)) 1) (index))
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
                        (x (load (input C1) (index (fulltile) (tile n tile_n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k tile_k)))
                            (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                        )
                    )
                    (index (fulltile) (tile n tile_n))
                )
            )
        )
        
    (seq
        (loop 0 N tile_n n 
            (store (input C1_exp) 
                (exp
                    (load (input C1) (index (fulltile) (tile n tile_n)))
                )
                (index (fulltile) (tile n tile_n))
            )
        )
    
    (seq
        (loop 0 N tile_n n 
            (loop 0 K tile_k k 
                (store (input C2) 
                    (+
                        (x (load (input C2) (index (fulltile) (tile n tile_n))) 1)
                        (*
                            (load (input X) (index (fulltile) (tile k tile_k)))
                            (load (input W2) (index (tile k tile_k) (tile n tile_n)))
                        )
                    )
                    (index (fulltile) (tile n tile_n))
                )
            )
        )
        (loop 0 N tile_n n 
            (store (output O) 
                (x
                    (load (input C1_exp) (index (fulltile) (tile n tile_n)))
                    (load (input C2) (index (fulltile) (tile n tile_n)))
                )
                (index (fulltile) (tile n tile_n))
            )
        )
    )
    )
    )
    ";
    let runner = run_until_saturated(
        expr,
        custom_rules(),
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
                            (x (load (input C1) (index (tile m tile_m) (tile n tile_n))) 1)
                            (*
                                (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                (load (input W1) (index (tile k tile_k) (tile n tile_n)))
                            )
                        )
                     (index (tile m tile_m) (tile n tile_n))
                    )
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (input C1_exp) 
                    (exp
                        (load (input C1) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    
    (seq
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (loop 0 K tile_k k 
                    (store (input C2) 
                        (+
                            (x (load (input C2) (index (tile m tile_m) (tile n tile_n))) 1)
                            (*
                                (load (input X) (index (tile m tile_m) (tile k tile_k)))
                                (load (input W2) (index (tile k tile_k) (tile n tile_n)))
                            )
                        )
                     (index (tile m tile_m) (tile n tile_n))
                    )
                )
            )
        )
        (loop 0 M tile_m m 
            (loop 0 N tile_n n 
                (store (output O) 
                    (x
                        (load (input C1_exp) (index (tile m tile_m) (tile n tile_n)))
                        (load (input C2) (index (tile m tile_m) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
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
    // let check_expr: RecExpr<TileLang> = "(load (input C1_exp) (index (tile m tile_m) (tile n tile_n)))".parse().unwrap();

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
                        (+ (x (load (input C) (index (tile m tile_m) (tile p tile_p))) 1)
                        (* (load (input X) (index (tile m tile_m) (tile n tile_n))) (load (input W) (index (tile n tile_n) (tile p tile_p)))
                        ))
                        (index (tile m tile_m) (tile p tile_p))
                    )
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 K tile_k k 
                (loop 0 N tile_n n
                    (store (input D)
                        (+ (x (load (input D) (index (tile m tile_m) (tile k tile_k))) 1)
                        (* (load (input X) (index (tile m tile_m) (tile n tile_n))) (load (input A) (index (tile n tile_n) (tile k tile_k)))
                        ))
                        (index (tile m tile_m) (tile k tile_k))
                    )
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 P tile_p p 
                (loop 0 K tile_k k
                    (store (input E)
                        (+ (x (load (input E) (index (tile m tile_m) (tile p tile_p))) 1)
                        (* (load (input D) (index (tile m tile_m) (tile k tile_k))) (load (input B) (index (tile k tile_k) (tile p tile_p)))
                        ))
                        (index (tile m tile_m) (tile p tile_p))
                    )
                )
            )
        )

        (loop 0 M tile_m m
            (loop 0 P tile_p p
                (store (output O)
                    (+ (load (input C) (index (tile m tile_m) (tile p tile_p))) (load (input E) (index (tile m tile_m) (tile p tile_p))))
                    (index (tile m tile_m) (tile p tile_p))
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
                    (+ (x (load (input C) (index (fulltile) (tile p tile_p))) 1)
                    (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input W) (index (tile n tile_n) (tile p tile_p)))
                    ))
                    (index (fulltile) (tile p tile_p))
                )
            )
        )
    (seq
        (loop 0 N tile_n n
            (store (input D)
                (+ (x (load (input D) (index (fulltile) (fulltile))) 1)
                (* (load (input X) (index (fulltile) (tile n tile_n))) (load (input A) (index (tile n tile_n) (fulltile)))
                ))
                (index (fulltile) (fulltile))
            )
        )

    (seq
        (loop 0 P tile_p p 
            (store (input E)
                (* (load (input D) (index (fulltile) (fulltile))) (load (input B) (index (fulltile) (tile p tile_p)))
                )
                (index (fulltile) (tile p tile_p))
            )
        )

        (loop 0 P tile_p p
            (store (output O)
                (+ (load (input C) (index (fulltile) (tile p tile_p))) (load (input E) (index (fulltile) (tile p tile_p))))
                (index (fulltile) (tile p tile_p))
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
fn saturate_flashattn2_skip_ft() {
    let expr = "
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C) 
                    (*
                        (load (input Q) (index (tile m tile_m) (fulltile)))
                        (load (input K) (index (fulltile) (tile n tile_n)))
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_exp)
                    (exp (load (input C) (index (tile m tile_m) (tile n tile_n))))
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_sum)
                    (+
                        (x (load (input C_sum) (index (tile m tile_m))) 1)
                        (rsum (load (input C_exp) (index (tile m tile_m) (tile n tile_n))) 1)
                    )
                    (index (tile m tile_m))
                )
            )
        )
    (seq
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (input C_div)
                    (/
                        (load (input C_exp) (index (tile m tile_m) (tile n tile_n)))
                        (bcast (load (input C_sum) (index (tile m tile_m))) 1)
                    )
                    (index (tile m tile_m) (tile n tile_n))
                )
            )
        )
        (loop 0 M tile_m m
            (loop 0 N tile_n n 
                (store (output O)
                    (+
                        (x (load (output O) (index (tile m tile_m) (fulltile))) 1)
                        (*
                            (load (input C_div) (index (tile m tile_m) (tile n tile_n)))
                            (load (input V) (index (tile n tile_n) (fulltile)))
                        )
                    )
                    (index (tile m tile_m) (fulltile))
                )
            )
        )
    ))))
    ";
    let runner = run_until_saturated(
        expr,
        rules(),
    );
}


fn save_egraph(runner: &egg::Runner<TileLang, LoopAnalysis>, filename: &str) {
    let dot_string = runner.egraph.dot().to_string();
    let mut file = File::create(filename).expect("Failed to create dot file");
    file.write_all(dot_string.as_bytes()).expect("Failed to write dot file");
}