use trinity::*;
use trinity::language::{TileLang, LoopAnalysis, SHAPE_TRACKER};
use trinity::shape::{ShapeTracker, TensorShape, Dimension};
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

// Use Once to ensure shape setup happens before test execution
static INIT_FUSIBLE1: Once = Once::new();
static INIT_FUSIBLE2: Once = Once::new();

// Custom rules function that sets up shapes before returning rules
fn rules_with_shapes_fusible1() -> Vec<egg::Rewrite<TileLang, LoopAnalysis>> {
    INIT_FUSIBLE1.call_once(|| {
        setup_shape_tracker(vec![("A", vec![128, 256]), ("B", vec![128, 256]), ("C", vec![128, 256])]);
    });
    rules()
}
// Custom rules function that sets up shapes before returning rules
fn rules_with_shapes_fusible2() -> Vec<egg::Rewrite<TileLang, LoopAnalysis>> {
    INIT_FUSIBLE2.call_once(|| {
        setup_shape_tracker(vec![("A", vec![128, 256]), ("B", vec![128, 256]), ("C", vec![128, 256]), ("D", vec![128, 512])]);
    });
    rules()
}

egg::test_fn2! {fusible1_with_shape, rules_with_shapes_fusible1(),
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
        (loop 0 128 tile_n n 
            (loop 0 256 tile_m m 
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile n) (tile m)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n) (tile m))
                )
            )    
        )

        (loop 0 128 tile_n n 
            (loop 0 256 tile_m m 
                (store (output C) 
                    (+ 
                        (load (input B) (index (tile n) (tile m)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n) (tile m))
                )
            )    
        )
    )
    "
    =>
    "
    (loop 0 128 tile_n n 
        (loop 0 256 tile_m m 
            (seq
                (store (input B) 
                    (+ 
                        (load (input A) (index (tile n) (tile m)))
                        (load (input A) (index (tile n) (tile m)))
                    )
                    (index (tile n) (tile m))
                )
                (store (output C) 
                    (+ 
                        (load (input A) (index (tile n) (tile m)))
                        (+ (load (input A) (index (tile n) (tile m))) (load (input A) (index (tile n) (tile m))))
                    )
                    (index (tile n) (tile m))
                )
            )
        )    
    )
    "
}

egg::test_fn2! {fusible2_with_shape, rules_with_shapes_fusible2(),
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
        (loop 0 128 tile_n n 
            (store (input B) 
                (+ 
                    (load (input D) (index (tile n) (fulltile)))
                    (load (input A) (index (tile n) (fulltile)))
                )
                (index (tile n) (fulltile))
            )
        )

        (loop 0 128 tile_n n 
            (store (output C) 
                (+ 
                    (load (input B) (index (tile n) (fulltile)))
                    (load (input A) (index (tile n) (fulltile)))
                )
                (index (tile n) (fulltile))
            )
        )
    )
    "
    =>
    "
    (loop 0 128 tile_n n 
        (seq
            (store (input B) 
                (+ 
                    (load (input D) (index (tile n) (fulltile)))
                    (load (input A) (index (tile n) (fulltile)))
                )
                (index (tile n) (fulltile))
            )
            (store (output C) 
                (+ 
                    (load (input D) (index (tile n) (fulltile)))
                    (+ (load (input A) (index (tile n) (fulltile))) (load (input A) (index (tile n) (fulltile))))
                )
                (index (tile n) (fulltile))
            )
        )
    )
    "
}
