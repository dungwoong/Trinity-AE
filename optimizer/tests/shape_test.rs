use trinity::*;
use trinity::language::{TileLang, LoopAnalysis, SHAPE_TRACKER};
use trinity::shape::{ShapeTracker, TensorShape, Dimension};
use egg::*;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_shape_basic() {
        // Set up shape tracker with a tensor A of shape [128, 256]
        setup_shape_tracker(vec![("A", vec![128, 256])]);

        
        // Test: (load A (index (tile i) (tile j))) should have shape [64, 64]
        test_fn2! {test_name,
            rules(),
            "(load (input A) (index (tile i) (tile j)))" =>
            "(load (input A) (index (tile i) (tile j)))"
        }

        // Create egraph and parse expression
        let expr = "(load (input A) (index (tile i) (tile j)))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        // Check the shape
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 2);
        assert_eq!(shape.dims[0], Dimension::Concrete(64)); // min(128, 64)
        assert_eq!(shape.dims[1], Dimension::Concrete(64)); // min(256, 64)
    }

    #[test]
    fn test_load_shape_fulltile() {
        setup_shape_tracker(vec![("B", vec![100, 200, 300])]);

        let expr = "(load (input B) (index fulltile (tile i) fulltile))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 3);
        assert_eq!(shape.dims[0], Dimension::Concrete(100)); // fulltile
        assert_eq!(shape.dims[1], Dimension::Concrete(64));  // min(200, 64)
        assert_eq!(shape.dims[2], Dimension::Concrete(300)); // fulltile
    }

    #[test]
    fn test_add_shape() {
        setup_shape_tracker(vec![
            ("A", vec![128, 256]),
            ("B", vec![128, 256])
        ]);

        let expr = "(+ (load (input A) (index (tile i) (tile j))) 
                       (load (input B) (index (tile i) (tile j))))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 2);
        assert_eq!(shape.dims[0], Dimension::Concrete(64));
        assert_eq!(shape.dims[1], Dimension::Concrete(64));
    }

    #[test]
    fn test_matmul_shape() {
        setup_shape_tracker(vec![
            ("A", vec![128, 256]),  // M x K
            ("B", vec![256, 512])   // K x N
        ]);

        let expr = "(* (load (input A) (index (tile i) (tile k)))
                       (load (input B) (index (tile k) (tile j))))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 2);
        assert_eq!(shape.dims[0], Dimension::Concrete(64)); // M dimension
        assert_eq!(shape.dims[1], Dimension::Concrete(64)); // N dimension
    }

    #[test]
    fn test_broadcast_wildcard() {
        setup_shape_tracker(vec![("A", vec![128, 256])]);

        let expr = "(bcast (load (input A) (index (tile i) (tile j))) 0)".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 3);
        assert_eq!(shape.dims[0], Dimension::Wildcard);
        assert_eq!(shape.dims[1], Dimension::Concrete(64));
        assert_eq!(shape.dims[2], Dimension::Concrete(64));
    }

    #[test]
    fn test_add_with_broadcast() {
        setup_shape_tracker(vec![
            ("A", vec![32, 128, 256]),
            ("B", vec![128, 256])
        ]);

        // B is broadcasted to match A's shape
        let expr = "(+ (load (input A) (index (tile b) (tile i) (tile j)))
                       (bcast (load (input B) (index (tile i) (tile j))) 0))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 3);
        assert_eq!(shape.dims[0], Dimension::Concrete(32)); // Resolved from A
        assert_eq!(shape.dims[1], Dimension::Concrete(64));
        assert_eq!(shape.dims[2], Dimension::Concrete(64));
    }

    #[test]
    fn test_nested_index() {
        setup_shape_tracker(vec![("A", vec![128, 256, 512])]);

        // Nested index: (index (tile i) (index (tile j) (tile k)))
        let expr = "(load (input A) (index (tile i) (index (tile j) (tile k))))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 3);
        assert_eq!(shape.dims[0], Dimension::Concrete(64)); // tile i -> dim 0
        assert_eq!(shape.dims[1], Dimension::Concrete(64)); // tile j -> dim 1
        assert_eq!(shape.dims[2], Dimension::Concrete(64)); // tile k -> dim 2
    }

    #[test]
    fn test_reduce_sum() {
        setup_shape_tracker(vec![("A", vec![128, 256, 512])]);

        let expr = "(rsum (load (input A) (index (tile i) (tile j) (tile k))) 1)".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 2); // One dimension removed
        assert_eq!(shape.dims[0], Dimension::Concrete(64)); // dim 0
        assert_eq!(shape.dims[1], Dimension::Concrete(64)); // dim 2 (dim 1 was removed)
    }

    #[test]
    fn test_concat() {
        setup_shape_tracker(vec![
            ("A", vec![128, 256]),
            ("B", vec![128, 256])
        ]);

        let expr = "(concat (load (input A) (index (tile i) (tile j)))
                            (load (input B) (index (tile i) (tile j))) 
                            1)".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 2);
        assert_eq!(shape.dims[0], Dimension::Concrete(64)); // Same
        assert_eq!(shape.dims[1], Dimension::Concrete(128)); // 64 + 64
    }

    #[test]
    fn test_permute() {
        setup_shape_tracker(vec![("A", vec![128, 256, 512])]);

        // Permute(A, 0, 2, 1) -> shape becomes [128, 512, 256]
        let expr = "(permute3 (load (input A) (index (tile i) (tile j) (tile k))) 0 2 1)".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        
        let shape = get_tensor_shape(&egraph, id);
        assert!(shape.is_some());
        let shape = shape.unwrap();
        assert_eq!(shape.dims.len(), 3);
        assert_eq!(shape.dims[0], Dimension::Concrete(64)); // Original dim 0
        assert_eq!(shape.dims[1], Dimension::Concrete(64)); // Original dim 2  
        assert_eq!(shape.dims[2], Dimension::Concrete(64)); // Original dim 1
    }
}