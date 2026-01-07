use std::env;
use std::fs;
use std::path::Path;

use trinity::language::{LoopAnalysis, TileLang, SHAPE_TRACKER};
use trinity::shape::ShapeTracker;
use trinity::*;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

fn setup_shape_tracker(shapes: Vec<(String, Vec<usize>)>) {
    SHAPE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        *tracker = ShapeTracker::new();
        for (name, dims) in shapes {
            tracker.add_tensor(&name, dims);
        }
    });
}

fn load_shapes(path: &str) -> Vec<(String, Vec<usize>)> {
    let text = fs::read_to_string(path).expect("Failed to read shapes file");
    let json: serde_json::Value = serde_json::from_str(&text).expect("Invalid JSON");
    let obj = json.as_object().expect("Shapes JSON must be an object");
    let mut out = Vec::new();
    for (name, dims_val) in obj {
        let Some(arr) = dims_val.as_array() else { continue };
        let mut dims = Vec::new();
        let mut ok = true;
        for dim in arr {
            if let Some(n) = dim.as_u64() {
                dims.push(n as usize);
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            out.push((name.clone(), dims));
        }
    }
    out
}

#[test]
fn optimize_exported_mainfunc() {
    let export_root = env::var("TRINITY_EXPORT_ROOT")
        .unwrap_or_else(|_| "/home/um3maru/TrinityFE_tvm/outputs".to_string());
    let base = env::var("TRINITY_EXPORT_BASENAME").unwrap_or_else(|_| "Roco".to_string());

    let ir_path = format!("{}/trinity_seq/{}_main_seq.txt", export_root, base);
    let shape_path = format!("{}/trinity/{}_main_shapes.json", export_root, base);

    if !Path::new(&ir_path).exists() || !Path::new(&shape_path).exists() {
        eprintln!("Missing export files: {} or {}", ir_path, shape_path);
        return;
    }

    let shapes = load_shapes(&shape_path);
    setup_shape_tracker(shapes);

    let expr_str = fs::read_to_string(&ir_path).expect("Failed to read IR file");
    let _runner = run_until_saturated(expr_str.as_str(), rules(), 5);
}
