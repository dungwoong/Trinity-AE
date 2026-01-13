use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{BufWriter, Write};

use rayon::prelude::*;
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

fn load_shapes(path: &Path) -> Vec<(String, Vec<usize>)> {
    let text = fs::read_to_string(path).expect("Failed to read shapes file");
    let json: serde_json::Value = serde_json::from_str(&text).expect("Invalid JSON");
    let obj = json.as_object().expect("Shapes JSON must be an object");
    let mut out = Vec::new();
    for (name, dims_val) in obj {
        let arr = if let Some(arr) = dims_val.as_array() {
            arr
        } else if let Some(obj) = dims_val.as_object() {
            match obj.get("shape").and_then(|shape| shape.as_array()) {
                Some(arr) => arr,
                None => continue,
            }
        } else {
            continue;
        };
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
fn optimize_custom_model() {
    let base = env::var("TRINITY_MODEL_NAME").unwrap_or_else(|_| "Roco".to_string());
    let optimizer_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let export_root = optimizer_root
        .join("..")
        .join("frontend")
        .join("outputs")
        .join("trinity");

    let ir_path = export_root.join(&base).join("ir.txt");
    let shape_path = export_root.join(&base).join("shapes.json");

    if !ir_path.exists() || !shape_path.exists() {
        eprintln!(
            "Missing export files: {} or {}",
            ir_path.display(),
            shape_path.display()
        );
        return;
    }

    let shapes = load_shapes(&shape_path);
    setup_shape_tracker(shapes);

    let expr_str = fs::read_to_string(&ir_path).expect("Failed to read IR file");
    let runner = run_until_saturated(expr_str.as_str(), rules(), 8);
    let all_possibilities = count_expressions_all_for_root(&runner);
    println!("{:?}", all_possibilities);
    let expressions_root = optimizer_root.join("expressions");
    let max_cost = 6;
    let max_num_kernel = 2;
    let semi_path = expressions_root
        .join("semi")
        .join(format!("{}_cost{}_kern{}.json", base, max_cost, max_num_kernel));
    let semi_path_str = semi_path
        .to_str()
        .expect("Semi path must be valid UTF-8");

    match list_expressions_with_target_cost_v3_part1(
        &runner,
        semi_path_str,
        max_cost,
        max_num_kernel,
    ) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }

    let (expressions, tile_sets) =
        match list_expressions_from_semi_with_cost(&runner, semi_path_str, usize::MAX) {
            Ok((expressions, tile_sets)) => {
                println!("Loaded {} final expressions", expressions.len());
                println!("{:?}", tile_sets);
                (expressions, tile_sets)
            }
            Err(e) => {
                println!("Load error: {}", e);
                return;
            }
        };

    let output_path = expressions_root
        .join(format!("{}_cost{}_kern{}.txt", base, max_cost, max_num_kernel));
    let file = File::create(&output_path).expect("Failed to create file");
    let mut writer = BufWriter::new(file);

    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{}: {}", i, new_expr)
        })
        .filter(|line| !line.contains("dummydata"))
        .collect::<Vec<String>>()
        .iter()
        .for_each(|line| {
            writeln!(writer, "{}", line).expect("Failed to write to file");
        });

    writer.flush().expect("Failed to flush writer");
}

fn count_all() {
    let base = env::var("TRINITY_MODEL_NAME").unwrap_or_else(|_| "Roco".to_string());
    let optimizer_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let export_root = optimizer_root
        .join("..")
        .join("frontend")
        .join("outputs")
        .join("trinity");

    let ir_path = export_root.join(&base).join("ir.txt");
    let shape_path = export_root.join(&base).join("shapes.json");

    if !ir_path.exists() || !shape_path.exists() {
        eprintln!(
            "Missing export files: {} or {}",
            ir_path.display(),
            shape_path.display()
        );
        return;
    }

    let shapes = load_shapes(&shape_path);
    setup_shape_tracker(shapes);

    let expr_str = fs::read_to_string(&ir_path).expect("Failed to read IR file");
    let runner = run_until_saturated(expr_str.as_str(), rules(), 8);
    let all_possibilities = count_expressions_all_for_root(&runner);
    println!("{:?}", all_possibilities);
}
