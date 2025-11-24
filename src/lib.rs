//! Loop optimization using e-graphs

pub mod language;
pub mod analysis;
pub mod rules;
pub mod applier;
pub mod dependency;
pub mod optimization;
pub mod postprocess;
pub mod utils;
pub mod visualizer;
pub mod extract;
pub mod extract_with_cost;
pub mod cost;
pub mod shape;
pub mod egraph_io;

// Re-export commonly used items
pub use language::{TileLang, LoopAnalysis, LoopData, Access};
pub use rules::{rules, custom_rules, default_tiling, only_seqcomm_rules, rules_wo_seqcomm};
pub use optimization::{run_until_saturated};
pub use visualizer::{save_egraph, save_egraph_from_egraph};
pub use utils::{measure_enode_proportions};
pub use postprocess::{postprocess_egraph, postprocess, postprocess_v2};
pub use extract::{count_expressions_all_for_root, list_expressions_all, count_expressions_num_kernel_for_root, list_expressions_num_kernel};
pub use extract_with_cost::{list_expressions_with_target_cost, list_expressions_with_target_cost_v2, list_expressions_with_target_cost_v3, list_expressions_with_target_cost_v3_part1, list_expressions_with_target_cost_v3_part2};
pub use extract_with_cost::{list_expressions_from_semi_all, list_expressions_from_semi_naive, list_expressions_from_semi_with_cost};
pub use cost::{create_extractor};
pub use egraph_io::{save_raw_egraph, load_raw_egraph};
// #[cfg(feature = "serde-1")]
// pub use egraph_io::{save_egraph_binary, load_egraph_binary};

// Re-export egg items for tests
pub use egg::{test_fn2, test_fn_not2, Rewrite, RecExpr, Runner, Extractor, AstSize};

// Type aliases for convenience
pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;
pub type LoopRunner = egg::Runner<TileLang, LoopAnalysis>;