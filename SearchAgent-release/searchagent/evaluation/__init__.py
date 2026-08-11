from searchagent.evaluation.metrics import (
    click_position_jsd,
    session_length_jsd,
    clicks_per_session_gap,
    abandon_rate_gap,
    reformulation_rate_gap,
    composite_fidelity_score,
    population_fidelity_discrepancy,
    replay_alignment_metrics,
)
from searchagent.evaluation.reference_data import REFERENCE, SearchLogReference
from searchagent.evaluation.analysis import (
    generate_main_comparison_table,
    generate_ablation_table,
    generate_descriptive_table,
    generate_trait_table,
    generate_markdown_report,
    save_all_tables,
)
from searchagent.evaluation.statistical_tests import (
    bootstrap_ci,
    paired_permutation_test,
    wilcoxon_test,
    cohens_d,
    significance_marker,
    pairwise_significance_table,
    format_significance_table_latex,
)
from searchagent.evaluation.human_eval import (
    EvalSession,
    JudgeAnnotation,
    HumanEvalResults,
    create_evaluation_set,
    compute_human_eval_results,
    format_results_latex as format_human_eval_latex,
)
from searchagent.evaluation.downstream_eval import (
    ClickSample,
    PositionBasedClickModel,
    run_downstream_evaluation,
    cross_validation_experiment,
    format_downstream_latex,
)
