#################################################################################

# Acquisition

#################################################################################


from acquisition.acquire import get_connection, new_data, get_data

from web_scraping.acquire import codeup_blogs, all_codeup_blogs, inshorts_articles, get_inshorts_articles



#################################################################################

# Evaluation

#################################################################################


from evaluation.evaluate_error import residuals, sse, mse, rmse, ess, tss, regression_errors, baseline_mean_errors, better_than_baseline, model_significance, plot_residuals



#################################################################################

# Preparation

#################################################################################


from preparation.prepare import split_dataframe, remove_outliers, X_train_validate_test, get_object_cols, get_numeric_X_cols, min_max_scale

from preparation.feature_engineering import select_kbest, rfe

from web_scraping.prepare import clean_and_toke, clean_lem_stop, clean_stem_stop



#################################################################################

# Explore functions

#################################################################################


from exploration.explore import missing_zero_values_table, missing_columns, handle_missing_values, explore_univariate, explore_bivariate, explore_multivariate

from exploration.stats import run_stats_on_everything, t_test, chi2

from exploration.clustering import create_cluster, cluster_scatter_plot



#################################################################################

# Modeling 

#################################################################################


from modeling.anomaly import generate_column_counts_df, generate_column_probability_df, generate_counts_and_probability_df, generate_conditional_probability_df, visualize_target_counts, bollinger_bands, plt_bands

from modeling.models import generate_xy_splits, get_metrics_bin, generate_baseline_model, generate_regression_model, apply_model_to_test_data