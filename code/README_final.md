# Final report code (run from the `code/` folder, in this order)
1. `python rq2_its.py`          -> rq2_results.json (RQ2 interrupted time series, placebo breaks)
2. `python final_analysis.py`   -> final_results.json + EDA figures + RQ1/RQ3/RQ4 models
3. `python sample_size.py`      -> sample_size_results.json
4. `python bootstrap.py`        -> bootstrap_results.json (bank-level pairs bootstrap, ~5 min)
5. `python benchmark.py`        -> benchmark_results.json, model figures, outputs/watchlist_next_month.csv (~5 min)
6. `python db_setup.py`         -> ../data/upi.db (SQLite)
7. `python workflow_figure.py`  -> outputs/fig_workflow.png
Needs: pandas numpy scipy statsmodels scikit-learn xgboost matplotlib. Seed 42. CPU only.
