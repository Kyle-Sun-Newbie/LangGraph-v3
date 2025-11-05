(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 1
{
  "scheme": "fewshot_faiss",
  "num_samples": 4942,
  "num_valid": 4005,
  "parse_error_rate": 0.1896,
  "nonempty_result_rate": 0.3832,
  "metrics": {
    "set": {
      "n": 4005,
      "strict_macro_f1": 0.1502,
      "exact_set_match_rate": 0.1501,
      "local_macro_f1": 0.1503,
      "simple_macro_f1": 0.1529,
      "best_macro_f1": 0.1529,
      "size_match_rate": 0.4504,
      "overlap_rate": 0.1513,
      "uri_rows": 2926,
      "uri_overlap_rate": 0.2057,
      "uri_exact_match_rate": 0.2051
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.1501,
  "overall_accuracy_relaxed": 0.1528
}

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 1 --use_llm --limit_test 100 --random_test --seed 42
{
  "scheme": "fewshot_faiss",
  "num_samples": 100,
  "num_valid": 87,
  "parse_error_rate": 0.13,
  "nonempty_result_rate": 0.77,
  "metrics": {
    "set": {
      "n": 87,
      "strict_macro_f1": 0.8391,
      "exact_set_match_rate": 0.8391,
      "local_macro_f1": 0.8506,
      "simple_macro_f1": 0.8736,
      "best_macro_f1": 0.8736,
      "size_match_rate": 0.8736,
      "overlap_rate": 0.8506,
      "uri_rows": 67,
      "uri_overlap_rate": 0.8955,
      "uri_exact_match_rate": 0.8955
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.8391,
  "overall_accuracy_relaxed": 0.8736
}

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 2
{
  "scheme": "fewshot_hnsw",
  "num_samples": 4942,
  "num_valid": 4005,
  "parse_error_rate": 0.1896,
  "nonempty_result_rate": 0.3893,
  "metrics": {
    "set": {
      "n": 4005,
      "strict_macro_f1": 0.1519,
      "exact_set_match_rate": 0.1518,
      "local_macro_f1": 0.152,
      "simple_macro_f1": 0.1546,
      "best_macro_f1": 0.1546,
      "size_match_rate": 0.4577,
      "overlap_rate": 0.1531,
      "uri_rows": 2925,
      "uri_overlap_rate": 0.2082,
      "uri_exact_match_rate": 0.2075
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.1518,
  "overall_accuracy_relaxed": 0.1546
}
Done in 526.0s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 2 --use_llm --limit_test 100 --random_test --seed 42
{
  "scheme": "fewshot_hnsw",
  "num_samples": 100,
  "num_valid": 87,
  "parse_error_rate": 0.13,
  "nonempty_result_rate": 0.78,
  "metrics": {
    "set": {
      "n": 87,
      "strict_macro_f1": 0.8506,
      "exact_set_match_rate": 0.8506,
      "local_macro_f1": 0.8621,
      "simple_macro_f1": 0.8851,
      "best_macro_f1": 0.8851,
      "size_match_rate": 0.8851,
      "overlap_rate": 0.8621,
      "uri_rows": 67,
      "uri_overlap_rate": 0.9104,
      "uri_exact_match_rate": 0.9104
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.8506,
  "overall_accuracy_relaxed": 0.8851
}
Done in 429.4s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments   

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 3
{
  "scheme": "langgraph_like_template",
  "num_samples": 4942,
  "num_valid": 4942,
  "parse_error_rate": 0.0,
  "nonempty_result_rate": 0.0002,
  "metrics": {
    "set": {
      "n": 4942,
      "strict_macro_f1": 0.0,
      "exact_set_match_rate": 0.0,
      "local_macro_f1": 0.0,
      "simple_macro_f1": 0.0,
      "best_macro_f1": 0.0,
      "size_match_rate": 0.0002,
      "overlap_rate": 0.0,
      "uri_rows": 3549,
      "uri_overlap_rate": 0.0,
      "uri_exact_match_rate": 0.0
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.0,
  "overall_accuracy_relaxed": 0.0
}
Done in 6.7s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 4 --limit_test 100 --random_test --seed 42
{
  "scheme": "langgraph_like_llm",
  "num_samples": 100,
  "num_valid": 100,
  "parse_error_rate": 0.0,
  "nonempty_result_rate": 0.01,
  "metrics": {
    "set": {
      "n": 100,
      "strict_macro_f1": 0.01,
      "exact_set_match_rate": 0.01,
      "local_macro_f1": 0.01,
      "simple_macro_f1": 0.01,
      "best_macro_f1": 0.01,
      "size_match_rate": 0.01,
      "overlap_rate": 0.01,
      "uri_rows": 75,
      "uri_overlap_rate": 0.0133,
      "uri_exact_match_rate": 0.0133
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.01,
  "overall_accuracy_relaxed": 0.01
}
Done in 684.6s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 5 --limit_test 50 --random_test --seed 42
{
  "scheme": "langgraph_like_ragllm",
  "num_samples": 50,
  "num_valid": 50,
  "parse_error_rate": 0.0,
  "nonempty_result_rate": 0.42,
  "metrics": {
    "set": {
      "n": 50,
      "strict_macro_f1": 0.34,
      "exact_set_match_rate": 0.34,
      "local_macro_f1": 0.34,
      "simple_macro_f1": 0.34,
      "best_macro_f1": 0.34,
      "size_match_rate": 0.42,
      "overlap_rate": 0.34,
      "uri_rows": 36,
      "uri_overlap_rate": 0.4167,
      "uri_exact_match_rate": 0.4167
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.34,
  "overall_accuracy_relaxed": 0.34
}
Done in 1023.9s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 6 --ts_dataset "F:\Task\RAG-LangGraph-Demo-bcp\data\dataset_ma" --ttl "F:\Task\RAG-LangGraph-Demo-bcp\data\dataset_ma\PV_System_data.ttl" --ts_eps 1e-3    
{
  "n": 220,
  "site_accuracy": 1.0,
  "value_mae": 0.0,
  "within_eps_ratio": 1.0,
  "overall_accuracy_strict": 1.0,
  "overall_accuracy_relaxed": 1.0,
  "eps": 0.001
}
Done in 0.6s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments     

(.venv) PS F:\Task\RAG-LangGraph-Demo-bcp> python -m app.experiments.run_experiments --scheme 2 --use_llm --limit_test 100 --random_test --seed 42
[Scheme2-HNSW] 建索引：train=13613, backend=faiss_hnsw, M=32, efC=300, efS=256, k_shot=6
[Scheme2-HNSW] 每文件抽样 K=100；use_llm=True；random_test=True；seed=42
[Scheme2-HNSW] 将按 6 个测试集分别输出报告…
Predicting (Scheme2-HNSW | test_data_1 | n=80): 100%|██████████████| 80/80 [06:27<00:00,  4.85s/it] 

=== JSON Summary (Scheme 2: HNSW | test_data_1) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 80,
  "num_valid": 75,
  "parse_error_rate": 0.0625,
  "nonempty_result_rate": 0.925,
  "metrics": {
    "set": {
      "n": 75,
      "strict_macro_f1": 0.9867,
      "exact_set_match_rate": 0.9867,
      "local_macro_f1": 0.9867,
      "simple_macro_f1": 0.9867,
      "best_macro_f1": 0.9867,
      "size_match_rate": 0.9867,
      "overlap_rate": 0.9867,
      "uri_rows": 75,
      "uri_overlap_rate": 0.9867,
      "uri_exact_match_rate": 0.9867
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.9867,
  "overall_accuracy_relaxed": 0.9867
}
Predicting (Scheme2-HNSW | test_data_2 | n=80): 100%|██████████████| 80/80 [03:57<00:00,  2.96s/it] 

=== JSON Summary (Scheme 2: HNSW | test_data_2) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 80,
  "num_valid": 75,
  "parse_error_rate": 0.0625,
  "nonempty_result_rate": 0.9375,
  "metrics": {
    "set": {
      "n": 0,
      "strict_macro_f1": null,
      "exact_set_match_rate": null,
      "local_macro_f1": null,
      "simple_macro_f1": null,
      "best_macro_f1": null,
      "size_match_rate": null,
      "overlap_rate": null,
      "uri_rows": 0,
      "uri_overlap_rate": null,
      "uri_exact_match_rate": null
    },
    "count": {
      "n": 75,
      "accuracy": 1.0
    }
  },
  "overall_accuracy_strict": 1.0,
  "overall_accuracy_relaxed": 1.0
}
Predicting (Scheme2-HNSW | test_data_3 | n=100): 100%|███████████| 100/100 [05:47<00:00,  3.48s/it]

=== JSON Summary (Scheme 2: HNSW | test_data_3) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 100,
  "num_valid": 79,
  "parse_error_rate": 0.21,
  "nonempty_result_rate": 0.76,
  "metrics": {
    "set": {
      "n": 79,
      "strict_macro_f1": 0.962,
      "exact_set_match_rate": 0.962,
      "local_macro_f1": 0.962,
      "simple_macro_f1": 0.962,
      "best_macro_f1": 0.962,
      "size_match_rate": 0.962,
      "overlap_rate": 0.962,
      "uri_rows": 79,
      "uri_overlap_rate": 0.962,
      "uri_exact_match_rate": 0.962
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.962,
  "overall_accuracy_relaxed": 0.962
}
Predicting (Scheme2-HNSW | test_data_4 | n=100): 100%|███████████| 100/100 [08:40<00:00,  5.21s/it] 

=== JSON Summary (Scheme 2: HNSW | test_data_4) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 100,
  "num_valid": 86,
  "parse_error_rate": 0.14,
  "nonempty_result_rate": 0.86,
  "metrics": {
    "set": {
      "n": 86,
      "strict_macro_f1": 0.9651,
      "exact_set_match_rate": 0.9651,
      "local_macro_f1": 0.9767,
      "simple_macro_f1": 1.0,
      "best_macro_f1": 1.0,
      "size_match_rate": 1.0,
      "overlap_rate": 0.9767,
      "uri_rows": 46,
      "uri_overlap_rate": 0.9348,
      "uri_exact_match_rate": 0.9348
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.9651,
  "overall_accuracy_relaxed": 1.0
}
Predicting (Scheme2-HNSW | test_data_5 | n=100): 100%|███████████| 100/100 [06:37<00:00,  3.98s/it] 

=== JSON Summary (Scheme 2: HNSW | test_data_5) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 100,
  "num_valid": 78,
  "parse_error_rate": 0.22,
  "nonempty_result_rate": 0.76,
  "metrics": {
    "set": {
      "n": 78,
      "strict_macro_f1": 0.9744,
      "exact_set_match_rate": 0.9744,
      "local_macro_f1": 0.9744,
      "simple_macro_f1": 0.9744,
      "best_macro_f1": 0.9744,
      "size_match_rate": 0.9744,
      "overlap_rate": 0.9744,
      "uri_rows": 78,
      "uri_overlap_rate": 0.9744,
      "uri_exact_match_rate": 0.9744
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.9744,
  "overall_accuracy_relaxed": 0.9744
}
Predicting (Scheme2-HNSW | test_data_6 | n=100): 100%|███████████| 100/100 [08:13<00:00,  4.94s/it]

=== JSON Summary (Scheme 2: HNSW | test_data_6) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 100,
  "num_valid": 89,
  "parse_error_rate": 0.11,
  "nonempty_result_rate": 0.5,
  "metrics": {
    "set": {
      "n": 89,
      "strict_macro_f1": 0.4494,
      "exact_set_match_rate": 0.4494,
      "local_macro_f1": 0.4494,
      "simple_macro_f1": 0.4719,
      "best_macro_f1": 0.4719,
      "size_match_rate": 0.4831,
      "overlap_rate": 0.4494,
      "uri_rows": 35,
      "uri_overlap_rate": 0.6857,
      "uri_exact_match_rate": 0.6857
    },
    "count": {
      "n": 0,
      "accuracy": null
    }
  },
  "overall_accuracy_strict": 0.4494,
  "overall_accuracy_relaxed": 0.4719
}

=== JSON Summary (Scheme 2: HNSW | COMBINED over per-file K=100) ===
{
  "scheme": "fewshot_hnsw",
  "num_samples": 560,
  "num_valid": 482,
  "parse_error_rate": 0.1393,
  "nonempty_result_rate": 0.7804,
  "metrics": {
    "set": {
      "n": 407,
      "strict_macro_f1": 0.8575,
      "exact_set_match_rate": 0.8575,
      "local_macro_f1": 0.86,
      "simple_macro_f1": 0.8698,
      "best_macro_f1": 0.8698,
      "size_match_rate": 0.8722,
      "overlap_rate": 0.86,
      "uri_rows": 313,
      "uri_overlap_rate": 0.9361,
      "uri_exact_match_rate": 0.9361
    },
    "count": {
      "n": 75,
      "accuracy": 1.0
    }
  },
  "overall_accuracy_strict": 0.8797,
  "overall_accuracy_relaxed": 0.89
}
Done in 2395.7s, results at: F:\Task\RAG-LangGraph-Demo-bcp\outputs\experiments
