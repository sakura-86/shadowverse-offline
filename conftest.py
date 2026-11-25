# -*- coding: utf-8 -*-
"""Pytest conftest: 自动将 CLI 集成测试标记为 slow。

快速验证（默认，~5s）: pytest
全量测试（提交前）:     pytest -m ""
只跑慢测试:            pytest -m slow
"""
import pytest

# 这些测试文件是 CLI 集成测试或重模板加载测试（subprocess + 重复加载模板），占总耗时 90%+
_SLOW_TEST_FILES = {
    # CLI 集成测试 (~75s)
    "test_replay_audit.py",
    "test_export_route_samples_cli.py",
    "test_consume_puzzle_replays_cli.py",
    "test_validate_replay_cli.py",
    "test_svhook_raw_recorded_scenario.py",
    "test_export_training_examples_from_route_samples.py",
    "test_run_self_play_generation.py",
    "test_run_candidate_policy_value_training.py",
    "test_run_policy_value_training_loop.py",
    "test_run_policy_value_training.py",
    "test_run_manifest.py",
    "test_train_policy_value_backend.py",
    "test_split_policy_training_data.py",
    "test_split_policy_value_training_data.py",
    "test_export_policy_value_tensors.py",
    "test_export_policy_training_data_from_training_examples.py",
    "test_export_policy_value_training_data.py",
    "test_decide_policy_value_promotion.py",
    "test_evaluate_candidate_policy_value_run.py",
    "test_analyze_training_examples.py",
    "test_analyze_policy_training_data.py",
    "test_build_policy_value_promotion_bundle.py",
    "test_build_policy_value_runtime_deployment_bundle.py",
    "test_split_route_samples_by_trust.py",
    "test_analyze_local_self_play_sampling_cli.py",
    "test_policy_value_inference.py",
    # 重模板加载 (~15s)
    "test_effect_binding.py",
    "test_card_templates.py",
    "test_action_space_diff.py",
    "test_effect_specs.py",
    "test_self_play_search.py",
}


def pytest_collection_modifyitems(config, items):
    """自动给 CLI 集成测试加 slow marker。"""
    slow_marker = pytest.mark.slow
    for item in items:
        filename = item.fspath.basename
        if filename in _SLOW_TEST_FILES:
            item.add_marker(slow_marker)
