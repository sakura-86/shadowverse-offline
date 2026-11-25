# -*- coding: utf-8 -*-
"""Training data schema and feature encoders."""

from decision_engine.training.action_features import action_features, legal_actions_features
from decision_engine.training.example_builder import build_training_example
from decision_engine.training.jsonl import load_training_examples_jsonl, write_training_examples_jsonl
from decision_engine.training.mulligan_example import MULLIGAN_EXAMPLE_SCHEMA, MULLIGAN_EXAMPLE_VERSION, MulliganTrainingExample, build_mulligan_training_example, validate_mulligan_example
from decision_engine.training.mulligan_features import MULLIGAN_FEATURES_SCHEMA, MULLIGAN_FEATURES_VERSION, mulligan_hand_card_features, mulligan_state_features
from decision_engine.training.state_features import state_features
from decision_engine.training.mlp_policy_value import MlpPolicyValueConfig, MlpPolicyValueNet, MlpPolicyValueTrainingResult, train_mlp_policy_value_model
from decision_engine.training.model_protocol import MODEL_ARTIFACT_SCHEMA, MODEL_ARTIFACT_VERSION, PolicyValueModel, PolicyValueModelArtifact, PolicyValuePrediction, PolicyValueTrainer
from decision_engine.training.run_manifest import RUN_MANIFEST_SCHEMA, RUN_MANIFEST_VERSION, PolicyValueTrainingRunManifest
from decision_engine.training.self_play_manifest import SELF_PLAY_MANIFEST_SCHEMA, SELF_PLAY_MANIFEST_VERSION, SelfPlayTrainingRunManifest
from decision_engine.training.tensor_dataset import TENSOR_DATASET_SCHEMA, TENSOR_DATASET_VERSION, flatten_tensor_dataset, load_tensor_dataset_dir, load_tensor_dataset_jsonl, summarize_tensor_dataset, validate_tensorized_example
from decision_engine.training.tensorization import ACTION_VECTOR_FEATURE_ORDER, STATE_VECTOR_FEATURE_ORDER, TensorizationResult, action_vectors_from_action_features, state_vector_from_state_features, tensorize_training_example, tensorize_training_examples
from decision_engine.training.training_example import TrainingExample, validate_training_example

__all__ = [
    "TrainingExample",
    "MODEL_ARTIFACT_SCHEMA",
    "MODEL_ARTIFACT_VERSION",
    "PolicyValueTrainingRunManifest",
    "RUN_MANIFEST_SCHEMA",
    "RUN_MANIFEST_VERSION",
    "SELF_PLAY_MANIFEST_SCHEMA",
    "SELF_PLAY_MANIFEST_VERSION",
    "SelfPlayTrainingRunManifest",
    "MlpPolicyValueConfig",
    "MlpPolicyValueNet",
    "MlpPolicyValueTrainingResult",
    "PolicyValueModel",
    "PolicyValueModelArtifact",
    "PolicyValuePrediction",
    "PolicyValueTrainer",
    "TENSOR_DATASET_SCHEMA",
    "TENSOR_DATASET_VERSION",
    "TensorizationResult",
    "build_training_example",
    "build_mulligan_training_example",
    "action_features",
    "legal_actions_features",
    "flatten_tensor_dataset",
    "load_tensor_dataset_dir",
    "load_tensor_dataset_jsonl",
    "load_training_examples_jsonl",
    "summarize_tensor_dataset",
    "validate_tensorized_example",
    "state_features",
    "tensorize_training_example",
    "tensorize_training_examples",
    "train_mlp_policy_value_model",
    "state_vector_from_state_features",
    "action_vectors_from_action_features",
    "STATE_VECTOR_FEATURE_ORDER",
    "ACTION_VECTOR_FEATURE_ORDER",
    "validate_training_example",
    "validate_mulligan_example",
    "MulliganTrainingExample",
    "MULLIGAN_EXAMPLE_SCHEMA",
    "MULLIGAN_EXAMPLE_VERSION",
    "MULLIGAN_FEATURES_SCHEMA",
    "MULLIGAN_FEATURES_VERSION",
    "mulligan_hand_card_features",
    "mulligan_state_features",
    "write_training_examples_jsonl",
]
