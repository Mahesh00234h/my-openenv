import pytest
from pydantic import ValidationError

from email_triage_env.models import Action, Reward


def test_action_rejects_missing_required_fields():
    """Action construction with missing required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        Action(email_id="e001")  # missing category, priority, suggested_response


def test_reward_has_score_field():
    """A valid Reward object has a .score attribute."""
    reward = Reward(
        score=0.75,
        category_score=1.0,
        priority_score=0.5,
        response_score=0.5,
        explanation="test",
    )
    assert hasattr(reward, "score")
    assert reward.score == 0.75


import json
from pathlib import Path


def test_easy_triage_fixture_has_5_emails():
    """easy_triage.json contains exactly 5 emails."""
    fixture = json.loads(Path("tasks/easy_triage.json").read_text())
    assert len(fixture["emails"]) == 5


def test_medium_triage_fixture_has_10_emails():
    """medium_triage.json contains exactly 10 emails."""
    fixture = json.loads(Path("tasks/medium_triage.json").read_text())
    assert len(fixture["emails"]) == 10


def test_hard_triage_fixture_has_15_emails():
    """hard_triage.json contains exactly 15 emails."""
    fixture = json.loads(Path("tasks/hard_triage.json").read_text())
    assert len(fixture["emails"]) == 15


from email_triage_env.env import EmailTriageEnv


def test_step_before_reset_raises_runtime_error():
    """Calling step() on a fresh EmailTriageEnv before reset() raises RuntimeError."""
    env = EmailTriageEnv()
    action = Action(
        email_id="e001",
        category="spam",
        priority="low",
        suggested_response="",
    )
    with pytest.raises(RuntimeError, match="Environment not initialized. Call reset\\(\\) first."):
        env.step(action)


def test_reset_with_unknown_task_id_raises_value_error():
    """Calling reset() with an unknown task_id raises ValueError with 'Unknown task' in message."""
    env = EmailTriageEnv()
    with pytest.raises(ValueError, match="Unknown task"):
        env.reset("nonexistent_task")


# --- Task 7.2: openenv.yaml metadata tests ---

import yaml


def _load_openenv_yaml():
    return yaml.safe_load(Path("openenv.yaml").read_text())


def test_openenv_yaml_required_fields_present():
    """openenv.yaml contains all required fields."""
    data = _load_openenv_yaml()
    required_fields = ["name", "version", "description", "observation_space",
                       "action_space", "reward_range", "tasks", "tags"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_openenv_yaml_tags_includes_openenv():
    """openenv.yaml tags includes 'openenv'."""
    data = _load_openenv_yaml()
    assert "openenv" in data["tags"]


def test_openenv_yaml_reward_range_is_0_to_1():
    """openenv.yaml reward_range is [0.0, 1.0]."""
    data = _load_openenv_yaml()
    assert data["reward_range"] == [0.0, 1.0]


def test_openenv_yaml_tasks_lists_3_identifiers():
    """openenv.yaml tasks lists exactly 3 identifiers."""
    data = _load_openenv_yaml()
    assert len(data["tasks"]) == 3


# --- Task 8.4: inference.py unit tests ---

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_inference_py_exists_at_root():
    """inference.py exists at the workspace root."""
    assert Path("inference.py").exists(), "inference.py not found at repository root"


def test_call_llm_with_retry_returns_dummy_action_on_all_failures():
    """When the LLM API always fails, call_llm_with_retry returns a dummy Action with score=0.0 fields."""
    from inference import call_llm_with_retry

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    with patch("time.sleep"):  # avoid real delays
        result = call_llm_with_retry(mock_client, "test-model", "test prompt", "e001")

    # Should have made MAX_RETRIES + 1 = 4 total attempts
    assert mock_client.chat.completions.create.call_count == 4
    # Returns a dummy Action
    assert result.email_id == "e001"
    assert result.category == "spam"
    assert result.priority == "low"
    assert result.suggested_response == ""


def test_call_llm_with_retry_succeeds_after_failures():
    """call_llm_with_retry retries and succeeds on the 4th attempt."""
    import json
    from inference import call_llm_with_retry

    success_response = MagicMock()
    success_response.choices[0].message.content = json.dumps({
        "email_id": "e001",
        "category": "urgent",
        "priority": "urgent",
        "suggested_response": "On it",
    })

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        Exception("fail 1"),
        Exception("fail 2"),
        Exception("fail 3"),
        success_response,
    ]

    with patch("time.sleep"):
        result = call_llm_with_retry(mock_client, "test-model", "test prompt", "e001")

    assert mock_client.chat.completions.create.call_count == 4
    assert result.category == "urgent"
    assert result.priority == "urgent"


def test_get_env_vars_exits_when_api_key_missing():
    """get_env_vars() calls sys.exit(1) when OPENAI_API_KEY is missing."""
    from inference import get_env_vars

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            get_env_vars()
    assert exc_info.value.code == 1


def test_get_env_vars_returns_values_when_api_key_present():
    """get_env_vars() returns values when OPENAI_API_KEY is set (API_BASE_URL and MODEL_NAME have defaults)."""
    from inference import get_env_vars

    env = {"OPENAI_API_KEY": "sk-test"}
    with patch.dict(os.environ, env, clear=True):
        api_key, base_url, model_name = get_env_vars()

    assert api_key == "sk-test"
    assert base_url  # has a default
    assert model_name  # has a default


# --- Task 9.3: Dockerfile unit tests ---


def test_dockerfile_exists():
    """Dockerfile exists at the workspace root."""
    assert Path("Dockerfile").exists(), "Dockerfile not found at repository root"


def test_dockerfile_contains_from_python_311():
    """Dockerfile contains 'FROM python:3.11'."""
    content = Path("Dockerfile").read_text()
    assert "FROM python:3.11" in content


def test_dockerfile_contains_copy_requirements():
    """Dockerfile contains 'COPY requirements.txt'."""
    content = Path("Dockerfile").read_text()
    assert "COPY requirements.txt" in content


def test_dockerfile_contains_env_port_7860():
    """Dockerfile contains 'ENV PORT=7860'."""
    content = Path("Dockerfile").read_text()
    assert "ENV PORT=7860" in content


# --- Task 10.2: README.md YAML front matter and required sections tests ---


def _parse_readme_front_matter():
    """Extract and parse the YAML front matter from README.md."""
    content = Path("README.md").read_text()
    # Front matter is between the first and second '---'
    parts = content.split("---", 2)
    # parts[0] is empty string before first ---, parts[1] is the YAML block
    return yaml.safe_load(parts[1])


def test_readme_front_matter_sdk_is_docker():
    """README.md YAML front matter has sdk: docker."""
    front_matter = _parse_readme_front_matter()
    assert front_matter["sdk"] == "docker"


def test_readme_front_matter_app_port_is_7860():
    """README.md YAML front matter has app_port: 7860."""
    front_matter = _parse_readme_front_matter()
    assert front_matter["app_port"] == 7860


def test_readme_front_matter_tags_includes_openenv():
    """README.md YAML front matter tags includes 'openenv'."""
    front_matter = _parse_readme_front_matter()
    assert "openenv" in front_matter["tags"]


def test_readme_contains_observation_space_section():
    """README.md contains an 'Observation Space' section heading."""
    content = Path("README.md").read_text().lower()
    assert "observation space" in content


def test_readme_contains_action_space_section():
    """README.md contains an 'Action Space' section heading."""
    content = Path("README.md").read_text().lower()
    assert "action space" in content


def test_readme_contains_task_section():
    """README.md contains a 'Task' descriptions section heading."""
    content = Path("README.md").read_text().lower()
    assert "task" in content


def test_readme_contains_setup_section():
    """README.md contains a 'Setup' instructions section heading."""
    content = Path("README.md").read_text().lower()
    assert "setup" in content


def test_readme_contains_baseline_section():
    """README.md contains a 'Baseline' scores section heading."""
    content = Path("README.md").read_text().lower()
    assert "baseline" in content
