"""
Prompt Management Layer
-----------------------
Loads versioned prompts from YAML files in the prompts/ folder.
Keeps a registry of which version was used for every pipeline run,
satisfying the GenAI Ops "prompt versioning" requirement.

Usage:
    pm = PromptManager()
    system_msg, user_msg = pm.render(stage=1, variables={"document_chunks": "..."})
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Map stage numbers to prompt file stems
_STAGE_MAP = {
    1: "stage_1_ground",
    2: "stage_2_extract",
    4: "stage_4_analyse",
    5: "stage_5_verify",
    6: "stage_6_generate",
    7: "stage_7_review",
    # Stage 3 is Python-only; no prompt needed
}

_DEFAULT_VERSION = "v1"
_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptManager:
    """
    Loads prompts by stage number and version.
    Raises a clear error if the prompt file is missing — never falls back
    to a hardcoded string, so version drift is caught immediately.
    """

    def __init__(self, prompts_dir: Path | str | None = None, version: str = _DEFAULT_VERSION):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
        self.version = version
        self._cache: dict[int, dict] = {}
        self._validate_all_prompts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_meta(self, stage: int) -> dict:
        """Return the full YAML dict for a stage (model, max_tokens, etc.)."""
        return self._load(stage)

    def render(self, stage: int, variables: dict[str, Any]) -> tuple[str, str]:
        """
        Return (system_message, user_message) with {variables} substituted.
        Raises KeyError if a required variable is missing.
        """
        prompt = self._load(stage)
        system_msg = prompt["system"].strip()
        user_template = prompt["user_template"].strip()

        try:
            user_msg = user_template.format(**variables)
        except KeyError as exc:
            raise KeyError(
                f"Prompt for stage {stage} requires variable {exc} "
                f"but it was not provided. Check your pipeline call."
            ) from exc

        logger.debug(
            "Prompt rendered",
            extra={
                "stage": stage,
                "prompt_id": prompt["prompt_id"],
                "prompt_version": prompt["version"],
            },
        )
        return system_msg, user_msg

    def prompt_version(self, stage: int) -> str:
        """Return the version string for audit logging."""
        return self._load(stage)["version"]

    def all_versions(self) -> dict[int, str]:
        """Return {stage: version} for the entire pipeline. Used in audit logs."""
        return {stage: self.prompt_version(stage) for stage in _STAGE_MAP}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, stage: int) -> dict:
        if stage in self._cache:
            return self._cache[stage]

        stem = _STAGE_MAP.get(stage)
        if stem is None:
            raise ValueError(
                f"Stage {stage} has no prompt (stage 3 is Python-only). "
                f"Valid stages: {sorted(_STAGE_MAP.keys())}"
            )

        path = self.prompts_dir / f"{stem}_{self.version}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
                f"Add the file or change version (current: '{self.version}')."
            )

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        self._cache[stage] = data
        return data

    def _validate_all_prompts(self) -> None:
        """Fail fast at startup if any prompt file is missing."""
        missing = []
        for stage, stem in _STAGE_MAP.items():
            path = self.prompts_dir / f"{stem}_{self.version}.yaml"
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                f"PromptManager: {len(missing)} prompt file(s) missing:\n"
                + "\n".join(f"  {p}" for p in missing)
            )
        logger.info(
            "PromptManager: all %d prompt files validated (version=%s)",
            len(_STAGE_MAP),
            self.version,
        )
