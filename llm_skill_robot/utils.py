"""Utility functions for resolving project paths."""

import os
from pathlib import Path


def _candidate_project_roots() -> list[Path]:
    """Return likely project roots for source and ROS 2 install layouts."""
    module_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    # Source checkout: <root>/llm_skill_robot/utils.py
    candidates.append(module_dir.parent)

    # Current working directory and its parents are useful when running from
    # ~/dev_ws/src/LLM_Ros after sourcing an installed workspace.
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    # ROS 2 install: prefer package share if ament_index_python is available.
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("llm_skill_robot")))
    except Exception:
        pass

    # Editable/symlink installs can land next to site-packages; walk upward too.
    candidates.extend(module_dir.parents)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def get_project_root() -> Path:
    """Get the project root directory.

    This works for both:
    - Direct execution: python -m llm_skill_robot.main_cli
    - ROS 2 installation: ros2 run llm_skill_robot llm-skill-demo
    """
    for root in _candidate_project_roots():
        if (root / "config").is_dir() or (root / "skills").is_dir():
            return root
    return Path(__file__).resolve().parent.parent


def get_config_dir() -> Path:
    """Use an explicit workstation configuration directory, then package defaults.

    An invalid override fails rather than silently loading another robot's limits.
    """
    override = os.environ.get("LLM_SKILL_CONFIG_DIR")
    if override:
        config_dir = Path(override).expanduser().resolve()
        if not config_dir.is_dir():
            raise FileNotFoundError(f"LLM_SKILL_CONFIG_DIR is not a directory: {config_dir}")
        return config_dir
    for root in _candidate_project_roots():
        config_dir = root / "config"
        if config_dir.is_dir():
            return config_dir
    searched = ", ".join(str(root / "config") for root in _candidate_project_roots())
    raise FileNotFoundError(f"Config directory not found. Searched: {searched}")


def get_skills_dir() -> Path:
    """Get the skills directory path."""
    for root in _candidate_project_roots():
        skills_dir = root / "skills"
        if skills_dir.is_dir():
            return skills_dir
    searched = ", ".join(str(root / "skills") for root in _candidate_project_roots())
    raise FileNotFoundError(f"Skills directory not found. Searched: {searched}")


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative file path across source and ROS install layouts."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for root in _candidate_project_roots():
        resolved = root / candidate
        if resolved.exists():
            return resolved
        # Convenience for commands run from a workspace root such as ~/dev_ws.
        source_resolved = root / "src" / "LLM_Ros" / candidate
        if source_resolved.exists():
            return source_resolved
    return get_project_root() / candidate
