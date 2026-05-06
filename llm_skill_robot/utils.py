"""Utility functions for resolving project paths."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.
    
    This works for both:
    - Direct execution: python -m llm_skill_robot.main_cli
    - ROS 2 installation: ros2 run llm_skill_robot llm-skill-demo
    """
    # Start from the current module location
    module_dir = Path(__file__).resolve().parent.parent  # llm_skill_robot

    # Try to find project root by looking for config/ and skills/ directories
    root = module_dir.parent  # Go up to project root
    if (root / "config").exists() and (root / "skills").exists():
        return root

    # Fallback: check if we're in an installed package
    # In ROS 2 colcon install, files are placed relative to the package root
    if (root / "config").is_dir():
        return root

    # Last resort: use the module directory
    return module_dir


def get_config_dir() -> Path:
    """Get the config directory path."""
    root = get_project_root()
    config_dir = root / "config"
    if config_dir.exists():
        return config_dir
    raise FileNotFoundError(f"Config directory not found at {config_dir}")


def get_skills_dir() -> Path:
    """Get the skills directory path."""
    root = get_project_root()
    skills_dir = root / "skills"
    if skills_dir.exists():
        return skills_dir
    raise FileNotFoundError(f"Skills directory not found at {skills_dir}")
