import yaml
from typing import Dict, List, Optional
from pathlib import Path
from .skill import Skill
from ..utils import get_skills_dir


class SkillLibrary:
    def __init__(self, skills_dir: str = None):
        if skills_dir is None:
            self.skills_dir = get_skills_dir()
        else:
            self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self.load_skills()

    def load_skills(self):
        """Load all skill definitions from YAML files in the skills directory."""
        if not self.skills_dir.exists():
            raise FileNotFoundError(f"Skills directory {self.skills_dir} does not exist")

        for yaml_file in self.skills_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                skill_data = yaml.safe_load(f)
                skill = Skill(**skill_data)
                self.skills[skill.skill_id] = skill

    def list_skills(self) -> List[str]:
        """Return a list of available skill IDs."""
        return list(self.skills.keys())

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by its ID."""
        return self.skills.get(skill_id)