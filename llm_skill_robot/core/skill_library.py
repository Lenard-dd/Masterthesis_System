import yaml
from typing import Dict, List, Optional
from pathlib import Path
from .skill import Skill


class SkillLibrary:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = self._resolve_skills_dir(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self.load_skills()

    def _resolve_skills_dir(self, skills_dir: str) -> Path:
        path = Path(skills_dir)
        if path.exists():
            return path

        if skills_dir == "skills":
            package_root = Path(__file__).resolve().parents[1]
            package_skills = package_root / "skills"
            if package_skills.exists():
                return package_skills

        return path

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