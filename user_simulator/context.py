from typing import Dict, Any


class StudyContext:
    """
    Mutable run context shared across agents during a study simulation.
    
    This class holds the study metadata, current discussion block being processed,
    and the user profile being simulated. It enables agents to maintain consistent
    context throughout a conversation simulation.
    """

    def __init__(
        self,
        study_title: str,
        study_summary: str,
        user_population: str,
        discussion_block: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> None:
        """
        Initialize the study context with required simulation parameters.
        """
        self.study_title = study_title
        self.study_summary = study_summary
        self.user_population = user_population
        self.discussion_block = discussion_block
        self.user_profile = user_profile