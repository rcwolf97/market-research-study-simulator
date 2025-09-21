import os
import json
from tqdm import tqdm
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

from agents import Runner

from .simulate_profiles import generate_profile
from .context import StudyContext
from .agents import user_agent, market_researcher_agent
from .constant import USER_PROFILE_DICT


class Simulator:
    """High-level orchestration for profile generation and interview simulation."""

    def __init__(
        self,
        study: str = "study_1",
        number_of_users: int = 30,
        user_population: str = "pulmonologist",
        simulation_id: Optional[str] = None,
    ) -> None:

        self.simulation_id = simulation_id or datetime.now().strftime("%Y%m%d%H%M%S")
        self.number_of_users = number_of_users
        self.user_population = user_population
        self.profile_dict = USER_PROFILE_DICT

        # Create directory for simulation data
        self.data_root = "../data"
        self.sim_root = os.path.join(self.data_root, "simulation", self.simulation_id)

        # Create conversations directory for saving dialogues
        self.conversations_dir = os.path.join(self.sim_root, "conversations")
        os.makedirs(self.sim_root, exist_ok=True)
        os.makedirs(self.conversations_dir, exist_ok=True)

        # import study json
        study_path = os.path.join(self.data_root, "studies", f"{study}.json")
        if not os.path.exists(study_path):
            raise FileNotFoundError(f"Study {study} not found")

        with open(study_path, "r") as f:
            self.study = json.load(f)

        # Generate or load user profiles for this simulation
        self.user_profiles_path = os.path.join(self.sim_root, "user_profiles.json")
        if not os.path.exists(self.user_profiles_path):
            self.user_profiles = self.generate_users()
        else:
            with open(self.user_profiles_path, "r") as f:
                self.user_profiles = json.load(f)

    def generate_users(self) -> List[Dict[str, Any]]:
        """Generate user profiles and persist them for this simulation run."""

        generated_profiles: List[Dict[str, Any]] = []
        for _ in tqdm(range(self.number_of_users), desc="Generating users"):
            # Draw a seed persona and expand via LLM
            profile_seed = {
                "age_range": random.choice(self.profile_dict["age"]),
                "gender": random.choice(self.profile_dict["gender"]),
                "urban": random.choice(self.profile_dict["urban"]),
                "academic": random.choice(self.profile_dict["academic"]),
                "state": random.choice(self.profile_dict["state"]),
                "other_characteristics": None,
            }
            age = random.randint(profile_seed["age_range"][0], profile_seed["age_range"][1])

            # simulate user profile
            profile = generate_profile(
                age=str(age),
                gender=profile_seed["gender"],
                urban=profile_seed["urban"],
                academic=profile_seed["academic"],
                state=profile_seed["state"],
                other_characteristics=profile_seed["other_characteristics"],
                number_of_profile=5,
            )

            profile_str = f"{age}yo {profile_seed['gender']}, {profile_seed['urban']}, {profile_seed['academic']}, {profile_seed['state']}"
            profile["profile"] = profile_str

            generated_profiles.append(profile)

        # save user profiles
        with open(self.user_profiles_path, "w") as f:
            json.dump(generated_profiles, f, indent=2)

        return generated_profiles

    def _initial_context(self, user_profile: Dict[str, Any]) -> StudyContext:
        # The guide is organized as an object with intro and blocks list
        first_block = self.study["discussion_guide"]["blocks"][0]
        return StudyContext(
            discussion_block=first_block,
            user_profile=user_profile,
            user_population=self.user_population,
            study_title=self.study['study_name'],
            study_summary=self.study['study_summary'],
        )

    def _next_block(self, index: int) -> Optional[Dict[str, Any]]:
        blocks = self.study["discussion_guide"]["blocks"]
        if 0 <= index < len(blocks):
            return blocks[index]
        return None

    def save_conversation(self, index: int, dialogue: List[Dict[str, str]], user_profile: Dict[str, Any]) -> str:
        """Save a conversation with its associated user profile to JSON."""
        conversation_data = {
            "profile": user_profile,
            "dialogue": dialogue,
            "metadata": {
                "simulation_id": self.simulation_id,
                "user_index": index,
                "study": self.study.get("study_name", "Unknown Study"),
                "timestamp": datetime.now().isoformat(),
                "total_turns": len(dialogue),
                "user_population": self.user_population
            }
        }
        
        # Create filename with timestamp and index for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{index:03d}_{timestamp}.json"
        filepath = os.path.join(self.conversations_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"Conversation saved: {filename}")
        return filepath

    async def simulate_conversation(self, index: int) -> List[Dict[str, str]]:
        """Simulate a conversation with a user."""
        context = self._initial_context(self.user_profiles[index])

        dialogue: List[Dict[str, str]] = []
        block_index = 0

        while True:
            agent_input = dialogue if dialogue else [{"role": "user", "content": "Hi"}]
            researcher_result = await Runner.run(
                starting_agent=market_researcher_agent,
                input=agent_input,
                context=context,
            )
            researcher_output = researcher_result.final_output

            if researcher_output and getattr(researcher_output, "finished", False):
                block_index += 1
                next_block = self._next_block(block_index)
                
                if next_block is None:
                    break
                context.discussion_block = next_block
                continue

            elif not researcher_output or not getattr(researcher_output, "next_question", None):
                raise ValueError("No next question from market researcher")

            question = researcher_output.next_question
            dialogue.append({"role": "assistant", "content": question})
            print("Researcher: {}".format(question))

            # Limit user agent to only see last 4 messages for more human-like memory
            user_dialogue = dialogue[-4:] if len(dialogue) > 4 else dialogue
            
            user_result = await Runner.run(
                starting_agent=user_agent,
                input=user_dialogue,
                context=context,
            )
            user_answer = user_result.final_output or ""
            dialogue.append({"role": "user", "content": user_answer})
            print("User: {}".format(user_answer))

        # Save the conversation with the user profile
        self.save_conversation(index, dialogue, self.user_profiles[index])
        
        return dialogue
