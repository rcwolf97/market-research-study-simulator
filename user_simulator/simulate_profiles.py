import os
import random
from typing import Optional, Dict, List, Any

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from .utils import load_prompt

load_dotenv("../.env")


class Profile(BaseModel):
    professional_background: str
    practice_setting: str
    treatment_philosophy: str
    personal_notes: str
    communication_style: str


class ProfileList(BaseModel):
    profiles: List[Profile]


def generate_profile(
    age: Optional[str] = None,
    gender: Optional[str] = None,
    urban: Optional[str] = None,
    academic: Optional[str] = None,
    state: Optional[str] = None,
    other_characteristics: Optional[str] = None,
    number_of_profile: int = 1,
) -> Dict[str, Any]:
    """
    Generate a synthetic clinician profile using the OpenAI Responses API.
    
    Args:
        age: Age or age range for the clinician profile
        gender: Gender specification for the profile
        urban: Urban/rural/suburban setting designation
        academic: Academic vs non-academic practice setting
        state: US state for geographic context
        other_characteristics: Additional profile characteristics
        number_of_profile: Number of profiles to generate (returns random selection)
        
    Returns:
        A dictionary containing the selected clinician profile data
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    template = load_prompt("generate_users", "v0.0.1")

    prompt = template.render(
        number_of_profile=number_of_profile,
        age=age,
        gender=gender,
        urban=urban,
        academic=academic,
        state=state,
        other_characteristics=other_characteristics,
    )

    messages = [
        {"role": "system", "content": prompt},
    ]

    response = client.responses.parse(
        input=messages,
        model="gpt-4.1",
        max_output_tokens=512 * number_of_profile,
        temperature=0.5,
        text_format=ProfileList,
        reasoning=None,
    )

    selected_profile = random.choice(response.output_parsed.profiles)
    return selected_profile.model_dump()