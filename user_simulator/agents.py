from typing import Dict, Optional
from pydantic import BaseModel

from .utils import load_prompt
from .context import StudyContext
from .hooks import SystemInstructionsHook
from agents import Agent, RunContextWrapper, ModelSettings, AgentOutputSchema

USER_PROMPT_NAME = "user_simulator"
USER_PROMPT_VERSION = "v0.0.1"

MARKET_RESEARCHER_PROMPT_NAME = "interviewer_simulator"
MARKET_RESEARCHER_PROMPT_VERSION = "v0.0.1"


class ResearcherOutput(BaseModel):
    """
    Structured output schema for the market researcher agent.
    
    This model defines the expected output format from the market researcher
    agent, controlling conversation flow and question progression.
    """
    next_question: Optional[str]
    finished: bool


def user_instructions(
    run_context: RunContextWrapper[StudyContext],
    agent: Agent[StudyContext]
) -> str:
    """
    Generate user instructions from template for agent configuration.
    """
    template = load_prompt(USER_PROMPT_NAME, USER_PROMPT_VERSION)
    prompt = template.render(
        user_profile=run_context.context.user_profile
    )
    return prompt


def market_researcher_instructions(
    run_context: RunContextWrapper[StudyContext],
    agent: Agent[StudyContext]
) -> str:
    """
    Generate market researcher instructions from template for agent configuration.
    """
    template = load_prompt(MARKET_RESEARCHER_PROMPT_NAME, MARKET_RESEARCHER_PROMPT_VERSION)

    prompt = template.render(
        population=run_context.context.user_population,
        study_title=run_context.context.study_title,
        study_summary=run_context.context.study_summary,
        discussion=run_context.context.discussion_block,
    )
    return prompt


user_agent = Agent(
    name="User",
    instructions=user_instructions,
    model_settings=ModelSettings(temperature=0.8, max_tokens=500),  # Higher temp for variation, shorter responses
    model="gpt-4.1-2025-04-14",
    tools=[],
    hooks=SystemInstructionsHook(),
)

market_researcher_agent = Agent(
    name="Market Researcher", 
    instructions=market_researcher_instructions,
    model="gpt-4.1-2025-04-14",
    model_settings=ModelSettings(temperature=0.3, max_tokens=500),  # Lower temp for consistency, shorter questions
    tools=[],
    output_type=AgentOutputSchema(ResearcherOutput, strict_json_schema=False),
)