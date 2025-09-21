from typing import Optional, List
import random

from agents import AgentHooks, RunContextWrapper, Agent
from agents.items import TResponseInputItem

from .context import StudyContext


class SystemInstructionsHook(AgentHooks[StudyContext]):
    """Agent hook that injects system instructions to make user responses more human-like."""

    async def on_llm_start(
        self,
        context: RunContextWrapper[StudyContext],
        agent: Agent[StudyContext],
        system_prompt: Optional[str],
        input_items: List[TResponseInputItem],
    ) -> None:
        """Inject human-like behavior instructions before LLM calls for the user agent."""
            
        # Create human-like system message based on user profile
        user_profile = context.context.user_profile
        
        # Extract key profile elements for personalization
        profile_elements = []
        if "professional_background" in user_profile:
            profile_elements.append(f"Professional background: {user_profile['professional_background']}")
        if "practice_setting" in user_profile:
            profile_elements.append(f"Practice setting: {user_profile['practice_setting']}")
        if "communication_style" in user_profile:
            profile_elements.append(f"Communication style: {user_profile['communication_style']}")
        
        profile_context = "\n".join(profile_elements) if profile_elements else "Medical professional"
        
        style_note = f"Communication style: {user_profile.get('communication_style', 'Professional')}"
        
        base_instruction = f"""Respond as a real clinician. {style_note}

Avoid AI patterns:
- Don't start with "That's a good question"
- Don't perfectly mirror question structure  
- Include natural speech patterns and minor imperfections
- Reference your actual practice context when relevant
"""

        # Inject the system message at the beginning of input_items
        system_message: TResponseInputItem = {
            "role": "system",
            "content": base_instruction
        }
        
        # Insert at the beginning to ensure it's processed first
        input_items.insert(0, system_message)
        
        # Add random conversational friction and specific details
        friction_instruction = self.generate_friction_instruction()
        if friction_instruction:
            friction_message: TResponseInputItem = {
                "role": "system", 
                "content": friction_instruction
            }
            input_items.insert(1, friction_message)
    
    def generate_friction_instruction(self) -> Optional[str]:
        """Randomly generate instructions for conversational friction and realistic details."""
        
        injected_list = []

        # 30% chance of adding low frequency friction
        if random.random() < 0.3:
            low_frequency_friction = [
                "Keep this response SHORT (1-2 sentences max)",
                "This is a complex topic for you - give a longer, more thoughtful response with specific examples",
                "In this response, include a brief false start or self-correction (e.g., 'Well, I usually... actually, let me think about that differently...')",
                "For this response, reference a very specific recent case with messy details (exact dates, specific numbers, real frustrations)",
                "In this answer, show some uncertainty or admit a knowledge gap rather than being overly confident",            
                "Reference a specific practice constraint or workaround you've had to develop (EMR quirks, insurance hassles, scheduling issues)",
                "Mention a specific time period or event that anchors your experience ('last winter when COVID cases spiked,' 'after the Epic upgrade,' 'during the formulary change')",
                "Show mild emotion about something in your practice - frustration, satisfaction, surprise, or concern",
                "Include a specific detail that reveals your practice's unique context (rural patient travel times, academic teaching load, specific payer mix, etc.)",
                "Reference a colleague interaction or case discussion that influenced your thinking",
                "Mention a patient outcome that surprised you or changed your approach slightly",
                "Use some colloquial language or sentence fragments that match your age and background",
            ]
            injected_friction = random.choice(low_frequency_friction)
            injected_list.append(injected_friction)

        # 50% chance of adding high frequency friction of keeping the response short
        if random.random() < 0.5:
            high_frequency_friction = "Keep this response SHORT (1-2 sentences max)"
            injected_list.append(high_frequency_friction)
        return "\n".join(injected_list) if injected_list else None
