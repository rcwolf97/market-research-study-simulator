import json
import os
from typing import List, Dict, Any

import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from .utils import load_prompt

load_dotenv("../.env")


class Study(BaseModel):
    """Model for a single study output."""
    study_title: str
    study_summary: str
    research_motivation: str
    research_directive: str

class StudyList(BaseModel):
    """Model for multiple studies output."""
    studies: List[Study]

class BlockQuestion(BaseModel):
    """Model for a single block question output."""
    big_question: str
    probes: List[str]

class Block(BaseModel):
    """Model for a single block output."""
    title: str
    questions: List[BlockQuestion]

class DiscussionGuide(BaseModel):
    """Model for discussion guide JSON structure."""
    intro: str
    blocks: List[Block]

class StudyDesigner:
    """Class to design studies and generate discussion guides for market research."""
    
    def __init__(self):
        """Initialize the StudyDesigner with OpenAI client."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Model configurations for different tasks
        self.study_generation_config = {
            "model": "gpt-5-2025-08-07",
            "tools": [{ "type": "web_search_preview" }],
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 40000,
            "text_format": StudyList,
        }
        
        self.discussion_guide_config = {
            "model": "gpt-5-2025-08-07", 
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 10000,
        }
        
        self.json_conversion_config = {
            "model": "gpt-4.1-2025-04-14",
            "temperature": 0.1,
            "max_output_tokens": 4000,
            "text_format": DiscussionGuide,
        }
    
    def design_studies_and_guides(self, population: str) -> List[Dict[str, Any]]:
        """
        Design studies and generate discussion guides for a given population.
        
        Args:
            population: The target population (e.g., "pulmonologists treating COPD patients within the US")
            
        Returns:
            List of complete study dictionaries with discussion guides
        """
        print(f"Starting study design for population: {population}")
        
        # Step 1: Generate studies using the generate_studies prompt
        print("Generating studies...")
        studies_data = self.generate_studies(population)
        
        complete_studies = []
        
        # Step 2: For each study, generate discussion guide and convert to final format
        for i, study in enumerate(studies_data):
            print(f"Processing study {i+1}/{len(studies_data)}: {study['study_title']}")
            
            # Generate discussion guide
            discussion_guide_text = self.generate_discussion_guide(
                study['study_title'],
                study['research_directive'], 
                study['research_motivation']
            )
            
            # Convert discussion guide to JSON structure
            discussion_guide_json = self.convert_discussion_guide_to_json(
                discussion_guide_text
            )
            
            # Create complete study structure
            complete_study = {
                "study_id": f"{str(i+1).zfill(3)}",
                "study_name": study['study_title'],
                "study_summary": study['study_summary'],
                "research_motivation": study['research_motivation'],
                "research_directive": study['research_directive'],
                "discussion_guide": discussion_guide_json
            }
            
            complete_studies.append(complete_study)
        
        # Step 3: Save all studies
        self.save_studies(complete_studies)
        
        print(f"Successfully created {len(complete_studies)} studies with discussion guides")
        return complete_studies
    
    def generate_studies(self, population: str) -> List[Dict[str, Any]]:
        """Generate studies using the generate_studies prompt."""
        
        # Load the generate_studies prompt template
        template = load_prompt("generate_studies", "v0.0.1")
        prompt = template.render(population=population)
        
        # Call OpenAI Responses API
        response = self.client.responses.parse(
            input=[{"role": "user", "content": prompt}],
            **self.study_generation_config
        )
        print(response.__dict__)
        # Extract structured output
        studies_data = [study.model_dump() for study in response.output_parsed.studies]
        print(f"Generated {len(studies_data)} studies")
        return studies_data
    
    def generate_discussion_guide(self, study_name: str, research_directive: str, 
                                       research_motivation: str) -> str:
        """Generate discussion guide using the discussion_guide prompt."""
        
        # Load the discussion_guide prompt template
        template = load_prompt("discussion_guide", "v0.0.1")
        prompt = template.render(
            study_name=study_name,
            research_directive=research_directive,
            research_motivation=research_motivation
        )
        
        # Call OpenAI Responses API
        response = self.client.responses.parse(
            input=[{"role": "user", "content": prompt}],
            **self.discussion_guide_config
        )
        
        return response.output_text
    
    def convert_discussion_guide_to_json(self, discussion_guide: str) -> Dict[str, Any]:
        """Convert discussion guide text to JSON using convert_discussion_guide prompt."""
        
        # Load the convert_discussion_guide prompt template
        template = load_prompt("convert_discussion_guide", "v0.0.1")
        prompt = template.render(discussion_guide=discussion_guide)
        
        # Call OpenAI Responses API
        response = self.client.responses.parse(
            input=[{"role": "user", "content": prompt}],
            **self.json_conversion_config
        )
        
        # Extract structured output
        return response.output_parsed.model_dump()
    
    def generate_study_summary(self, study: Dict[str, Any]) -> str:
        """Generate a study summary from the study data."""
        # Create a concise summary combining motivation and directive
        return f"This study {study['research_motivation']} The research focuses on {study['research_directive'][:200]}..."
    
    def save_studies(self, studies: List[Dict[str, Any]]) -> None:
        """Save studies to the data/studies directory."""
        
        # Ensure the studies directory exists
        studies_dir =  "../data/studies"
        os.makedirs(studies_dir, exist_ok=True)
        
        # Save each study as a separate JSON file
        for study in studies:
            filename = f"study_{study['study_id']}.json"
            filepath = os.path.join(studies_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(study, f, indent=4, ensure_ascii=False)
            
            print(f"Saved study to {filepath}")


def design_studies_for_population(population: str) -> List[Dict[str, Any]]:
    """
    Function to design studies and generate discussion guides for a population.
    """
    designer = StudyDesigner()
    return designer.design_studies_and_guides(population)
