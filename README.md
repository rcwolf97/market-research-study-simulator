# Market Research Study Simulator

AI-powered market research study generator and interview simulator.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your OpenAI API key:**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Usage

### Generate Studies
```python
from user_simulator.study_designer import design_studies_for_population

studies = design_studies_for_population("pulmonologists")
```

### Run Interview Simulation
```python
import asyncio
from user_simulator.simulator import Simulator

sim = Simulator(study="study_001", number_of_users=5)
dialogue = asyncio.run(sim.simulate_conversation(0))
```

### Generate User Profiles
```python
from user_simulator.simulate_profiles import generate_profile

profile = generate_profile(age="45", gender="Female", state="California")
```

## Requirements
- Python 3.8+
- OpenAI API key