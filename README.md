# MedQA-Multiagent

A multi-agent AI system for medical question answering that dynamically assembles teams of specialized medical experts based on case complexity.

## Overview

MedQA-Multiagent is an intelligent medical question-answering system that leverages OpenAI's GPT models to simulate multi-disciplinary team (MDT) consultations. The system automatically:

1. **Assesses complexity** of medical queries
2. **Recruits appropriate specialists** based on case requirements
3. **Facilitates collaborative decision-making** using proven medical communication frameworks (SBAR)
4. **Tracks costs** and performance metrics

The system implements three distinct workflows based on case complexity:

- **Low Complexity**: Single Primary Care Physician handles the case
- **Moderate Complexity**: Multi-Disciplinary Team (MDT) with iterative consensus building
- **High Complexity**: Integrated Care Team (ICT) with three specialized sub-teams

## Features

- **Intelligent Complexity Assessment**: Automatically classifies medical questions into low, moderate, or high complexity
- **Dynamic Team Assembly**: Recruits relevant medical specialists based on case-specific keywords and requirements
- **SBAR Communication Framework**: Uses Situation-Background-Assessment-Recommendation format for structured assessments
- **Iterative Consensus Building**: Implements Delphi-lite consensus methodology with moderator feedback
- **Cost Tracking**: Monitors API usage and costs by model type and complexity level
- **Comprehensive Logging**: Tracks all agent interactions, decisions, and token usage
- **16+ Medical Specialties**: Includes cardiologists, neurologists, oncologists, emergency medicine, and more

## Architecture

### Agent Hierarchy

```
┌─────────────────────────────────────────────┐
│         Complexity Checker                   │
│  (Classifies case as low/moderate/high)     │
└──────────────┬──────────────────────────────┘
               │
        ┌──────┴──────┐
        │  Recruiter   │
        │    Agent     │
        └──────┬───────┘
               │
   ┌───────────┼───────────┐
   │           │           │
   ▼           ▼           ▼
 LOW      MODERATE       HIGH
 PCP         MDT          ICT
```

### Complexity-Based Workflows

#### Low Complexity (Single Agent)
- Primary Care Physician makes decision independently
- Best for: routine checkups, common conditions, straightforward diagnoses

#### Moderate Complexity (MDT - Multi-Disciplinary Team)
- Team Lead (Internal Medicine) coordinates
- 4 Specialist Agents selected based on relevance
- Process:
  1. Silent SBAR assessments
  2. Round-robin discussion
  3. Iterative consensus building (up to 5 rounds)
  4. Moderator feedback if no consensus
  5. Team Lead makes final decision

#### High Complexity (ICT - Integrated Care Team)
- Three specialized sub-teams:
  - **Initial Assessment Team (IAT)**: Emergency triage and primary assessment
  - **Diagnostic Evidence Team (DET)**: Laboratory and imaging analysis
  - **Final Review & Decision Team (FRDT)**: Senior consultation and treatment planning
- Each team has 3 agents with specific expertise

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenAI API key
- (Optional) Weights & Biases account for experiment tracking

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/VasistP/MedQA-Multiagent.git
   cd MedQA-Multiagent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   WANDB_API_KEY=your_wandb_key_here  # Optional
   ```

4. **Download the MedQA dataset**
   ```bash
   python src/utils/download_dataset.py
   ```

5. **Verify setup**
   ```bash
   python setup.py
   ```

   You should see:
   ```
   ✅ Python version OK
   ✅ OpenAI API key found
   ✅ MedQA dataset found
   ✅ Environment setup complete!
   ```

## Configuration

Edit `config.py` to customize settings:

```python
# Model Configuration
MODEL_NAME = "gpt-4o"           # OpenAI model to use
MAX_TOKENS = 1500               # Maximum tokens per response
TEMPERATURE = 0.5               # Response temperature (0.0-1.0)

# Dataset Configuration
NUM_SAMPLES = 50                # Number of questions to process
RANDOM_SEED = 42                # For reproducible experiments

# Agent Configuration
MAX_ROUNDS = 5                  # Maximum consensus rounds for MDT
MAX_TURNS = 5                   # Maximum discussion turns
NUM_AGENTS = {
    "low": 1,                   # Number of agents for low complexity
    "moderate": 5,              # Team size for moderate complexity
    "high": 9                   # Total agents for high complexity (3x3)
}

# Logging
LOG_DIR = "./logs/{timestamp}"  # Log output directory
WANDB_PROJECT = "mdagents-medical"  # Weights & Biases project name
```

## Usage

### Basic Usage

Run the main script to process medical questions:

```bash
python main.py
```

### Using Individual Components

#### 1. Complexity Assessment

```python
from src.agents.complexity_checker import ComplexityChecker

checker = ComplexityChecker()
question = "A 55-year-old with chest pain and shortness of breath..."
complexity = checker.check_complexity(question)
print(f"Complexity: {complexity}")  # Output: moderate
```

#### 2. Recruiting Specialists

```python
from src.agents.recruiter import RecruiterAgent

recruiter = RecruiterAgent()
team = recruiter.recruit_for_moderate_complexity(question)

for agent_info in team:
    print(f"{agent_info['specialty']}: {agent_info['relevance_score']}")
```

#### 3. Cost Tracking

```python
from src.utils.cost_tracker import CostTracker

tracker = CostTracker(log_dir="./logs")
cost = tracker.add_api_call(
    model="gpt-4",
    input_tokens=1500,
    output_tokens=300,
    complexity="moderate"
)
print(f"API call cost: ${cost:.4f}")

# Get summary
summary = tracker.get_summary()
print(f"Total cost: {summary['total_cost']}")
print(f"Total tokens: {summary['total_tokens']}")
```

## Project Structure

```
MedQA-Multiagent/
├── config.py                    # Configuration settings
├── main.py                      # Main entry point
├── setup.py                     # Environment setup checker
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Base agent class with OpenAI integration
│   │   ├── medical_agent.py     # Base medical agent with SBAR support
│   │   ├── complexity_checker.py # Complexity classification agent
│   │   ├── recruiter.py         # Specialist recruitment agent
│   │   ├── pcp_agent.py         # Primary Care Physician agent
│   │   ├── specialist_agent.py  # Specialist agent for MDT
│   │   ├── team_lead_agent.py   # Team Lead for MDT coordination
│   │   └── medical_specialties.py # Medical specialty definitions
│   │
│   └── utils/
│       ├── cost_tracker.py      # API cost tracking utility
│       ├── logger.py            # Logging utility
│       ├── data_loader.py       # MedQA dataset loader
│       └── download_dataset.py  # Dataset download script
│
├── data/
│   └── medqa/                   # MedQA dataset (downloaded separately)
│
└── logs/                        # Experiment logs and cost tracking
    └── {timestamp}/
        ├── cost_tracking.json   # Cost breakdown
        └── experiment_log.json  # Full conversation logs
```

## Medical Specialties

The system supports 16+ medical specialties:

**Primary Care:**
- Primary Care Physician
- Internal Medicine

**Organ System Specialists:**
- Cardiologist (heart & cardiovascular)
- Pulmonologist (lungs & respiratory)
- Gastroenterologist (digestive system)
- Nephrologist (kidneys & renal)
- Neurologist (brain & nervous system)
- Endocrinologist (hormones & metabolism)

**Surgical Specialists:**
- General Surgeon
- Orthopedic Surgeon

**Other Specialists:**
- Emergency Medicine
- Pediatrician
- Psychiatrist
- Dermatologist
- Hematologist
- Infectious Disease
- Rheumatologist
- Oncologist
- Radiologist
- Pathologist

Each specialty has:
- Defined expertise areas
- Keyword matching for relevance scoring
- Specialty-specific assessment capabilities

## Cost Tracking

The system tracks API costs across different dimensions:

- **By Model**: Separate tracking for GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **By Complexity**: Track costs for low/moderate/high complexity cases
- **Token Usage**: Input tokens, output tokens, and total tokens
- **Call Counts**: Number of API calls per category

Cost data is automatically saved to `logs/{timestamp}/cost_tracking.json`.

### Current Pricing (as of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| gpt-3.5-turbo | $0.0005 | $0.0015 |
| gpt-4 | $0.01 | $0.03 |
| gpt-4-turbo | $0.01 | $0.03 |

## Development

### Running Tests

```bash
# Test complexity checker
python -c "from src.agents.complexity_checker import ComplexityChecker; print(ComplexityChecker().check_complexity('Simple headache case'))"

# Test recruiter
python -c "from src.agents.recruiter import RecruiterAgent; r = RecruiterAgent(); print(r.recruit_for_low_complexity('test'))"
```

### Adding New Specialties

Edit `src/agents/medical_specialties.py`:

```python
SPECIALTIES = {
    "New Specialty": {
        "expertise": "description of expertise",
        "keywords": ["keyword1", "keyword2", "keyword3"]
    },
    # ... existing specialties
}
```

## Performance Metrics

The system logs comprehensive metrics for each case:

- Complexity classification accuracy
- Specialist selection relevance scores
- Consensus achievement rates
- Number of rounds to consensus
- Token usage per agent/phase
- Total cost per case
- Decision accuracy (when ground truth available)

## Limitations

- Requires OpenAI API access and credits
- Not a substitute for real medical consultation
- Performance depends on quality of medical question dataset
- Costs can accumulate with high-complexity cases (9 agents × multiple rounds)
- Response quality varies with model selection (GPT-4 > GPT-3.5-turbo)

## Troubleshooting

### Common Issues

**Issue: `OPENAI_API_KEY not found`**
- Solution: Create a `.env` file with your API key

**Issue: `MedQA dataset not found`**
- Solution: Run `python src/utils/download_dataset.py`

**Issue: High API costs**
- Solution: Reduce `NUM_SAMPLES` in `config.py` or use GPT-3.5-turbo
- Monitor costs in real-time via `cost_tracking.json`

**Issue: Consensus not reached after max rounds**
- Solution: This is expected for genuinely ambiguous cases
- The Team Lead makes the final decision based on available evidence

## Citation

If you use this code in your research, please cite:

```bibtex
@software{medqa_multiagent_2024,
  title={MedQA-Multiagent: A Multi-Agent System for Medical Question Answering},
  author={Your Name},
  year={2024},
  url={https://github.com/VasistP/MedQA-Multiagent}
}
```

## License

[Specify your license here - MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Acknowledgments

- MedQA dataset creators
- OpenAI for GPT models
- Medical professionals who inspired the MDT framework

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [https://github.com/VasistP/MedQA-Multiagent/issues](https://github.com/VasistP/MedQA-Multiagent/issues)
- Email: [Your email]

---

**Disclaimer**: This is a research tool for educational purposes. It is not intended for clinical use or real medical diagnosis. Always consult qualified healthcare professionals for medical advice.
