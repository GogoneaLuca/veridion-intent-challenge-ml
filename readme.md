# Veridion Search Intent Challenge - ML Engineer Intern
**Candidate:** Gogonea Luca

## Project Overview
This repository contains a hybrid Search Engine designed to process complex B2B search intents. The system follows a 3-stage "Funnel" architecture to balance semantic understanding with strict attribute filtering and high-performance retrieval.

## Repository Structure
* `solution.py`: The core search engine implementation containing the Intent Parser, Fast Retriever, and Deep Judge modules.
* `WRITEUP.md`: A detailed technical report covering the architecture, engineering decisions, trade-offs, and error analysis.
* `data_exploration.py`: An Exploratory Data Analysis (EDA) script used to profile the dataset and identify data quality issues.
* `requirements.txt`: List of Python dependencies required to run the project.
* `companies.jsonl`: The source dataset.

## Setup and Installation

1. Clone the repository and navigate to the project folder.
2. Ensure you have Python 3.8+ installed.
3. Install the required libraries using pip:
    pip install -r requirements.txt

## How to Run
To ensure a frictionless evaluation, a temporary Groq API key has been included directly in the `solution.py` script. You do not need to configure environment variables.

### 1. Run Official Test Cases
To execute the engine against the 12 official test queries provided in the challenge:
    python solution.py

### 2. Run a Custom Query
To test the system with a specific natural language intent:
    python solution.py --query "B2B SaaS companies in Europe with more than 100 employees"

## Documentation
For a deep dive into the engineering reasoning, the "Shopify Trap" case study, and scaling considerations, please refer to the **WRITEUP.md** file.