# UAV Drone Sensor Report Analysis LLM

A fine-tuned language model that extracts structured information from unstructured UAV (Unmanned Aerial Vehicle) sensor reports. Converts natural language military reports into consistent JSON format for efficient data processing.

## Features

- **Structured Data Extraction**: Transforms unstructured UAV reports to JSON
- **Military Context**: Specialized for military UAV terminology and scenarios
- **Parameter-Efficient**: Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Production Ready**: Includes comprehensive inference pipeline
- **Colab Compatible**: Runs within Google Colab free tier constraints

## Model Performance

- **Training Loss**: 4.58 → 0.47 (89% improvement)
- **Test Loss**: 0.406
- **Success Rate**: ~83% on real-world examples
- **Trainable Parameters**: 1.1M (0.9% of total parameters)

## Quick Start

### Installation

```bash
git clone https://github.com/EbenezerXNate/uav-llm-analysis.git
cd uav-llm-analysis
pip install -r requirements.txt

## BASIC USAGE

from src.inference import UAVReportAnalyzer

# Initialize the analyzer
analyzer = UAVReportAnalyzer("./models/uav-llm-finetuned")

# Analyze a UAV report
report = "UAV Patrol Alpha detected three thermal signatures at grid 38FRP123456. No civilian activity."
result = analyzer.analyze(report)

if result["success"]:
    print("Extracted data:", result["data"])
else:
    print("Error:", result["error"])


Example Output
Input: "UAV Patrol Alpha detected three thermal signatures at grid 38FRP123456. No civilian activity."

Output:
{
  "unit": "UAV Patrol Alpha",
  "entities": ["thermal signatures"],
  "count": [3],
  "grid": "38FRP123456",
  "civilian_activity": false,
  "confidence": 0.85
}

## Project Structure

uav-llm-analysis/
├── notebooks/
│   └── uav_llm_finetuning.ipynb    # Complete training notebook
├── src/
│   └── inference.py                 # Production inference pipeline
├── models/
│   └── uav-llm-finetuned/          # Fine-tuned model weights
├── examples/
│   └── example_usage.py            # Usage examples
├── requirements.txt
└── README.md


Use Cases
    Military Intelligence: Automate UAV report processing

    Data Analysis: Structured data for analytics pipelines

    Command & Control: Real-time situational awareness

    Training: Synthetic data generation for ML systems

Training Details
    Base Model: microsoft/DialoGPT-small

    Fine-tuning: LoRA (Low-Rank Adaptation)

    Dataset: 1,500 synthetic UAV reports

    Training Time: ~30 minutes on Colab T4 GPU

    Techniques: Mixed-precision training, gradient accumulation

Technical Implementation
The model uses:

    PEFT (Parameter-Efficient Fine-Tuning) with LoRA

    Instruction Tuning format for structured generation

    Synthetic Data Generation for domain adaptation

    JSON Schema Enforcement for consistent outputs

    