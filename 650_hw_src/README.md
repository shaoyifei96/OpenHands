# Homework Agent Using OpenHands

This folder contains scripts to run an OpenHands agent on your homework assignments. The agent will read the homework instructions and attempt to complete the assignment.

## Prerequisites

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) installed and configured
- Python 3.9+
- Poetry
- Docker

## Configuration

1. Copy the template configuration:
   ```bash
   cp 650_hw_src/config.template.toml config.toml
   ```

2. Edit the `config.toml` file to add your API keys:
   ```toml
   [llm.claude]
   model = "claude-3-opus-20240229"
   api_key = "your-anthropic-api-key"
   custom_llm_provider = "anthropic"
   ```

   Or set environment variables:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

## Usage

### Using the shell script (recommended)

Run the shell script with the desired parameters:

```bash
chmod +x 650_hw_src/run_agent.sh
./650_hw_src/run_agent.sh --llm-config claude --hw-dir 650_hw
```

### Available options

- `--llm-config`: LLM configuration to use (default: `claude`)
- `--hw-dir`: Path to the homework directory (default: `650_hw`)
- `--agent-cls`: Agent class to use (default: `CodeActAgent`)
- `--max-iterations`: Maximum number of iterations (default: 50)

### Running directly with Python

```bash
poetry run python 650_hw_src/call_agent.py \
  --hw-dir 650_hw \
  --agent-cls CodeActAgent \
  --max-iterations 50 \
  --llm-config claude
```

## Output

The script creates a new directory with the format:
```
<hw_dir>_<agent_cls>_<timestamp>/
```

This directory contains:
- All the modified files from the agent
- `changes.patch`: A git patch showing all changes made
- `agent_history.log`: Complete conversation history with the agent

## Technical Details

This script uses a pre-built OpenHands runtime container image (`ghcr.io/all-hands-ai/runtime:0.29-nikolaik`) to create a sandbox environment for the agent to work in. The agent completes your homework in this container and then extracts the changes back to your local machine. 