#!/usr/bin/env bash
set -eo pipefail

# Default values
HW_DIR="650_hw"
AGENT="CodeActAgent"
MAX_ITER=50
LLM_CONFIG="claude"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --hw-dir)
      HW_DIR="$2"
      shift 2
      ;;
    --agent-cls)
      AGENT="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITER="$2"
      shift 2
      ;;
    --llm-config)
      LLM_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Strip 'llm.' prefix if present in LLM_CONFIG
if [[ "$LLM_CONFIG" == llm.* ]]; then
  LLM_CONFIG="${LLM_CONFIG#llm.}"
  echo "Removing 'llm.' prefix from LLM_CONFIG. Using: $LLM_CONFIG"
fi

echo "Running agent with these parameters:"
echo "  HW_DIR: $HW_DIR"
echo "  AGENT: $AGENT"
echo "  MAX_ITER: $MAX_ITER"
echo "  LLM_CONFIG: $LLM_CONFIG"

# Check for API keys in environment
if [[ "$LLM_CONFIG" == *"claude"* && -z "$ANTHROPIC_API_KEY" ]]; then
  echo "WARNING: ANTHROPIC_API_KEY environment variable not set."
  echo "You'll need to configure a Claude API key in your config.toml under [llm.${LLM_CONFIG}]"
fi

if [[ "$LLM_CONFIG" == *"gpt"* && -z "$OPENAI_API_KEY" ]]; then
  echo "WARNING: OPENAI_API_KEY environment variable not set."
  echo "You'll need to configure an OpenAI API key in your config.toml under [llm.${LLM_CONFIG}]"
fi

# Check if LLM config exists in config.toml
if ! grep -q "\[llm\.${LLM_CONFIG}\]" config.toml; then
  echo "Warning: [llm.${LLM_CONFIG}] section not found in config.toml."
  echo "Make sure your config.toml has a proper LLM configuration."
  echo "Using default configuration from the OpenHands base parser."
fi

# Run the Python script using Poetry
poetry run python 650_hw_src/call_agent.py \
  --hw-dir "$HW_DIR" \
  --agent-cls "$AGENT" \
  --max-iterations "$MAX_ITER" \
  --llm-config "$LLM_CONFIG"