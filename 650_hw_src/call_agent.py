#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import os
import shutil
import subprocess
import tempfile
import tarfile
import base64
from pathlib import Path
from typing import Any, Dict, Optional

from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    SandboxConfig,
    get_llm_config_arg,
    get_parser as get_base_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync


def get_parser():
    parser = get_base_parser()
    parser.add_argument(
        "--hw-dir",
        type=str,
        default="650_hw",
        help="Path to the homework directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for the output (default: parent of hw-dir)",
    )
    # We no longer need the base-image parameter since we're using a pre-built image
    return parser


def get_config(args) -> AppConfig:
    """Configure OpenHands based on command-line arguments."""
    # Get LLM config
    llm_config = get_llm_config_arg(args.llm_config)
    
    # If no LLM config found, try to create one from environment variables
    if llm_config is None:
        logger.warning(f"LLM config [llm.{args.llm_config}] not found in config.toml. Attempting to create one from environment variables.")
        
        # Check if the model name suggests Claude
        if "claude" in args.llm_config.lower():
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                logger.info("Using ANTHROPIC_API_KEY from environment variables.")
                llm_config = LLMConfig(
                    api_key=api_key,
                    model="claude-3-opus-20240229",  # Default to a reasonable Claude model
                    custom_llm_provider="anthropic",
                )
            else:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it or configure [llm.claude] in config.toml.")
        
        # Check if the model name suggests GPT
        elif "gpt" in args.llm_config.lower():
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                logger.info("Using OPENAI_API_KEY from environment variables.")
                llm_config = LLMConfig(
                    api_key=api_key,
                    model="gpt-4",  # Default to a reasonable OpenAI model
                    custom_llm_provider="openai",
                )
            else:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it or configure [llm.gpt] in config.toml.")
        
        # If we still don't have a config, raise an error
        if llm_config is None:
            raise ValueError(f"Could not create LLM config for '{args.llm_config}'. Please configure it in config.toml or use a known model type (claude/gpt).")
    
    llm_config.log_completions = True

    # Set up sandbox configuration
    # Use a pre-built runtime container image instead of building one
    sandbox_config = SandboxConfig(
        # Use a pre-built image from OpenHands registry instead of building from base_image
        runtime_container_image="ghcr.io/all-hands-ai/runtime:0.29-nikolaik",
        enable_auto_lint=True,
        use_host_network=False,
        platform="linux/amd64",
    )

    # Set up agent configuration
    agent_config = AgentConfig(
        codeact_enable_jupyter=False,
        codeact_enable_browsing=False,
        codeact_enable_llm_editor=False,
        enable_prompt_extensions=False,
    )

    # Create the full app configuration
    config = AppConfig(
        default_agent=args.agent_cls,
        run_as_openhands=False,
        max_iterations=args.max_iterations,
        runtime="docker",
        sandbox=sandbox_config,
    )
    config.set_llm_config(llm_config)
    config.set_agent_config(agent_config)
    
    return config


def initialize_runtime(runtime: Runtime, hw_dir: str):
    """Initialize the runtime by copying homework files (excluding .txt) to the sandbox."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization')
    logger.info('-' * 30)
    
    # Create workspace directory
    action = CmdRunAction(command='mkdir -p /workspace')
    action.set_hard_timeout(60)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError(f'Failed to create workspace: {str(obs)}')
    
    # Walk the homework directory and create corresponding directories in the container
    for root, dirs, _ in os.walk(hw_dir):
        for dir_name in dirs:
            rel_dir_path = os.path.relpath(os.path.join(root, dir_name), hw_dir)
            container_dir_path = f'/workspace/{rel_dir_path}'
            action = CmdRunAction(command=f'mkdir -p {container_dir_path}')
            action.set_hard_timeout(30)
            runtime.run_action(action)
    
    # List all files in homework directory
    files = []
    for root, _, filenames in os.walk(hw_dir):
        for filename in filenames:
            if filename.startswith('.') and filename != '.gitignore':
                continue  # Skip hidden files except .gitignore
            
            # Only copy specific file types that are needed for code implementation
            # Exclude .txt files from the initial copy
            file_ext = os.path.splitext(filename)[1].lower()
            allowed_extensions = ['.py', '.tex', '.md', '.json', '.ipynb', '.csv', '.yml', '.yaml', '.sh']
            if file_ext == '.txt' or file_ext not in allowed_extensions:
                if file_ext == '.txt':
                    logger.info(f"Skipping initial copy of .txt file: {filename}")
                continue  # Skip .txt files and other non-allowed files
            
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, hw_dir)
            files.append((full_path, rel_path))
    
    # For each file, write content using echo
    for src_path, rel_path in files:
        try:
            # For text files, we can use echo with base64 encoding
            with open(src_path, 'rb') as f:
                content = f.read()
            
            # Use a temporary file approach
            temp_file = f"/tmp/tmp_file_{os.path.basename(rel_path)}"
            # Write content as base64
            b64_content = base64.b64encode(content).decode('utf-8')
            
            # First create the base64 file
            action = CmdRunAction(command=f'echo "{b64_content}" > {temp_file}.b64')
            action.set_hard_timeout(60)
            runtime.run_action(action)
            
            # Then decode it 
            action = CmdRunAction(command=f'base64 -d {temp_file}.b64 > {temp_file}')
            action.set_hard_timeout(60)
            runtime.run_action(action)
            
            # Finally move it to the correct location
            action = CmdRunAction(command=f'mkdir -p /workspace/$(dirname "{rel_path}") && cp {temp_file} /workspace/{rel_path}')
            action.set_hard_timeout(60)
            logger.info(f"Copying {rel_path}", extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
                logger.error(f'Failed to copy {rel_path}: {str(obs)}')
        except Exception as e:
            logger.error(f'Error copying {rel_path}: {str(e)}')
    
    # Initialize git repository in workspace to track changes
    action = CmdRunAction(command='cd /workspace && git init')
    action.set_hard_timeout(60)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError(f'Failed to initialize git: {str(obs)}')
    
    # Configure git
    action = CmdRunAction(command='cd /workspace && git config --global user.email "agent@example.com" && git config --global user.name "CodeAct Agent" && git config --global core.pager ""')
    action.set_hard_timeout(60)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError(f'Failed to configure git: {str(obs)}')
    
    # Add all files and create initial commit
    action = CmdRunAction(command='cd /workspace && git add -A && git commit -m "Initial commit"')
    action.set_hard_timeout(120)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError(f'Failed to create initial commit: {str(obs)}')
    
    logger.info('-' * 30)
    logger.info('END Runtime Initialization')
    logger.info('-' * 30)


def extract_changes(runtime: Runtime) -> Dict[str, Any]:
    """Extract changes made by the agent in the sandbox using a tarball via base64 stdout."""
    logger.info('-' * 30)
    logger.info('BEGIN Extracting Changes')
    logger.info('-' * 30)
    
    # Stage all changes
    action = CmdRunAction(command='cd /workspace && git add -A')
    action.set_hard_timeout(120)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError(f'Failed to stage changes: {str(obs)}')
    
    # Get git diff
    action = CmdRunAction(command='cd /workspace && git diff --cached')
    action.set_hard_timeout(300)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError(f'Failed to get git diff: {str(obs)}')
    
    git_patch = obs.content.strip()
    logger.info(f'Git patch size: {len(git_patch)} bytes')

    # Create a temporary directory on the host
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory on host: {temp_dir}")
    archive_name = 'workspace_archive.tar.gz'
    container_archive_path = f'/tmp/{archive_name}'
    host_archive_path = os.path.join(temp_dir, archive_name)

    try:
        # Create tarball inside the container
        # Exclude .git directory from the tarball
        tar_command = f'cd /workspace && tar czf {container_archive_path} --exclude=.git .'
        action = CmdRunAction(command=tar_command)
        action.set_hard_timeout(300)  # Increased timeout for tarring
        logger.info(f"Creating tarball in container: {container_archive_path}", extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
            raise RuntimeError(f'Failed to create tarball in container: {str(obs)}')

        # Get base64 encoded tarball content via stdout
        base64_command = f'base64 {container_archive_path}'
        action = CmdRunAction(command=base64_command)
        action.set_hard_timeout(300) # Increased timeout for base64 encoding large files
        logger.info(f"Getting base64 content of tarball: {container_archive_path}", extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(f"Received base64 output (first 100 chars): {obs.content[:100]}...", extra={'msg_type': 'OBSERVATION'})
        if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
            raise RuntimeError(f'Failed to get base64 content of tarball: {str(obs)}')
        base64_content = obs.content.strip()

        # Decode base64 content and write to host archive file
        logger.info(f"Decoding base64 ({len(base64_content)} chars) and writing to {host_archive_path}")
        decoded_bytes = base64.b64decode(base64_content)
        with open(host_archive_path, 'wb') as f:
            f.write(decoded_bytes)
        logger.info(f"Successfully wrote decoded tarball to {host_archive_path}")

        # Extract the tarball on the host
        logger.info(f"Extracting tarball {host_archive_path} to {temp_dir}")
        with tarfile.open(host_archive_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)
        logger.info(f"Successfully extracted tarball to {temp_dir}")
        os.remove(host_archive_path) # Clean up the archive file after extraction

    except Exception as e:
        logger.error(f"Error during tarball creation/base64/extraction: {str(e)}")
        # Clean up host temp dir if extraction failed
        if os.path.exists(temp_dir):
           shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to extract changes via tarball/base64: {str(e)}")
    finally:
        # Clean up tarball inside the container regardless of success/failure
        action = CmdRunAction(command=f'rm -f {container_archive_path}')
        action.set_hard_timeout(60)
        logger.info(f"Cleaning up tarball in container: {container_archive_path}", extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
             logger.warning(f"Failed to remove tarball from container: {str(obs)}")

    logger.info('-' * 30)
    logger.info('END Extracting Changes')
    logger.info('-' * 30)
    
    # temp_dir now contains the extracted files from the workspace
    return {
        'git_patch': git_patch,
        'temp_dir': temp_dir,
    }


def get_instruction() -> str:
    """Create the initial instruction for the agent."""
    instruction = """
I need you to complete the homework assignment located in the /workspace directory.

Please follow these steps:

1. First explore the /workspace directory to understand the structure and available files.
   - Use commands like `ls`, `find`, and `grep` to explore
   - Look for hw.tex or similar files that contain the homework instructions

2. Read the homework instructions (likely in hw.tex or README) to understand what needs to be done.
   - You can use `cat`, `head`, or other commands to read the files

3. Complete the homework by implementing the necessary code, analysis, or other required work.
   - Make sure to write any code files, reports, or other outputs directly in the /workspace directory
   - Install any necessary dependencies using pip if needed
   - Run and test your code to verify it works correctly

4. When you're finished, let me know you've completed the homework.

IMPORTANT - DATA STRUCTURE INFORMATION:
Some data files were not copied into the workspace to save space. Here's information about the data structure:

- KITTI Dataset Structure (not copied):
  - KITTI/poses/ - Contains txt files (00.txt, 01.txt, 02.txt, 03.txt) with vehicle pose data
  - KITTI/odometry/ - Contains odometry data
  - KITTI/calib/ - Contains calibration data for the sensors

- p3/data/images/ Structure (not copied):
  - Contains a 'train' folder with 100 PNG image files (r_0.png through r_99.png)
  - These are images for NeRF training at approximately 400-500KB each

If you need to work with these files, you can simulate their presence and usage in your code.
You can assume the data files are in the expected locations with the expected formats.

Please work directly in the /workspace directory so all your changes are preserved.
"""
    return instruction


def codeact_user_response(
    state: State, agent_output: Optional[str] = None
) -> Optional[str]:
    """Fake user response function for the CodeAct agent."""
    # Mimic how run_infer.py handles the fake user response
    if agent_output and "should I proceed" in agent_output.lower():
        return "Yes, please proceed."
    return None


async def main():
    # Parse command line arguments
    parser = get_parser()
    args, _ = parser.parse_known_args()
    
    # Verify the homework directory exists
    hw_dir = os.path.abspath(args.hw_dir)
    if not os.path.isdir(hw_dir):
        raise ValueError(f"Homework directory does not exist: {hw_dir}")
    
    # Determine output directory
    if args.output_dir:
        output_base_dir = os.path.abspath(args.output_dir)
    else:
        output_base_dir = os.path.dirname(hw_dir)
    
    # Create a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the output directory with timestamp - this will contain all files from Docker
    hw_dir_basename = os.path.basename(hw_dir)
    output_dir = os.path.join(output_base_dir, f"{hw_dir_basename}_{args.agent_cls}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Configure the agent
    config = get_config(args)
    
    # Create and connect to the runtime
    runtime = create_runtime(config)
    await runtime.connect()
    
    # Select the appropriate user response function
    fake_user_response_fn = None
    if args.agent_cls == "CodeActAgent":
        fake_user_response_fn = codeact_user_response
    
    try:
        # Initialize the runtime
        initialize_runtime(runtime, hw_dir)
        
        # Create the initial instruction
        instruction = get_instruction()
        initial_action = MessageAction(content=instruction)
        
        # Run the agent
        logger.info("Running the agent...")
        state = await run_controller(
            config=config,
            initial_user_action=initial_action,
            runtime=runtime,
            fake_user_response_fn=fake_user_response_fn,
        )
        
        if state.last_error:
            logger.error(f"Agent encountered an error: {state.last_error}")
        
        # Extract changes
        logger.info("Extracting changes...")
        changes = extract_changes(runtime)
        
        # Save the git patch to a file
        patch_path = os.path.join(output_dir, "changes.patch")
        with open(patch_path, "w") as f:
            f.write(changes["git_patch"])
        logger.info(f"Saved git patch to {patch_path}")
        
        # Save agent history
        history_path = os.path.join(output_dir, "agent_history.log")
        with open(history_path, "w") as f:
            for event in state.history:
                f.write(f"{event}\n")
                f.write("-" * 80 + "\n")
        logger.info(f"Saved agent history to {history_path}")
        
        # Copy all files from the Docker container to the output directory
        logger.info(f"Copying files from Docker container to {output_dir}...")
        for item in os.listdir(changes["temp_dir"]):
            src = os.path.join(changes["temp_dir"], item)
            dst = os.path.join(output_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                # Ensure parent directory exists before copying file
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        
        logger.info(f"""
==========================================================
Homework processing complete!

Results are available in: {output_dir}

- All files from Docker container (original, modified, and new): {output_dir}/*
- Git patch: {os.path.basename(patch_path)}
- Agent history: {os.path.basename(history_path)}
==========================================================
""")
        
    finally:
        # Close the runtime
        runtime.close()
        
        # Clean up the temporary directory
        if "changes" in locals() and os.path.exists(changes["temp_dir"]):
            shutil.rmtree(changes["temp_dir"])
            logger.info(f"Cleaned up temporary directory")


if __name__ == "__main__":
    # Use call_async_from_sync to run the async main function
    call_async_from_sync(main)
