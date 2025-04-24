import os
from collections import deque

from litellm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import Action, AgentDelegateAction, AgentFinishAction, ProgressParentAgentAction
from openhands.events.action.commands import CmdRunAction
from openhands.events.action.message import MessageAction
from openhands.events.event import EventSource
from openhands.events.tool import ToolCallMetadata
from openhands.llm.llm import LLM
from openhands.memory.condenser import Condenser
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


# Create a minimal valid ModelResponse for manual actions
def create_dummy_model_response(tool_call_id: str, function_name: str, content: str) -> ModelResponse:
    return ModelResponse(
        id='dummy_response_id',
        choices=[
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': [
                        {
                            'id': tool_call_id,
                            'type': 'function',
                            'function': {'name': function_name, 'arguments': '{}'},
                        }
                    ],
                },
                'finish_reason': 'tool_calls',
            }
        ],
        model='dummy_model',
        usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    )


class CodeActAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        is_delegate: bool = False,
        master_progress: int = 0,
        delegate_count: int = 0,
    ) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config)
        self.pending_actions: deque[Action] = deque()
        self.is_delegate = is_delegate
        self.is_plan_agent = not bool(is_delegate)
        self.git_init = True
        self.branch_init = False
        self.checkedout_main = False
        self.reset()

        # Retrieve the enabled tools
        self.tools = codeact_function_calling.get_tools(
            codeact_enable_browsing=self.config.codeact_enable_browsing,
            codeact_enable_jupyter=self.config.codeact_enable_jupyter,
            codeact_enable_llm_editor=self.config.codeact_enable_llm_editor,
            codeact_enable_delegate=not self.is_delegate,  # only delegate if not a delegate
            llm=self.llm,
        )
        logger.debug(
            f"TOOLS loaded for CodeActAgent: {', '.join([tool.get('function').get('name') for tool in self.tools])}"
        )
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
        )

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        self.master_branch_base_name = 'openhands_master'
        # self.pending_actions.append(
        if self.is_plan_agent:
            self.git_init = False
            self.master_progress = 0
            # self.plan_agent_feature_name = None
            self.num_cur_delegates = 0
        else:
            self.child_branch_base_name = 'child'
            self.master_progress = master_progress
            # self.plan_agent_stage = None
            self.child_agent_id = delegate_count

    def create_branch_name(self) -> str:
        if self.is_plan_agent:
            return self.master_branch_base_name + '_' + str(self.master_progress)
        else:
            return self.master_branch_base_name + '_' + str(self.master_progress) + self.child_branch_base_name + '_' + str(self.child_agent_id)

    def reset(self) -> None:
        """Resets the CodeAct Agent."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> Action:
        """Performs one step using the CodeAct Agent.
        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        # Inject a commit before delegating
        if not self.git_init:
            # Create a unique tool call ID for this manual action
            dummy_tool_id = 'git_init_action'

            # Create action with proper tool call metadata
            init_git_action = CmdRunAction(
                command='git init',
                thought='Initializing git for the current task.',
                is_input=False,  # This decides if inputting to running process
            )

            # Add required tool call metadata with a valid ModelResponse
            model_response = create_dummy_model_response(dummy_tool_id, 'run_cmd', "Initializing git for the current task.")
            init_git_action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=dummy_tool_id,
                function_name='run_cmd',
                model_response=model_response,
                total_calls_in_response=1,
            )

            self.pending_actions.append(init_git_action)
            self.git_init = True

        if not self.branch_init:
            branch_name = self.create_branch_name()
            # Create a unique tool call ID for this manual action
            dummy_tool_id = 'branch_init_action'

            new_branch_and_commit_action = CmdRunAction(
                command='git checkout -b '
                + branch_name
                + ' && git add . && git commit --allow-empty -m "Auto-commit"',
                thought='Creating a new branch for the current task.',
                is_input=False,  # This decides if inputting to running process
            )

            # Add required tool call metadata with a valid ModelResponse
            model_response = create_dummy_model_response(dummy_tool_id, 'run_cmd', "Creating a new branch for the current task.")
            new_branch_and_commit_action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=dummy_tool_id,
                function_name='run_cmd',
                model_response=model_response,
                total_calls_in_response=1,
            )

            self.pending_actions.append(new_branch_and_commit_action)
            self.branch_init = True

        if self.is_plan_agent and not self.checkedout_main:
            # check out to master branch
            dummy_tool_id = 'check_out_to_master_action'
            check_out_to_master_action = CmdRunAction(
                command='git checkout ' + self.create_branch_name(),
                thought='Checking out to master branch.',
                is_input=False,
            )
            model_response = create_dummy_model_response(dummy_tool_id, 'run_cmd', "Checking out to master branch.")
            check_out_to_master_action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=dummy_tool_id,
                function_name='run_cmd',
                model_response=model_response,
                total_calls_in_response=1,
            )
            self.checkedout_main = True
            self.pending_actions.append(check_out_to_master_action)
            
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        # Check if this is a HighLevelPlanAgent
        print("Is Plan Agent: ", self.is_plan_agent)
        # Print total number of messages
        print(f"Total number of messages: {len(messages)}")
        
        # Check for empty messages or content
        for i, msg in enumerate(messages):
            if not msg.content:
                print(f"\033[91mWARNING: Empty content in message {i} with role {msg.role}\033[0m")
                continue
                
            for j, content in enumerate(msg.content):
                if not hasattr(content, 'text') or not content.text:
                    print(f"\033[91mWARNING: Empty text in message {i}, content {j}, type {content.type}, role {msg.role}\033[0m")
                else:
                    text_preview = content.text[:60] if len(content.text) > 60 else content.text
                    print(f"Message {i} ({msg.role}), content {j}, type: {content.type}, preview: '{text_preview}', length: {len(content.text)}")
        
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        params['tools'] = self.tools
        # log to litellm proxy if possible
        params['extra_body'] = {'metadata': state.to_llm_metadata(agent_name=self.name)}
        response = self.llm.completion(**params)
        if self.is_plan_agent:
            actions = codeact_function_calling.response_to_actions(
                response,
                self.is_delegate,
                self.num_cur_delegates,  # Pass delegate count
                self.master_progress # Pass master progress to delegate
            )
        else:
            actions = codeact_function_calling.response_to_actions(
                response,
                self.is_delegate,
            )
        # Increment delegate count if any delegate actions were created
        for action in actions:
            if isinstance(action, AgentDelegateAction):
                self.num_cur_delegates += 1
            if isinstance(action, ProgressParentAgentAction) and self.is_plan_agent:
                # Increment the master progress counter when the plan agent uses the progress tool
                self.master_progress += 1
                self.branch_init = False  # Force creation of a new branch
                self.checkedout_main = False  # Force checkout to new branch
                # print(f"\033[92mMaster progress incremented to: {self.master_progress}\033[0m")
                
                # # Create a commit action to preserve current state
                dummy_tool_id = 'progress_commit_action'
                commit_action = CmdRunAction(
                    command='git add . && git commit --allow-empty -m "Auto-commit before progressing to next stage"',
                    thought=f'Committing changes before progressing to stage {self.master_progress}.',
                    is_input=False,
                )
                model_response = create_dummy_model_response(
                    dummy_tool_id, 
                    'run_cmd', 
                    f"Committing changes before progressing to stage {self.master_progress}."
                )
                commit_action.tool_call_metadata = ToolCallMetadata(
                    tool_call_id=dummy_tool_id,
                    function_name='run_cmd',
                    model_response=model_response,
                    total_calls_in_response=1,
                )
                # user_message = MessageAction(
                #     content="continue",             # The message text
                #     wait_for_response=False,         # Wait for agent to respond
                #     source=EventSource.USER         # Set the source as USER
                # )
                self.pending_actions.append(commit_action)  # Add at the beginning
                # self.pending_actions.append(user_message)
            else:
                self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_messages(self, state: State) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Initializes with system prompt and optional initial user message
        2. Processes events (Actions and Observations) into messages
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            state (State): The current state object containing conversation history and other metadata

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt
                - Initial user message (if configured)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process initial messages
        messages = self.conversation_memory.process_initial_messages(
            with_caching=self.llm.is_caching_prompt_active()
        )

        # Condense the events from the state.
        events = self.condenser.condensed_history(state)

        logger.debug(
            f'Processing {len(events)} events from a total of {len(state.history)} events'
        )

        # Use ConversationMemory to process events
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_messages=messages,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        messages = self._enhance_messages(messages)

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _enhance_messages(self, messages: list[Message]) -> list[Message]:
        """Enhances the user message with additional context based on keywords matched.

        Args:
            messages (list[Message]): The list of messages to enhance

        Returns:
            list[Message]: The enhanced list of messages
        """
        assert self.prompt_manager, 'Prompt Manager not instantiated.'

        results: list[Message] = []
        is_first_message_handled = False
        prev_role = None

        for msg in messages:
            if msg.role == 'user' and not is_first_message_handled:
                is_first_message_handled = True
                # compose the first user message with examples
                self.prompt_manager.add_examples_to_initial_message(msg)

            elif msg.role == 'user':
                # Add double newline between consecutive user messages
                if prev_role == 'user' and len(msg.content) > 0:
                    # Find the first TextContent in the message to add newlines
                    for content_item in msg.content:
                        if isinstance(content_item, TextContent):
                            # If the previous message was also from a user, prepend two newlines to ensure separation
                            content_item.text = '\n\n' + content_item.text
                            break

            results.append(msg)
            prev_role = msg.role

        return results
