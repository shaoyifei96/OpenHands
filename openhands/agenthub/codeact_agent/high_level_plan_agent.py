import os

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.action import Action, CmdRunAction, MessageAction
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory
from openhands.utils.prompt import PromptManager


class HighLevelPlanAgent(CodeActAgent):
    """
    HighLevelPlanAgent acts as a root agent that handles high-level planning for complex tasks.
    This agent doesn't perform coding tasks directly but breaks problems into smaller parts
    and delegates them to CodeActAgent instances for implementation.

    It maintains a high-level view of the overall task, coordinates between subtasks,
    and integrates results from delegate agents.
    """

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config, is_delegate=False)
        # Always ensure delegation is enabled for this agent
        self.tools = codeact_function_calling.get_tools(
            codeact_enable_browsing=False,
            codeact_enable_jupyter=False,  # Disable Jupyter for high-level agent
            codeact_enable_llm_editor=False,  # Disable direct code editing
            codeact_enable_delegate=True,  # Always enable delegation
            enable_code_edit_tool=False,
            llm=self.llm,
        )
        print(
            f"TOOLS!!!!!!! loaded for HighLevelPlanAgent: {', '.join([tool.get('function').get('name') for tool in self.tools])}"
        )

        # Override prompt manager to use the high-level planning prompt
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts')
        )
        self.prompt_manager.system_template = self.prompt_manager._load_template(
            'high_level_plan_prompt'
        )

        # Create a ConversationMemory instance with the new prompt manager
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

    def handle_child_completion(self, state: State) -> list[Action]:
        """Handle the completion of a child agent's task.

        This method:
        1. Reviews the changes made by the child agent
        2. Checks the success status
        3. Makes a decision on how to proceed

        Args:
            state: The current state containing child agent's completion info

        Returns:
            list[Action]: List of actions to take based on the review
        """
        actions = []

        # Get the last observation which should contain child completion info
        last_observation = state.get_last_observation()
        if not last_observation:
            return actions

        # Extract child agent completion info
        if hasattr(last_observation, 'outputs'):
            outputs = last_observation.outputs
            self.current_child_status = outputs.get('status')
            self.current_child_branch = outputs.get('result_branch')
            self.current_child_message = outputs.get('message')

            # Review changes if we have a branch name
            if self.current_child_branch:
                # First checkout the child's branch to review changes
                checkout_action = CmdRunAction(
                    command=f'git checkout {self.current_child_branch}',
                    thought="Checking out child agent's branch to review changes.",
                    is_input=False,
                )
                actions.append(checkout_action)

                # Get diff of changes
                diff_action = CmdRunAction(
                    command='git diff HEAD~1',
                    thought='Reviewing changes made by child agent.',
                    is_input=False,
                )
                actions.append(diff_action)

                # Add a think action to analyze the changes
                review_thought = f"Reviewing child agent's work:\nStatus: {self.current_child_status}\nMessage: {self.current_child_message}"
                think_action = AgentThinkAction(thought=review_thought)
                actions.append(think_action)

                # Based on status, decide next steps
                if self.current_child_status == 'success':
                    # If successful, merge changes back to master
                    merge_action = CmdRunAction(
                        command='git checkout master && git merge --no-ff {self.current_child_branch} -m "Merge successful child agent changes"',
                        thought='Merging successful child agent changes back to master.',
                        is_input=False,
                    )
                    actions.append(merge_action)
                else:
                    # If failed, create a new branch for retry
                    retry_branch = f'{self.current_child_branch}_retry'
                    retry_action = CmdRunAction(
                        command=f'git checkout -b {retry_branch}',
                        thought='Creating new branch for retry after child agent failure.',
                        is_input=False,
                    )
                    actions.append(retry_action)

                    # Add a message to inform about the failure
                    message_action = MessageAction(
                        content=f'Child agent task failed. Status: {self.current_child_status}\nMessage: {self.current_child_message}\nCreated new branch {retry_branch} for retry.',
                        wait_for_response=True,
                    )
                    actions.append(message_action)

        return actions

    # def step(self, state: State) -> Action:
    #     """Override the step method to handle child agent completion."""
    #     # Check if we have a child completion to handle
    #     if state.get_last_observation() and hasattr(state.get_last_observation(), 'outputs'):
    #         child_actions = self.handle_child_completion(state)
    #         if child_actions:
    #             return child_actions[0]

    #     # If no child completion to handle, proceed with normal step
    #     return super().step(state)
