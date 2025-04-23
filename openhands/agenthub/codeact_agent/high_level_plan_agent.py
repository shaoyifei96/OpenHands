import os

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
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
