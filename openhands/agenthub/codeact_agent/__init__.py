from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.high_level_plan_agent import HighLevelPlanAgent
from openhands.controller.agent import Agent

__all__ = ['CodeActAgent', 'HighLevelPlanAgent']

# Register agents
Agent.register('CodeActAgent', CodeActAgent)
Agent.register('HighLevelPlanAgent', HighLevelPlanAgent)
