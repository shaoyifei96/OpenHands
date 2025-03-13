from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_DELEGATE_DESCRIPTION = """Delegates the current task or conversation to another agent, use if you are unsure about the next steps.

Use this tool when you need to code something
"""

DelegateToAgentTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='delegate_to_agent',
        description=_DELEGATE_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['task'],
            'properties': {
                'task': {
                    'type': 'string',
                    'description': 'The task to delegate to the other agent',
                },
            },
        },
    ),
)
