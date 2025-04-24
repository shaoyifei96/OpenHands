from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_PROGRESS_PARENT_DESCRIPTION = """Progresses the parent agent to the next stage of execution.

Use this tool when a major phase of the task has been completed and you need to progress to the next stage.
This will increment the master progress counter and create a new branch for the next phase of work.
"""

ProgressParentAgentTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='progress_parent_agent',
        description=_PROGRESS_PARENT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['reason'],
            'properties': {
                'reason': {
                    'type': 'string',
                    'description': 'The reason for progressing to the next stage',
                },
            },
        },
    ),
) 