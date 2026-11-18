from anthropic.types import TextBlockParam, ToolResultBlockParam

from agentdojo.agent_pipeline.llms.anthropic_llm import _merge_tool_result_messages


def test_merge_tool_result_messages():
    messages = [
        dict(role="user", content=[TextBlockParam(type="text", text="Hi!")]),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_1",
                    content=[TextBlockParam(type="text", text="output_1")],
                    is_error=False,
                )
            ],
        ),
        dict(
            role="assistant",
            content="Hello",
        ),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_2",
                    content=[TextBlockParam(type="text", text="output_2")],
                    is_error=False,
                )
            ],
        ),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_3",
                    content=[TextBlockParam(type="text", text="output_3")],
                    is_error=False,
                )
            ],
        ),
        dict(role="user", content=[TextBlockParam(type="text", text="Bye")]),
        dict(
            role="assistant",
            content="Hello",
        ),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_4",
                    content=[TextBlockParam(type="text", text="output_4")],
                    is_error=False,
                )
            ],
        ),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_5",
                    content=[TextBlockParam(type="text", text="output_5")],
                    is_error=False,
                )
            ],
        ),
    ]
    expected_messages = [
        dict(
            role="user",
            content=[
                TextBlockParam(type="text", text="Hi!"),
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_1",
                    content=[TextBlockParam(type="text", text="output_1")],
                    is_error=False,
                ),
            ],
        ),
        dict(
            role="assistant",
            content="Hello",
        ),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_2",
                    content=[TextBlockParam(type="text", text="output_2")],
                    is_error=False,
                ),
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_3",
                    content=[TextBlockParam(type="text", text="output_3")],
                    is_error=False,
                ),
                TextBlockParam(type="text", text="Bye"),
            ],
        ),
        dict(
            role="assistant",
            content="Hello",
        ),
        dict(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_4",
                    content=[TextBlockParam(type="text", text="output_4")],
                    is_error=False,
                ),
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id="tool_use_id_5",
                    content=[TextBlockParam(type="text", text="output_5")],
                    is_error=False,
                ),
            ],
        ),
    ]

    merged_messages = _merge_tool_result_messages(messages)
    assert merged_messages == expected_messages
