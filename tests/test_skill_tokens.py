"""Skill attachments: one compose rule and one display record for both clients."""

from __future__ import annotations

from dizel_core import skills as skills_core


def test_compose_prompt_puts_skills_before_text():
    assert skills_core.compose_prompt(["First skill.", "Second skill."], "do this") == (
        "First skill.\n\nSecond skill.\n\ndo this"
    )


def test_compose_prompt_without_typed_text_returns_the_skills():
    assert skills_core.compose_prompt(["Only skill."], "  ") == "Only skill."


def test_compose_prompt_without_skills_returns_the_text():
    assert skills_core.compose_prompt([], "just text") == "just text"


def test_compose_prompt_strips_each_part():
    assert skills_core.compose_prompt(["  padded  "], " text ") == "padded\n\ntext"


def test_display_metadata_is_none_without_skills():
    assert skills_core.display_metadata("hi", []) is None


def test_display_metadata_records_text_and_skill_names():
    meta = skills_core.display_metadata(
        "hi",
        [
            {"name": "outline", "title": "Outline", "prompt": "long body"},
            {"name": "email", "title": "Email", "prompt": "another body"},
        ],
    )
    assert meta == {
        "text": "hi",
        "skills": [
            {"name": "outline", "title": "Outline"},
            {"name": "email", "title": "Email"},
        ],
    }


def test_display_metadata_keeps_empty_text():
    meta = skills_core.display_metadata("", [{"name": "outline", "title": "Outline"}])
    assert meta == {"text": "", "skills": [{"name": "outline", "title": "Outline"}]}


def test_display_metadata_keeps_skill_args_for_editing():
    meta = skills_core.display_metadata(
        "hi", [{"name": "outline", "title": "Outline", "args": "a topic"}]
    )
    assert meta["skills"] == [{"name": "outline", "title": "Outline", "args": "a topic"}]


# ── ChatManager: the model sees content; only the session sees display ─────


def manager():
    from dizel_core.chat_manager import ChatManager

    instance = ChatManager()
    instance.history = []
    return instance


def test_user_turn_records_display_and_keeps_model_content():
    chat = manager()
    display = {"text": "hi", "skills": [{"name": "outline", "title": "Outline"}]}

    chat._record_user_turn("expanded body\n\nhi", display=display)

    assert chat.history[-1]["content"] == "expanded body\n\nhi"
    assert chat.history[-1]["display"] == display


def test_user_turn_without_display_stays_plain():
    chat = manager()

    chat._record_user_turn("hello", attachments=["pasted"])

    assert chat.history[-1] == {
        "role": "user",
        "content": "hello",
        "attachments": ["pasted"],
    }


def test_api_messages_do_not_send_display_to_providers():
    chat = manager()
    chat.system_prompt = ""
    chat.history = [
        {
            "role": "user",
            "content": "expanded body\n\nhi",
            "display": {"text": "hi", "skills": [{"name": "outline", "title": "Outline"}]},
        },
        {"role": "assistant", "content": "ok"},
    ]

    assert chat._api_messages() == [
        {"role": "user", "content": "expanded body\n\nhi"},
        {"role": "assistant", "content": "ok"},
    ]
