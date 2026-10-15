from pixelchat.chat import ChatSession


def test_new_prompt():
    session = ChatSession()
    result = session.handle("a cat on a sofa")
    assert result == {"type": "new", "prompt": "a cat on a sofa"}
    assert session.last_prompt == "a cat on a sofa"


def test_edit_combines_with_previous_prompt():
    session = ChatSession()
    session.handle("a cat on a sofa")
    result = session.handle("make it orange")
    assert result["type"] == "edit"
    assert result["prompt"] == "a cat on a sofa, orange"
    assert session.last_prompt == "a cat on a sofa, orange"


def test_lone_color_word_is_edit():
    session = ChatSession()
    session.handle("a cat on a sofa")
    result = session.handle("blue")
    assert result["type"] == "edit"
    assert result["prompt"] == "a cat on a sofa, blue"


def test_edit_without_previous_prompt_is_treated_as_new():
    session = ChatSession()
    result = session.handle("make it orange")
    assert result["type"] == "new"
    assert result["prompt"] == "make it orange"


def test_chained_edits_accumulate():
    session = ChatSession()
    session.handle("a cat on a sofa")
    session.handle("make it orange")
    result = session.handle("now at night")
    assert result["prompt"] == "a cat on a sofa, orange, at night"


def test_new_command_resets_context():
    session = ChatSession()
    session.handle("a cat on a sofa")
    result = session.handle("/new")
    assert result == {"type": "reset", "prompt": None}
    assert session.last_prompt is None

    # after reset, an edit-shaped message is treated as new
    result = session.handle("make it orange")
    assert result["type"] == "new"


def test_help_command():
    session = ChatSession()
    result = session.handle("/help")
    assert result == {"type": "help", "prompt": None}


def test_empty_message():
    session = ChatSession()
    result = session.handle("   ")
    assert result == {"type": "empty", "prompt": None}
    assert session.last_prompt is None
