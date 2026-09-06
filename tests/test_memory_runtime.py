from src.memory.runtime import memory_enabled, memory_store_path


def test_memory_enabled_defaults_on_for_application_config():
    assert memory_enabled({"configurable": {}}) is True


def test_memory_enabled_can_be_disabled_per_invocation():
    config = {"configurable": {"memory_enabled": False}}
    assert memory_enabled(config) is False


def test_memory_store_path_is_scoped_per_invocation():
    config = {"configurable": {"memory_store_path": "/tmp/phase4-memory"}}
    assert memory_store_path(config) == "/tmp/phase4-memory"
