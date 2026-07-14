"""Tests for the configuration loader and env-var overrides."""


def test_config_attribute_and_dict_access():
    from fraud_platform.config import CONFIG

    assert CONFIG.kafka.input_topic == CONFIG["kafka"]["input_topic"]
    assert CONFIG.get_path("kafka.input_topic") == CONFIG.kafka.input_topic
    assert CONFIG.get_path("does.not.exist", "fallback") == "fallback"


def test_env_override(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker-1:9092")
    monkeypatch.setenv("SERVING_PORT", "9999")

    import fraud_platform.config as config_module

    config_module.load_config.cache_clear()
    cfg = config_module.load_config()

    assert cfg.kafka.bootstrap_servers == "broker-1:9092"
    # Port coerced to int.
    assert cfg.serving.port == 9999
    assert isinstance(cfg.serving.port, int)

    # Reset the cached singleton for other tests.
    config_module.load_config.cache_clear()
