###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest

import primus_turbo.common.envs as envs


@pytest.mark.parametrize("raw, expected", [(None, None), ("", None), ("  ", None), (" AITER ", " AITER ")])
def test_backend_spec(monkeypatch, raw, expected):
    name = "PRIMUS_TURBO_GEMM_BACKEND"
    monkeypatch.delenv(name, raising=False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    assert envs.backend_spec(name)() == expected


@pytest.mark.parametrize("raw, expected", [(None, 3), ("", 3), ("2", 2), (" 7 ", 7)])
def test_positive_integer_default_and_value(monkeypatch, raw, expected):
    name = "PT_MEGA_FP8_COMBINE_TUNE_REPS"
    monkeypatch.delenv(name, raising=False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    assert getattr(envs, name) == expected


@pytest.mark.parametrize("raw", ["0", "-2", "1.5", "invalid"])
def test_positive_integer_rejects_invalid_value(monkeypatch, raw):
    name = "PT_MEGA_FP8_COMBINE_TUNE_REPS"
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        getattr(envs, name)


@pytest.mark.parametrize("raw, expected", [("16,32 48", (16, 32, 48)), ("8,,  16", (8, 16))])
def test_candidate_list(monkeypatch, raw, expected):
    monkeypatch.setenv("PT_MEGA_FP8_COMBINE_CU_CANDIDATES", raw)
    assert envs.PT_MEGA_FP8_COMBINE_CU_CANDIDATES == expected


@pytest.mark.parametrize("raw", [" , ", "0,16", "-1", "a,16"])
def test_candidate_list_rejects_invalid_value(monkeypatch, raw):
    name = "PT_MEGA_FP8_COMBINE_CU_CANDIDATES"
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        getattr(envs, name)


@pytest.mark.parametrize(
    "raw, expected", [(None, 256), ("", 32), ("junk", 32), ("1", 32), ("64", 64), ("2048", 1024)]
)
def test_clamped_integer(monkeypatch, raw, expected):
    name = "PRIMUS_TURBO_ODC_GDA_BLOCK"
    monkeypatch.delenv(name, raising=False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    assert getattr(envs, name) == expected


@pytest.mark.parametrize("raw, expected", [(None, 300), ("", 300), ("bad", 300), ("60", 60)])
def test_deepep_timeout(monkeypatch, raw, expected):
    name = "PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT"
    monkeypatch.delenv(name, raising=False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    assert getattr(envs, name) == expected


def test_cache_invalidation_and_unset(monkeypatch):
    name = "PRIMUS_TURBO_AUTO_TUNE"
    monkeypatch.setenv(name, "0")
    assert not getattr(envs, name)
    monkeypatch.setenv(name, "1")
    assert not getattr(envs, name)
    envs.reset_cache(name)
    assert getattr(envs, name)
    envs.set_env(name, "0")
    assert envs.is_set(name) and not getattr(envs, name)
    envs.set_env(name, None)
    assert not envs.is_set(name) and not getattr(envs, name)


def test_disable_and_reenable_cache(monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "0")
    try:
        envs.disable_envs_cache()
        assert not envs.PRIMUS_TURBO_AUTO_TUNE
        monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "1")
        assert envs.PRIMUS_TURBO_AUTO_TUNE
        envs.enable_envs_cache()
        assert envs.PRIMUS_TURBO_AUTO_TUNE
        monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "0")
        assert envs.PRIMUS_TURBO_AUTO_TUNE
    finally:
        envs.enable_envs_cache()
        envs.reset_cache()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: envs.UNDECLARED,
        lambda: envs.is_set("UNDECLARED"),
        lambda: envs.set_env("UNDECLARED", "1"),
        lambda: envs.reset_cache("UNDECLARED"),
    ],
)
def test_unknown_variable_is_rejected(operation):
    with pytest.raises(AttributeError, match="UNDECLARED"):
        operation()


def test_directory_exposes_declared_variables():
    assert set(dir(envs)) == set(envs.environment_variables)
