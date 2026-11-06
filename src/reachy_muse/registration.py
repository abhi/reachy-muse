# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""How Reachy describes itself when it registers with Muse."""

from __future__ import annotations

import dataclasses

from musegadget.link_client import DeviceDescription

from reachy_muse import __version__
from reachy_muse.contract.commands import COMMAND_SPECS
from reachy_muse.contract.guide import MODEL_ID, MODEL_NAME, REACHY_GUIDE


def reachy_description(node_id: str, description=DeviceDescription) -> DeviceDescription:
    """Reachy's commands, plus its model and device guide when the Muse SDK can register them."""
    fields = {field.name for field in dataclasses.fields(description)}
    extra = {"model_id": MODEL_ID, "model_name": MODEL_NAME, "guide": REACHY_GUIDE}
    return description(node_id, "Reachy Mini", __version__, COMMAND_SPECS,
                       **{key: value for key, value in extra.items() if key in fields})
