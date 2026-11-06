"""How Reachy describes itself to Muse: its model and device guide when the SDK takes them."""

import dataclasses

from reachy_muse.contract.commands import COMMAND_SPECS
from reachy_muse.contract.guide import MAX_GUIDE_BYTES, REACHY_GUIDE
from reachy_muse.registration import reachy_description


@dataclasses.dataclass(frozen=True)
class GuidedDescription:
    node_id: str
    display_name: str
    version: str
    commands: dict
    model_id: str = "linux"
    model_name: str = "Linux"
    guide: str | None = None


def test_reachy_registers_its_model_and_guide_when_the_sdk_takes_them():
    description = reachy_description("homelink-abcdef", GuidedDescription)
    assert (description.model_id, description.model_name, description.guide) == (
        "pollen-reachy-mini", "Reachy Mini", REACHY_GUIDE)
    assert description.commands is COMMAND_SPECS


def test_an_sdk_without_device_guides_registers_only_the_commands():
    from musegadget.link_client import DeviceDescription

    fields = {field.name for field in dataclasses.fields(DeviceDescription)}
    description = reachy_description("homelink-abcdef")
    assert description.node_id == "homelink-abcdef"
    assert ("guide" in fields) == (getattr(description, "guide", None) == REACHY_GUIDE)


def test_the_guide_fits_the_device_guide_limit_and_names_the_commands_muse_may_call():
    assert MAX_GUIDE_BYTES == 4096
    assert len(REACHY_GUIDE.encode("utf-8")) <= MAX_GUIDE_BYTES
    assert REACHY_GUIDE.startswith("# Reachy Mini\n")
    for command in ("`reachy.expression`", "`reachy.move`", "`nod`", "`shake`"):
        assert command in REACHY_GUIDE
