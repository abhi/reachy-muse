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


def test_owner_place_reaches_guides_and_command_only_sdks_without_mutating_defaults():
    description = reachy_description("homelink-abcdef", GuidedDescription, place="office")
    assert description.guide.endswith('\nOwner-configured place: "office".\n')
    assert description.commands["reachy.presence"]["description"].endswith('Owner-configured place: "office".')
    assert "Owner-configured place" not in COMMAND_SPECS["reachy.presence"]["description"]
    legacy = reachy_description("homelink-abcdef", place="office")
    assert legacy.commands["reachy.presence"]["description"].endswith('Owner-configured place: "office".')
    assert len(reachy_description("homelink-abcdef", GuidedDescription, place="x" * 64).guide.encode()) <= 4096


def test_guide_explains_presence_first_delivery_and_recipient_limit():
    assert "preferred way to reach the user" in REACHY_GUIDE
    assert "check `reachy.presence`, then call `reachy.notify`" in REACHY_GUIDE
    assert "delivery goes to whoever is there" in REACHY_GUIDE
    assert "These commands grant no additional permissions." in REACHY_GUIDE
