"""
Tests for the .ork exporter (src/agents/export_ork.py). No RocketPy flight
sims here (fast) -- just building a motor object and generating/validating XML.
"""

import math
import zipfile
from io import BytesIO
from xml.etree import ElementTree as ET

import pytest

from src.agents.export_ork import build_ork_bytes
from src.agents.oracle import Config, Metrics

MISSION = {"wind_u": 6.0, "wind_v": 0.0}


def _fake_metrics() -> Metrics:
    return Metrics(apogee_m=3000.0, stability_margin_cal=2.0,
                   rail_exit_velocity_ms=30.0, landing_downrange_m=1200.0,
                   landing_dispersion_m=300.0)


def test_ork_is_a_valid_zip_with_well_formed_xml():
    config = Config(motor_total_impulse=6000.0)
    data = build_ork_bytes(config, _fake_metrics(), MISSION)

    zf = zipfile.ZipFile(BytesIO(data))
    assert zf.testzip() is None  # no corrupt members
    names = zf.namelist()
    assert "rocket.ork" in names
    assert any(n.startswith("thrustcurves/") and n.endswith(".rse") for n in names)

    # Both embedded files must be well-formed XML (raises on failure).
    for name in names:
        ET.fromstring(zf.read(name))


def test_ballast_appears_only_when_nonzero():
    with_ballast = build_ork_bytes(
        Config(motor_total_impulse=6000.0, ballast_mass=1.5), _fake_metrics(), MISSION)
    without_ballast = build_ork_bytes(
        Config(motor_total_impulse=6000.0, ballast_mass=0.0), _fake_metrics(), MISSION)

    zf1 = zipfile.ZipFile(BytesIO(with_ballast))
    zf2 = zipfile.ZipFile(BytesIO(without_ballast))
    assert "<masscomponent>" in zf1.read("rocket.ork").decode()
    assert "<masscomponent>" not in zf2.read("rocket.ork").decode()


def test_rail_angle_and_wind_converted_correctly():
    # rail_inclination is deg-from-horizontal (90=vertical); OpenRocket's
    # launchrodangle is radians-from-vertical (0=vertical) -> 90 - inclination.
    config = Config(motor_total_impulse=6000.0, rail_inclination=80.0)
    mission = {"wind_u": 4.0, "wind_v": 3.0}
    data = build_ork_bytes(config, _fake_metrics(), mission)
    xml = zipfile.ZipFile(BytesIO(data)).read("rocket.ork").decode()

    # abs tolerance: values round-trip through 5-decimal string formatting.
    angle_rad = float(ET.fromstring(xml).find(".//launchrodangle").text)
    assert math.degrees(angle_rad) == pytest.approx(10.0, abs=1e-3)

    wind_avg = float(ET.fromstring(xml).find(".//windaverage").text)
    assert wind_avg == pytest.approx(5.0, abs=1e-3)  # hypot(4, 3)


def test_motor_curve_matches_the_menu_impulse():
    """The embedded .rse thrust curve must integrate to the same total impulse
    as the motor the design loop actually verified -- not an approximation."""
    config = Config(motor_total_impulse=6000.0)
    data = build_ork_bytes(config, _fake_metrics(), MISSION)
    zf = zipfile.ZipFile(BytesIO(data))
    rse_name = next(n for n in zf.namelist() if n.endswith(".rse"))
    rse = ET.fromstring(zf.read(rse_name))
    engine = rse.find(".//engine")
    assert float(engine.get("Itot")) == pytest.approx(6000.0)
    assert float(engine.get("burn-time")) == pytest.approx(3.9)
