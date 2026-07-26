"""
Phase 3 — export a verified design as an OpenRocket (.ork) file.

OpenRocket is the industry-standard GUI tool the hobby/collegiate rocketry
community actually uses to view, present, and further inspect a design. This
lets a design the closed-loop designer verified be opened there directly.

Format -- verified against REAL OpenRocket source, not guessed: downloaded and
unzipped actual example .ork files from github.com/openrocket/openrocket, and
cross-checked the custom-motor format against OpenRocket's own wiki. Key facts:
  - .ork is a ZIP containing `rocket.ork` (the real XML, same name as the outer
    file) built as nested <subcomponents> tags: <nosecone>, <bodytube> (holding
    <trapezoidfinset>, <transition> for the tail, <masscomponent> for ballast,
    <parachute>, <innertube> for the motor mount).
  - Launch conditions (rail length/angle, wind) live separately, under
    <simulations><simulation><conditions>.
  - A motor not in OpenRocket's built-in database is loaded from an embedded
    `thrustcurves/<digest>.rse` file (RockSim engine XML) inside the same zip.
    We always embed our own curve here: our motor menu is synthetic (scaled
    from one base curve, not real commercial motors), so embedding is the only
    way the export reproduces EXACTLY what was simulated, rather than
    approximating a match to an unrelated real motor.

Units: rocket.ork uses SI (meters, kilograms, radians). The .rse motor file
uses RockSim's traditional units (millimeters, grams, Newtons, seconds) --
verified against a real OpenRocket wiki example, not assumed.

Honesty note on mass: OpenRocket computes each component's mass from its own
geometry x material density, same as a real CAD tool. Our own simulation never
modeled the airframe at that level of detail -- it uses one lumped NOMINAL_MASS
for the whole structure, so we override the body tube's computed mass to match
it (see _build_rocket_xml). The two numbers that most matter for confirming
this is the same design the agents verified -- the motor thrust curve and the
ballast mass -- are both exact; the airframe's internal mass distribution is a
reasonable approximation, not a physical breakdown we ever modeled ourselves.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import zipfile
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import typer

from src.agents.designer import LAST_DESIGN_PATH
from src.agents.mission import MOTOR_MENU, Mission
from src.agents.oracle import Config, Metrics
from src.simulate.rocket import (
    BALLAST_POSITION,
    FIN_CANT_ANGLE,
    FIN_COUNT,
    FIN_POSITION,
    FIN_ROOT_CHORD,
    FIN_SPAN,
    FIN_TIP_CHORD,
    MOTOR_POSITION,
    NOMINAL_MASS,
    NOSE_LENGTH,
    NOSE_POSITION,
    RADIUS,
    RAIL_LENGTH,
    TAIL_BOTTOM_RADIUS,
    TAIL_LENGTH,
    TAIL_POSITION,
    TAIL_TOP_RADIUS,
    build_motor_for_impulse,
)

MATERIAL = dict(type="bulk", density="1850.0", group="Composites")  # fiberglass-ish


def _motor_name(impulse: float) -> str:
    return next((m.name for m in MOTOR_MENU if m.total_impulse == impulse),
                f"custom{impulse:.0f}")


def _sub(parent, tag, text=None, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = str(text)
    return el


def _pretty(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def _cd_s_to_diameter(cd_s: float, assumed_cd: float = 0.8) -> float:
    """RocketPy models a parachute as one lumped cd*S drag area; OpenRocket
    wants a physical diameter + cd separately. Assume a typical round-canopy
    cd and back out the diameter that reproduces the same drag area."""
    area = cd_s / assumed_cd
    return math.sqrt(4 * area / math.pi)


def _build_rse_xml(motor, motor_name: str) -> str:
    """Sample the REAL RocketPy motor object's thrust/mass/cg and write a
    RockSim engine (.rse) file -- schema verified against an OpenRocket wiki
    example. Units here are RockSim's own: mm, g, N, s."""
    burn_time = motor.burn_time[1]
    n_samples = 40
    times = [burn_time * i / n_samples for i in range(n_samples + 1)]

    propellant_mass_kg = float(motor.propellant_initial_mass)
    total_mass_0_kg = float(motor.total_mass(0))
    total_impulse = motor.total_impulse
    isp_s = total_impulse / (propellant_mass_kg * 9.80665)
    # Grain stack length as a stand-in for overall motor length/diameter --
    # our motor has no separate "case" dimensions of its own.
    motor_length_m = (motor.grain_number * motor.grain_initial_height
                      + (motor.grain_number - 1) * motor.grain_separation)
    motor_diameter_m = 2 * motor.nozzle_radius

    db = ET.Element("engine-database")
    elist = ET.SubElement(db, "engine-list")
    engine = ET.SubElement(elist, "engine", {
        "mfg": "LaunchLoop", "code": motor_name, "Type": "reloadable",
        "dia": f"{motor_diameter_m * 1000:.1f}",
        "len": f"{motor_length_m * 1000:.1f}",
        "delays": "0",
        "burn-time": f"{burn_time:.3f}",
        "initWt": f"{total_mass_0_kg * 1000:.1f}",
        "propWt": f"{propellant_mass_kg * 1000:.1f}",
        "Itot": f"{total_impulse:.1f}",
        "avgThrust": f"{total_impulse / burn_time:.1f}",
        "peakThrust": f"{max(float(motor.thrust(t)) for t in times):.1f}",
        "Isp": f"{isp_s:.1f}",
        "auto-calc-cg": "0", "auto-calc-mass": "0",
        "massFrac": f"{100 * propellant_mass_kg / total_mass_0_kg:.1f}",
        "throatDia": "0.", "exitDia": "0.",
        "FDiv": "10", "FFix": "1", "FStep": "-1.",
        "cgDiv": "10", "cgFix": "1", "cgStep": "-1.",
        "mDiv": "10", "mFix": "1", "mStep": "-1.",
    })
    data = ET.SubElement(engine, "data")
    for t in times:
        mass_g = float(motor.total_mass(t)) * 1000
        # RockSim's cg is measured from the motor's forward (leading) end;
        # RocketPy's is from the nozzle -- flip via (length - rocketpy_cg).
        cg_mm = (motor_length_m - float(motor.center_of_mass(t))) * 1000
        ET.SubElement(data, "eng-data", {
            "t": f"{t:.4f}", "f": f"{float(motor.thrust(t)):.2f}",
            "m": f"{mass_g:.3f}", "cg": f"{cg_mm:.2f}",
        })
    return _pretty(db)


def _build_rocket_xml(config: Config, mission: Mission, digest: str,
                      motor_name: str, motor) -> str:
    """Build rocket.ork's XML: our fixed airframe geometry (from rocket.py)
    plus the agents' chosen ballast/chute/motor/rail-angle."""
    root = ET.Element("openrocket", version="1.10",
                      creator="LaunchLoop closed-loop flight designer")
    rocket = ET.SubElement(root, "rocket")
    _sub(rocket, "name", f"LaunchLoop design ({motor_name})")
    _sub(rocket, "designer", "LaunchLoop (github.com/aaditya4401-dot/launchloop)")
    _sub(rocket, "referencetype", "maximum")

    configid = "00000000-0000-0000-0000-000000000001"
    mc = ET.SubElement(rocket, "motorconfiguration", configid=configid, default="true")
    ET.SubElement(mc, "stage", number="0", active="true")

    subs = ET.SubElement(rocket, "subcomponents")
    stage = ET.SubElement(subs, "stage")
    _sub(stage, "name", "Sustainer")
    body_subs_parent = ET.SubElement(stage, "subcomponents")

    # --- nose cone --- RocketPy's "von karman" = Haack series, shape param 0
    # (verified: HAACK_SERIES with C=0 is Von Karman, per OpenRocket's own docs).
    nose = ET.SubElement(body_subs_parent, "nosecone")
    _sub(nose, "name", "Nose cone")
    _sub(nose, "material", "Fiberglass", **MATERIAL)
    _sub(nose, "length", f"{NOSE_LENGTH:.5f}")
    _sub(nose, "thickness", "0.003")
    _sub(nose, "shape", "haack")
    _sub(nose, "shapeparameter", "0.0")
    _sub(nose, "aftradius", f"{RADIUS:.5f}")

    # --- body tube --- physical length spans nose attachment to tail's aft
    # end, derived from our own simulated component positions in rocket.py.
    body_length = NOSE_POSITION - (TAIL_POSITION - TAIL_LENGTH)

    def from_top(rocketpy_position: float) -> float:
        """Convert a RocketPy absolute axial position to OpenRocket's
        position-from-the-body-tube's-top-edge convention."""
        return NOSE_POSITION - rocketpy_position

    body = ET.SubElement(body_subs_parent, "bodytube")
    _sub(body, "name", "Body tube")
    _sub(body, "material", "Fiberglass", **MATERIAL)
    _sub(body, "length", f"{body_length:.5f}")
    _sub(body, "thickness", "0.003")
    _sub(body, "radius", f"auto {RADIUS:.5f}")
    # Force the tube's own computed mass to match our simulation's lumped dry
    # airframe mass (see module docstring's honesty note on mass).
    _sub(body, "overridemass", f"{NOMINAL_MASS:.4f}")
    _sub(body, "overridesubcomponentsmass", "false")

    body_subs = ET.SubElement(body, "subcomponents")

    # --- ballast: a real point mass, exactly the agents' chosen value ---
    if config.ballast_mass > 0:
        offset = from_top(BALLAST_POSITION)
        ballast = ET.SubElement(body_subs, "masscomponent")
        _sub(ballast, "name", "Ballast (agent-selected)")
        _sub(ballast, "axialoffset", f"{offset:.5f}", method="top")
        _sub(ballast, "position", f"{offset:.5f}", type="top")
        _sub(ballast, "mass", f"{config.ballast_mass:.4f}")
        _sub(ballast, "masscomponenttype", "masscomponent")

    # --- main parachute: cd*S -> physical diameter under an assumed cd ---
    chute = ET.SubElement(body_subs, "parachute")
    _sub(chute, "name", "Main parachute")
    _sub(chute, "axialoffset", "0.1", method="top")
    _sub(chute, "position", "0.1", type="top")
    _sub(chute, "cd", "0.8")
    _sub(chute, "diameter", f"{_cd_s_to_diameter(config.parachute_cd_s):.4f}")
    _sub(chute, "deployevent", "apogee")
    _sub(chute, "deployaltitude", "200.0")
    _sub(chute, "deploydelay", "0.0")

    # --- fins ---
    fins = ET.SubElement(body_subs, "trapezoidfinset")
    _sub(fins, "name", "Fins")
    _sub(fins, "fincount", FIN_COUNT)
    offset = from_top(FIN_POSITION)
    _sub(fins, "axialoffset", f"{offset:.5f}", method="top")
    _sub(fins, "position", f"{offset:.5f}", type="top")
    _sub(fins, "material", "Fiberglass", **MATERIAL)
    _sub(fins, "thickness", "0.004")
    _sub(fins, "cant", f"{FIN_CANT_ANGLE:.2f}")
    _sub(fins, "rootchord", f"{FIN_ROOT_CHORD:.5f}")
    _sub(fins, "tipchord", f"{FIN_TIP_CHORD:.5f}")
    # RocketPy's add_trapezoidal_fins call doesn't set an explicit sweep -- a
    # symmetric taper is a reasonable visual default, not a verified value.
    _sub(fins, "sweeplength", f"{(FIN_ROOT_CHORD - FIN_TIP_CHORD) / 2:.5f}")
    _sub(fins, "height", f"{FIN_SPAN:.5f}")

    # --- tail (a boat-tail; OpenRocket calls this a "transition") ---
    tail = ET.SubElement(body_subs, "transition")
    _sub(tail, "name", "Tail")
    offset = from_top(TAIL_POSITION)
    _sub(tail, "axialoffset", f"{offset:.5f}", method="top")
    _sub(tail, "position", f"{offset:.5f}", type="top")
    _sub(tail, "material", "Fiberglass", **MATERIAL)
    _sub(tail, "thickness", "0.003")
    _sub(tail, "shape", "conical")
    _sub(tail, "foreradius", f"{TAIL_TOP_RADIUS:.5f}")
    _sub(tail, "aftradius", f"{TAIL_BOTTOM_RADIUS:.5f}")
    _sub(tail, "length", f"{TAIL_LENGTH:.5f}")

    # --- motor mount (inner tube housing the embedded custom motor) ---
    inner = ET.SubElement(body_subs, "innertube")
    _sub(inner, "name", "Motor mount")
    offset = from_top(MOTOR_POSITION)
    _sub(inner, "axialoffset", f"{offset:.5f}", method="top")
    _sub(inner, "position", f"{offset:.5f}", type="top")
    _sub(inner, "material", "Fiberglass", **MATERIAL)
    motor_len = motor.grain_number * motor.grain_initial_height
    _sub(inner, "length", f"{motor_len:.5f}")
    _sub(inner, "radius", f"{motor.nozzle_radius:.5f}")
    _sub(inner, "thickness", "0.002")

    mount = ET.SubElement(inner, "motormount")
    _sub(mount, "ignitionevent", "automatic")
    _sub(mount, "ignitiondelay", "0.0")
    _sub(mount, "overhang", "0.0")
    motor_el = ET.SubElement(mount, "motor", configid=configid)
    _sub(motor_el, "type", "single")
    _sub(motor_el, "manufacturer", "LaunchLoop")
    _sub(motor_el, "digest", digest)
    _sub(motor_el, "designation", motor_name)
    _sub(motor_el, "diameter", f"{2 * motor.nozzle_radius:.5f}")
    _sub(motor_el, "length", f"{motor_len:.5f}")
    _sub(motor_el, "delay", "0")
    ic = ET.SubElement(mount, "ignitionconfiguration", configid=configid)
    _sub(ic, "ignitionevent", "automatic")
    _sub(ic, "ignitiondelay", "0.0")

    # --- simulation conditions: rail length/angle/heading + wind ---
    # our rail_inclination is deg-from-horizontal (90=vertical); OpenRocket's
    # launchrodangle is radians-from-vertical (0=vertical).
    launchrodangle = math.radians(90.0 - config.rail_inclination)
    wind_speed = math.hypot(mission["wind_u"], mission["wind_v"])
    wind_dir = math.atan2(mission["wind_v"], mission["wind_u"]) if wind_speed > 0 else 0.0

    sims = ET.SubElement(root, "simulations")
    sim = ET.SubElement(sims, "simulation", status="uptodate")
    _sub(sim, "name", "LaunchLoop verified flight")
    _sub(sim, "simulator", "RK4Simulator")
    _sub(sim, "calculator", "BarrowmanCalculator")
    cond = ET.SubElement(sim, "conditions")
    _sub(cond, "configid", configid)
    _sub(cond, "launchrodlength", f"{RAIL_LENGTH:.2f}")
    _sub(cond, "launchrodangle", f"{launchrodangle:.5f}")
    _sub(cond, "launchroddirection", "90.0")
    _sub(cond, "windaverage", f"{wind_speed:.3f}")
    _sub(cond, "winddirection", f"{wind_dir:.5f}")
    _sub(cond, "launchaltitude", "0.0")

    return _pretty(root)


def build_ork_bytes(config: Config, metrics: Metrics, mission: dict) -> bytes:
    """Build a complete, self-contained .ork file (as bytes) for a verified
    design: the exact motor curve embedded, ballast/chute/rail matching the
    agents' chosen config. `mission` is the plain dict form (wind_u/wind_v).
    See module docstring for the airframe-mass approximation note."""
    motor_name = _motor_name(config.motor_total_impulse)
    motor = build_motor_for_impulse(config.motor_total_impulse)
    digest = f"launchloop-{motor_name}"

    rocket_xml = _build_rocket_xml(config, mission, digest, motor_name, motor)
    rse_xml = _build_rse_xml(motor, motor_name)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("rocket.ork", rocket_xml)
        zf.writestr(f"thrustcurves/{digest}.rse", rse_xml)
    return buf.getvalue()


app = typer.Typer(add_completion=False, help="Export the last verified design as an .ork file.")


@app.command()
def main(
    out: str = typer.Option("launchloop_design.ork", help="Output .ork file path."),
    force: bool = typer.Option(False, help="Export even if the last design run was a NO_GO."),
):
    """Read data/last_design.json (written by `make design`) and export it."""
    if not LAST_DESIGN_PATH.exists():
        raise SystemExit(
            f"{LAST_DESIGN_PATH} not found — run `make design` first, "
            "then `make export-ork`."
        )
    record = json.loads(LAST_DESIGN_PATH.read_text())
    if record["verdict"] != "go" and not force:
        raise SystemExit(
            f"Last design run ended in {record['verdict'].upper()}, not GO — "
            "not exporting an unverified design.\n"
            "  Rationale: " + record["rationale"] + "\n"
            "  Re-run `make design` to get a GO, or pass --force to export anyway."
        )

    config = Config(**record["config"])
    metrics = Metrics(**record["metrics"])
    data = build_ork_bytes(config, metrics, record["mission"])
    Path(out).write_bytes(data)
    print(f"Wrote {out} ({len(data):,} bytes) — verdict was {record['verdict'].upper()}")
    print("Open it in OpenRocket (openrocket.info) to inspect the verified design.")


if __name__ == "__main__":
    app()
