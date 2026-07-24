"""
Phase 1 — the parametric rocket builder.

`build_flight` constructs a full RocketPy Flight from the four knobs the Monte
Carlo loop randomizes: two wind components, airframe mass, and a thrust-curve
scale factor. Everything else (geometry, drag, recovery, launch rail) is fixed.

All data is self-defined (inline thrust curve + constant drag), so the whole
simulation runs offline — no motor/drag files are ever read from disk or network.
Geometry follows RocketPy's public "Calisto" tutorial rocket.
"""

from rocketpy import Environment, Flight, Rocket, SolidMotor

# Fixed launch-site elevation (m above sea level). Wind is set per-flight.
ELEVATION = 1400.0

# Nominal airframe mass (kg), WITHOUT the motor. The loop jitters this.
NOMINAL_MASS = 14.426

# Base motor thrust curve: [time_s, thrust_N]. Peaks early, then tapers.
# The loop multiplies every thrust value by a per-flight `thrust_scale`.
BASE_THRUST_CURVE = [
    [0.0, 0.0],
    [0.05, 1500.0],
    [0.10, 2000.0],
    [0.20, 2200.0],
    [0.50, 2050.0],
    [1.00, 1950.0],
    [2.00, 1720.0],
    [3.00, 1450.0],
    [3.80, 1000.0],
    [3.90, 0.0],
]
BURN_TIME = 3.9

# Total impulse of the base thrust curve (N·s) = area under it (trapezoid rule).
# Phase 3's motor menu scales this curve to hit other class impulses.
BASE_TOTAL_IMPULSE = sum(
    0.5 * (BASE_THRUST_CURVE[i + 1][0] - BASE_THRUST_CURVE[i][0])
    * (BASE_THRUST_CURVE[i + 1][1] + BASE_THRUST_CURVE[i][1])
    for i in range(len(BASE_THRUST_CURVE) - 1)
)

# Fixed launch-rail setup (shared by the loop and the dispersion study).
RAIL_LENGTH = 5.0
INCLINATION = 85.0
HEADING = 0.0


def build_motor_for_impulse(total_impulse: float) -> SolidMotor:
    """Build a motor whose total impulse equals `total_impulse` (N·s),
    by scaling the base M1670 thrust curve. Used by the Phase 3 motor menu."""
    return build_motor(thrust_scale=total_impulse / BASE_TOTAL_IMPULSE)


def build_motor(thrust_scale: float = 1.0) -> SolidMotor:
    """Solid motor with its whole thrust curve scaled by `thrust_scale`."""
    scaled_curve = [[t, thrust * thrust_scale] for t, thrust in BASE_THRUST_CURVE]
    return SolidMotor(
        thrust_source=scaled_curve,
        burn_time=BURN_TIME,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def build_environment(wind_u: float = 0.0, wind_v: float = 0.0) -> Environment:
    """Fixed-elevation launch site with a constant custom wind (offline)."""
    env = Environment(latitude=32.99, longitude=-106.97, elevation=ELEVATION)
    env.set_atmospheric_model(type="custom_atmosphere", wind_u=wind_u, wind_v=wind_v)
    return env


# Motor mount position on the airframe (used by both loop and dispersion).
MOTOR_POSITION = -1.255

# Base airframe properties (no ballast). The base center of mass is the
# coordinate-system origin (0), so nominal center_of_mass_without_motor = 0.
BASE_LATERAL_INERTIA = 6.321
BASE_AXIAL_INERTIA = 0.034
# Where Phase 3 ballast is placed (m, forward toward the nose at +1.278).
# Forward ballast both adds mass (lowers apogee) and moves CG forward (raises
# the stability margin) — the core performance-vs-safety tradeoff.
BALLAST_POSITION = 1.0
NOMINAL_MAIN_CD_S = 10.0  # nominal main-parachute drag area (cd * S)


def build_rocket(
    mass: float,
    motor: SolidMotor,
    ballast_mass: float = 0.0,
    main_cd_s: float = NOMINAL_MAIN_CD_S,
) -> Rocket:
    """Airframe with the given mass and motor, plus nose/fins/tail/parachutes.

    ballast_mass : extra mass (kg) placed forward at BALLAST_POSITION. Increases
                   total mass and shifts the CG forward (Phase 3 knob). Default 0
                   leaves Phase 1 behavior unchanged.
    main_cd_s    : main-parachute drag area (cd * S); larger -> slower descent ->
                   more wind drift. Default is the Phase 1 nominal.
    """
    # Fold ballast (a point mass at BALLAST_POSITION) into the airframe's mass,
    # center of mass, and lateral inertia via the parallel-axis theorem.
    total_mass = mass + ballast_mass
    com = (mass * 0.0 + ballast_mass * BALLAST_POSITION) / total_mass
    lateral_inertia = (
        BASE_LATERAL_INERTIA
        + mass * (0.0 - com) ** 2
        + ballast_mass * (BALLAST_POSITION - com) ** 2
    )

    rocket = Rocket(
        radius=127 / 2000,
        mass=total_mass,
        inertia=(lateral_inertia, lateral_inertia, BASE_AXIAL_INERTIA),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=com,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=MOTOR_POSITION)
    rocket.add_nose(length=0.55829, kind="von karman", position=1.278)
    rocket.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        position=-1.04956,
        cant_angle=0.5,
    )
    rocket.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )
    rocket.add_parachute(name="drogue", cd_s=1.0, trigger="apogee")
    rocket.add_parachute(name="main", cd_s=main_cd_s, trigger=800)
    return rocket


def build_flight(
    wind_u: float,
    wind_v: float,
    mass: float,
    thrust_scale: float,
) -> Flight:
    """Build a complete Flight for the given per-flight knob values.

    wind_u, wind_v : constant wind components (m/s), east and north.
    mass           : airframe mass without motor (kg).
    thrust_scale   : multiplier applied to the whole base thrust curve.
    """
    env = build_environment(wind_u, wind_v)
    motor = build_motor(thrust_scale)
    rocket = build_rocket(mass, motor)
    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=RAIL_LENGTH,
        inclination=INCLINATION,
        heading=HEADING,
    )
