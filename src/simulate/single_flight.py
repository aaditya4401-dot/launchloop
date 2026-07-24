"""
Phase 1 — Step 1: define ONE rocket that flies a sensible flight.

This is an exploration script (run it directly) — not the final pipeline entry
point. Its whole job is to prove we can build a working Flight fully offline and
get a believable apogee. Once this looks right, we generalize it into the
Monte Carlo loop.

Build order mirrors RocketPy's model:  Environment -> Motor -> Rocket -> Flight.

All numbers are self-defined (no downloaded motor/drag files), so the whole
simulation is offline. The airframe/motor geometry follows RocketPy's public
"Calisto" tutorial rocket; the thrust curve and drag are our own inline values.
"""

from rocketpy import Environment, SolidMotor, Rocket, Flight

# ---------------------------------------------------------------------------
# 1) ENVIRONMENT — the sky the rocket flies through.
#    We use a fixed launch-site elevation and a *custom* atmosphere so nothing
#    is ever fetched from the network. wind_u = west->east wind, wind_v =
#    south->north wind, both in m/s. Here: a light, steady breeze.
# ---------------------------------------------------------------------------
env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
env.set_atmospheric_model(
    type="custom_atmosphere",
    wind_u=5,   # 5 m/s wind toward the east
    wind_v=0,
)

# ---------------------------------------------------------------------------
# 2) MOTOR — a solid rocket motor. thrust_source is our own [time_s, thrust_N]
#    curve (peaks early, then tapers), so no .eng file is needed. The geometry
#    args describe the solid propellant grains.
# ---------------------------------------------------------------------------
thrust_curve = [
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

motor = SolidMotor(
    thrust_source=thrust_curve,
    burn_time=3.9,
    dry_mass=1.815,                       # motor casing mass (no propellant)
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

# ---------------------------------------------------------------------------
# 3) ROCKET — the airframe. Drag is a constant coefficient (offline-friendly).
#    Then we bolt on nose cone, fins, tail, and two parachutes for recovery.
# ---------------------------------------------------------------------------
rocket = Rocket(
    radius=127 / 2000,                    # 63.5 mm radius
    mass=14.426,                          # airframe mass WITHOUT motor
    inertia=(6.321, 6.321, 0.034),
    power_off_drag=0.5,                   # constant drag coefficient (Cd)
    power_on_drag=0.5,
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

rocket.add_motor(motor, position=-1.255)

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
    top_radius=0.0635,
    bottom_radius=0.0435,
    length=0.060,
    position=-1.194656,
)

# Recovery: drogue at apogee, main lower down (triggers on the way back down).
rocket.add_parachute(
    name="drogue",
    cd_s=1.0,
    trigger="apogee",
)
rocket.add_parachute(
    name="main",
    cd_s=10.0,
    trigger=800,          # deploy at 800 m above ground level, descending
)

# ---------------------------------------------------------------------------
# 4) FLIGHT — fly the rocket off a 5 m launch rail, tilted 5 deg from vertical.
# ---------------------------------------------------------------------------
flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=5,
    inclination=85,       # 85 deg = 5 deg off straight up
    heading=0,
)

# ---------------------------------------------------------------------------
# Report the headline numbers.
# ---------------------------------------------------------------------------
print("=" * 55)
print(f"Apogee (above ground) : {flight.apogee - env.elevation:,.1f} m")
print(f"Apogee (above sea lvl): {flight.apogee:,.1f} m")
print(f"Apogee time           : {flight.apogee_time:,.2f} s")
print(f"Max speed             : {flight.max_speed:,.1f} m/s")
print(f"Max Mach              : {flight.max_mach_number:,.2f}")
print(f"Rail departure speed  : {flight.out_of_rail_velocity:,.1f} m/s")
print(f"Impact/landing speed  : {flight.impact_velocity:,.1f} m/s")
print("=" * 55)
