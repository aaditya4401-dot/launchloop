-- Staging: one row per flight — the randomization knobs (Phase 2 features),
-- the is_weak_motor answer key, and the outcome numbers RocketPy reported during
-- simulation. The sim_* outcomes are kept so we can cross-check them against the
-- values dbt re-derives from the traces downstream.

select
    flight_id,
    is_weak_motor,

    -- continuous randomization knobs (become ML features in Phase 2)
    wind_magnitude,
    wind_u,
    wind_v,
    mass,
    thrust_scale,

    -- outcomes recorded straight from the Flight object at simulation time
    apogee_agl            as sim_apogee_agl,
    max_speed             as sim_max_speed,
    out_of_rail_velocity,

    seed
from {{ source('raw', 'raw_labels') }}
