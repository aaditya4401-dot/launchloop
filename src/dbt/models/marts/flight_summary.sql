-- Mart (final): the Phase 1 deliverable — ONE row per flight with apogee,
-- max speed, the is_weak_motor label, and the input knobs. This is the table
-- Phase 2's models and Phase 3's agents read from.
--
-- apogee_m is re-derived from the traces (flight_metrics); max_speed_ms comes
-- from the simulation outcome (the traces only carry vertical velocity, so true
-- 3D speed lives in the labels). apogee_gap_m surfaces any disagreement between
-- the trace-derived apogee and the simulation's reported apogee (a data check).

with metrics as (
    select * from {{ ref('flight_metrics') }}
),

labels as (
    select * from {{ ref('stg_labels') }}
)

select
    m.flight_id,

    -- headline outcomes
    round(m.apogee_m, 1)                          as apogee_m,
    round(l.sim_max_speed, 1)                     as max_speed_ms,
    round(m.max_ascent_rate_ms, 1)               as max_ascent_rate_ms,
    round(m.apogee_time_s, 2)                     as apogee_time_s,
    round(m.flight_duration_s, 1)                as flight_duration_s,

    -- the answer key
    l.is_weak_motor,

    -- input knobs (Phase 2 features)
    round(l.wind_magnitude, 2)                    as wind_magnitude_ms,
    round(l.mass, 2)                              as mass_kg,
    round(l.thrust_scale, 3)                      as thrust_scale,
    round(l.out_of_rail_velocity, 1)             as rail_exit_ms,

    -- data-quality check: trace-derived apogee vs simulation-reported apogee
    round(m.apogee_m - l.sim_apogee_agl, 2)       as apogee_gap_m,

    l.seed
from metrics m
join labels l using (flight_id)
order by m.flight_id
