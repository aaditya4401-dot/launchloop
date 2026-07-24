-- Staging: the per-timestamp flight traces, lightly cleaned.
-- One row per (flight_id, time). No aggregation here — that happens downstream.

select
    flight_id,
    time              as time_s,
    altitude_agl,
    vertical_velocity,
    vertical_accel
from {{ source('raw', 'raw_traces') }}
