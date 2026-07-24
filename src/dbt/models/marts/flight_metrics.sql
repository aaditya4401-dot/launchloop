-- Mart (intermediate): collapse each flight's time-series into per-flight metrics
-- using real SQL aggregation over the traces. This is the step that turns
-- ~1.1M trace rows into 1,000 rows and independently re-derives apogee etc.
-- from the raw data (rather than trusting the simulation's reported numbers).

with traces as (
    select * from {{ ref('stg_traces') }}
)

select
    flight_id,
    max(altitude_agl)                as apogee_m,            -- highest point reached
    max(vertical_velocity)           as max_ascent_rate_ms,  -- fastest climb
    arg_max(time_s, altitude_agl)    as apogee_time_s,       -- time at that apogee
    max(time_s)                      as flight_duration_s,   -- last timestamp
    count(*)                         as n_samples
from traces
group by flight_id
