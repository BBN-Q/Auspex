# Status inspection

`auspex_status.py` queries both local services without connecting to hardware,
changing settings, acquiring leases, or performing recovery.

```sh
python utils/auspex_status.py
python utils/auspex_status.py --json
```

Use `--instrument-socket PATH` and `--experiment-socket PATH` when not using the
shared defaults. Use the same `AUSPEX_HOME` as the services.

The snapshot shows reachability, inventory/ownership blockers, active-run progress,
result folder, and degraded persistence information. Exit code 0 means both
services were reachable, **not** that hardware is safe, all devices are free, or a
measurement succeeded. If either service is unavailable, exit code is nonzero.

## When something is wrong

1. Inspect the run state/reason and its `execution.log`; returned arrays and every
   intermediate calculation are not automatically retained there.
2. Check active/orphaned ownership and UNKNOWN/TERMINATED instrument states.
3. Inspect the physical setup before changing software ownership or reconnecting.
4. Use explicit [reconciliation/recovery](../components/instruments.md#ownership-and-recovery)
   when justified. Do not erase uncertain state with `--clean` simply to get past
   an error.

A stale progress timestamp means no new reported progress, not proof of a hung
worker. An in-flight instrument call or Python calculation may delay cancellation.
A published dataset can coexist with failed teardown or finalization; check both
data and terminal status.

The status tool is a snapshot, not a live monitor or durable audit history.
[Implementation](https://github.com/BBN-Q/Auspex/blob/auspex-v2/utils/auspex_status.py)
and [utility notes](https://github.com/BBN-Q/Auspex/blob/auspex-v2/utils/README.md)
remain in the repository.
