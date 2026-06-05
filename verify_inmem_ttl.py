"""
verify_inmem_ttl.py — Proves the in-memory Redis fallback honors TTL.

This is the crash-safety guarantee the architecture claims: if an owner agent
dies mid-query, its registry lock must auto-expire so other agents aren't
blocked forever. Previously expire()/ex were no-ops in the in-memory fallback,
silently breaking this guarantee whenever real Redis was unavailable.

No real Redis used here — we instantiate _InMemoryRedis directly.
"""

import asyncio

from blackboard.client import _InMemoryRedis


async def main():
    r = _InMemoryRedis()
    failures = []

    # 1. SETNX claim with a 1s TTL (owner takes the lock).
    ok = await r.set("registry:q1", "owner=A", nx=True, ex=1)
    assert ok is True, "owner should acquire lock"
    print("1. owner A claimed lock (ex=1s): OK")

    # 2. A competing SETNX must fail while the owner is alive.
    ok2 = await r.set("registry:q1", "owner=B", nx=True, ex=1)
    if ok2 is not False:
        failures.append("competing SETNX should fail while lock is live")
    print(f"2. competing claim while live -> acquired={ok2} (expected False)")

    # 3. Heartbeat: owner renews TTL; lock stays alive past original 1s.
    await asyncio.sleep(0.6)
    await r.expire("registry:q1", 1)  # renew for another 1s
    await asyncio.sleep(0.6)          # 1.2s total elapsed, but renewed at 0.6s
    still_there = await r.get("registry:q1")
    if still_there is None:
        failures.append("heartbeat expire() did not renew TTL")
    print(f"3. after heartbeat renew, lock still present: {still_there is not None} (expected True)")

    # 4. Owner 'crashes' (stops renewing). After TTL lapses the lock must vanish.
    await asyncio.sleep(1.1)
    expired = await r.get("registry:q1")
    if expired is not None:
        failures.append("lock did not expire after owner stopped heartbeat")
    print(f"4. after crash (no renew) lock auto-expired: {expired is None} (expected True)")

    # 5. A new agent can now claim the freed slot.
    ok3 = await r.set("registry:q1", "owner=C", nx=True, ex=1)
    if ok3 is not True:
        failures.append("new owner could not claim expired slot")
    print(f"5. new owner C claims freed slot: {ok3} (expected True)")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("ALL CRASH-SAFETY CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
