# Feature Maturity Levels

Every public feature of Primus-Turbo carries one of four maturity levels. The level tells a user how
far to trust the feature: whether its API can change, where it has been tested, and whether it has
been shown to speed up real training. The current level of each feature is in the
[feature index](./README.md).

## Levels

Each level includes the requirements of the levels below it.

| Level | What a user can expect | Requirements |
|---|---|---|
| **Experimental** | It works for the cases its tests cover. The API, behavior, and supported shapes can change in any release, without notice. | Forward and backward unit tests exist. Off by default: a user has to call it or turn it on explicitly. |
| **Beta** | Correct and fast on the architectures it claims. The API can still change, but changes are announced in the release notes. | Tests pass on every architecture the feature claims. Has a benchmark in `benchmark/`, with its command in the Feature Card. Has a [Feature Card](./_template.md). |
| **Stable** | Production use. The API changes only under the deprecation policy below. | Validated end to end: at least one real training run shows a measured training-step gain, recorded in the Feature Card. API reviewed and frozen. May become a default backend. |
| **Deprecated** | Still works, but is going away. | Has a named replacement, and the release notes say when it will be removed. |

A feature can claim fewer architectures or precisions than it runs on. The claim is what the
Feature Card's support matrix says, and only claimed cells count toward the requirements.

## Changing a level

A level changes through a pull request that:

1. updates the level in the Feature Card and in the [feature index](./README.md);
2. lists the requirements of the new level, each with its evidence: test runs per architecture,
   the benchmark entry, and for Stable the training run (model, scale, and step time before and
   after);
3. gets approved by a maintainer.

A feature that stops meeting its requirements, for example because its tests no longer pass on a
claimed architecture, is moved down a level or has the failing claim removed in the next release.

The release notes of each version list every level change in that version.

## Initial levels

The first version of the feature index gave some features a level from how they are used today,
before their evidence was written down. The index marks these as *provisional*. A provisional
feature gets its Feature Card and evidence by the next minor release, or moves down to the level
its evidence supports.

## Deprecation policy

- Deprecating a Beta or Stable feature is announced in the release notes of the version that
  deprecates it, together with the replacement.
- A deprecated feature stays for at least two minor releases after the one that deprecated it
  (deprecated in 0.7 means removed in 0.10 at the earliest).
- Experimental features can be removed without a deprecation period. Their removal is still listed
  in the release notes.
