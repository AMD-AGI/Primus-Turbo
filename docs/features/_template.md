# <Feature name>

> **Maturity:** Experimental | Beta | Stable | Deprecated ([what this means](./maturity.md))
>
> **Since:** v0.x.0

<!--
Copy this file to docs/features/<feature>.md and fill in every section. Keep claims checkable:
every number names its hardware, software, and shape, and every "supported" names its tests.
Link to existing design docs instead of copying them.
-->

## Summary

One paragraph: what the feature does, which training workload it is for, and what it replaces
(for example "instead of an all-to-all followed by a grouped GEMM").

## Support matrix

Where the feature runs. The level of each cell is in the matching rows of
[Supported Operators](../supported_ops.md); link them, and keep the two in sync.

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| <precision or mode> | ✅ / — | ✅ / — | ✅ / — |

Frameworks: PyTorch / JAX.

## Usage

The smallest example that runs it, or a link to the section of [Examples](../examples.md) that has one.

## Configuration

Backends it can run on and how to choose one, and the environment variables that affect it, each
linked to [Environment variables](../env.md).

## Performance

Headline numbers, each with the device, software stack, shape, and how it was timed, and the command
that reproduces them.

## End-to-end impact

Training runs that use the feature: model, parallelism, scale, and step time with and without it.
Required for Stable.

## Known limitations

Shapes, layouts, modes, or environments it does not handle, and what to use instead.

## Further reading

Design docs, tests, and benchmarks.
