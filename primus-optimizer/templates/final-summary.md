# Optimization Session Summary -- {session_id}

**Hardware**: {hw}
**Duration**: {elapsed}
**Workers**: {num_workers} ({num_completed} completed, {num_failed} failed)

## Results Overview

| Worker | Status | Rounds | Baseline TFLOPS | Final TFLOPS | Geomean Gain | Best Round |
|--------|--------|--------|----------------|-------------|-------------|-----------|
{worker_rows}

## Cross-Pollination Discoveries

{cross_pollination_findings}

## Backend Dispatch Recommendations

{dispatch_recommendations}

## Remaining Optimization Opportunities

{remaining_opportunities}

## Merge Plan

See: `merge-plan.md`

## PR Descriptions

{pr_description_links}
