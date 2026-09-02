# Project context

This document is a compact orientation brief for anyone continuing the Team
RiDE Ackermann-steering reinforcement-learning project. The binding technical
status and handoff order are in `BRIAN_HANDOFF.md`.

## Project objective

Develop a defensible robotics-workshop study of how action-space inductive bias
affects reinforcement learning and closed-loop simulation-to-real transfer on
the F1TENTH car Frank.

## Current status

- There is no authorized main physical experiment.
- The historical 120-run Frank protocol is paused. It replays offline command
  sequences and does not run a trained policy closed loop.
- The ROS 1 safety, logging, configuration-capture, localization, and
  engineering-pilot infrastructure should be retained and reused.
- Earlier stateful-decoder and action-interface ensemble directions failed
  their prospective gates. Their negative results must remain part of the
  project record.
- Direct-policy seeds 0 and 4 are the only checkpoints in the handoff archive
  that have passed the documented simulator competence screen. They are not
  authorized for live execution on Frank.

## Working principles

- Read `BRIAN_HANDOFF.md`, the two negative-result reports, the Rung 2 report,
  and the August 26 lab-session record before planning new work.
- Treat committed code, frozen protocols, and durable reports as authoritative
  over informal summaries.
- Do not compare action spaces using incompetent policies.
- Do not select checkpoints using test-track or physical outcomes.
- Separate engineering qualification from binding scientific experiments.
- Freeze hypotheses, competence gates, selection rules, and physical protocols
  before observing their binding outcomes.

## Smallest next milestone

Finish the outstanding general Frank qualification checks, then implement a
zero-motion shadow path that converts live robot observations into the exact
simulator observation contract, runs a checkpoint, decodes its action, and
records the proposed and realized commands without authorizing motion. This is
the minimum end-to-end evidence needed before considering a closed-loop pilot.

For large artifacts and background planning material, use the Drive index:
<https://docs.google.com/document/d/1ONnym9thjiRQZhfAKamTnqvwLU9wGF9-HCeULk6soI8/edit>
