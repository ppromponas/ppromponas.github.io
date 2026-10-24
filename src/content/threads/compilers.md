---
title: Compilers
pillar: distributed-quantum-computing
order: 1
summary: From a circuit to physical operations across several processors.
media:
  type: demo
  src: /media/dqc-compiler.js
  caption: "Interactive: the modeling environment proposed in the paper, executing a circuit across two processors. Play it, step through it, or scrub the timeline."
---

A quantum circuit is written independently of the hardware that will run it. Two compilation steps bring it to a machine, and they may be one and the same: the first rewrites the circuit in the gate set the machine supports, and the second translates those gates into physical operations, mapping logical qubits to physical ones, inserting swaps, and so on. On a distributed system the second step also has to generate entanglement between processors, schedule gate and qubit teleportations and route qubits across the links, all under link fidelity, decoherence and congestion constraints.

With Cisco Systems I built compilers for both gate-based and measurement-based distributed execution, including a reinforcement learning compiler that minimizes expected execution time and a genetic approach that minimizes the number of gate and qubit teleportations. The demo below showcases the modeling environment proposed in the ICC 2025 paper, on a two-processor system.
