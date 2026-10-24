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

A quantum circuit is written without reference to the hardware that will run it, and compilers close that gap. One kind rewrites the circuit in the gate set that the machine supports. The other turns those gates into operations on the physical device. A single compiler could optimize both jointly, but this work focuses on the second kind, whose job is twofold. Mapping assigns logical qubits to physical qubits across the processors, anticipating which gates will have to act across a link. Routing then brings qubits together where the gates need them, generating entanglement between processors, scheduling gate and qubit teleportations and inserting swaps to move qubits within and across processors, all under link fidelity, decoherence and congestion constraints.

With Cisco Systems I built compilers for both gate-based and measurement-based distributed execution, including a reinforcement learning compiler that minimizes expected execution time and a genetic approach that minimizes the number of gate and qubit teleportations. The demo below showcases the modeling environment proposed in the ICC 2025 paper, on a two-processor system.
