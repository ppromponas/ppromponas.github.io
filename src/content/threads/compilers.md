---
title: Compilers
pillar: distributed-quantum-computing
order: 1
summary: From a circuit to physical operations across several processors.
media:
  type: demo
  src: /media/dqc-compiler.js
  caption: "Interactive: an online greedy compiler executing a circuit across two processors. Play it, step through it, or scrub the timeline."
---

A circuit written for one machine has to be rewritten for several. The compiler first maps logical qubits to physical qubits across processors, anticipating which gates will have to act across a link. It then routes: generating entanglement between processors, scheduling gate and qubit teleportations, and inserting swaps to move qubits within and across processors, all under link fidelity, decoherence and congestion constraints.

With Cisco Systems I built compilers for both gate-based and measurement-based distributed execution, including a reinforcement learning compiler that minimizes expected execution time and a genetic approach that minimizes the number of gate and qubit teleportations. The demo below runs the compiler's own engine in the browser on a two-processor system.
