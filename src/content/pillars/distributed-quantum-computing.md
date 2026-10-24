---
title: Distributed quantum computing
order: 2
lead: Running one quantum circuit across several processors joined by quantum links.
illustration: dqc
---

A single quantum processor will not hold the qubits that large applications need, so several processors are linked over quantum links and run as one machine. Executing a circuit on such a system needs a compiler that maps logical qubits to physical ones across processors and then routes them, generating entanglement, scheduling teleportations and inserting swaps, under link fidelity, decoherence and congestion constraints.
