---
title: Entanglement distillation
pillar: quantum-networks
order: 2
summary: When to wait for a distillation outcome and when to continue, while stored pairs decohere.
media:
  type: demo
  src: /media/distillation-demo/distillation-demo.js
  tag: distillation-demo
  frame: false
  caption: "The same link scheduled twice: on the left by a baseline that always or never waits, or by you, and on the right by the learned policy."
---

The entangled pairs that a link produces are noisy. Entanglement distillation consumes several of them to obtain one of higher fidelity. It proceeds in steps that can fail, and each step is confirmed by a classical message. The stored pairs decohere while this message crosses the network. After every step the nodes therefore choose between waiting for the message and continuing at the risk that the step has failed.

This thread treats distillation as an online scheduling problem. The nodes decide step by step which pairs to distill and whether to wait, and the goal is the highest rate of pairs that meet a required fidelity. We derived an upper bound on the throughput of any policy, formulated the problem as a Markov decision process, and trained a reinforcement learning policy that works for any number of memories. Where always waiting or never waiting is the right choice, the learned policy matches it, and in the regimes between the two it outperforms both.
