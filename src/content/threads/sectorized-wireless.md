---
title: Sectorized wireless networks
pillar: network-control
order: 1
summary: Capacity and stability when nodes reshape the topology with directional sectors.
media:
  type: video
  src: /media/sectorization.mp4
  poster: /media/sectorization-poster.jpg
  caption: How re-sectorizing the nodes changes the effective network.
  duration: "0:59"
---

Sectorization lets a node activate several directional antennas at once, each covering a sector, which reshapes the network topology itself. We model a multi-hop mesh of sectorized nodes together with the interference the sectors create, and characterize its capacity region through an auxiliary graph that depends on how every node is sectorized.

Choosing the sectorization is then an optimization problem: we define the sectorization gain, which measures how far a given network flow can be scaled by sectorizing the nodes, and we give a distributed algorithm that finds a near-optimal sectorization. We also propose a class of sectorizations, even homogeneous sectorizations, that makes the auxiliary graph bipartite. Routing then becomes faster and the capacity region larger at the same time, and the more the network is sectorized, the stronger both effects. When sectors can be reconfigured over time, the network keeps this advantage while staying stable under unknown traffic.
