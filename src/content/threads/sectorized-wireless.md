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

Sectorization lets a node activate several directional antennas, each covering a sector, which reshapes the network topology itself. We define a sectorized network model that captures how sectors form multi-hop paths and how interference depends on the sector configuration, and we characterize the resulting capacity region.

Choosing the sectorization becomes an optimization problem, and reconfiguring it over time changes the effective topology as traffic changes. Our even homogeneous sectorization gives the effective graph a bipartite structure that simplifies capacity computation, speeds up routing and makes backpressure-style distributed control practical.
