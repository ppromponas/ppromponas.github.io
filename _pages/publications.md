---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

Test markdown:

# Publications

## Journals

Pelekis, C., Promponas, P., Alvarado, J., Tsiropoulou, E. E., \& Papavassiliou, S. \href{https://arxiv.org/pdf/2102.12820.pdf}{A Fragile multi-CPR Game}.
Mathematical Methods of Operations Research (Springer). 


## Conferences
