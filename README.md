# ppromponas.com

Personal academic site of Panagiotis (Panos) Promponas, built with Astro and deployed to GitHub Pages by the workflow in `.github/workflows/deploy.yml`.

## Run locally

```
npm install
npm run dev        # live-reload server at http://localhost:4321
npm run build      # static output in dist/
```

## Add a paper

Add an entry to `src/data/publications.yaml`. Fields: `id`, `title`, `authors` (write yourself as `P. Promponas`), `year`, `type` (journal, conference, workshop, preprint), `venue`, optional `pillar` (area id) and `thread` (thread id), optional `selected: true` for the home page (quantum papers are listed first there, newest first within each group), optional `badge`, and `links` (`doi`, `ieee`, `arxiv`, `pdf`). A paper with a `thread` appears on that area page; a paper with only a `pillar` appears on the publications page under that area filter; a paper with neither appears only under "All areas".

## Add or rename a thread

Create `src/content/threads/<id>.md` with `title`, `pillar`, `order`, `summary`, and optional `media` (video or demo) and write the text below the frontmatter. Add a 28x28 line icon under the same id in `src/data/icons.ts` for the home-page pop-up. Tag papers with `thread: <id>`.

## Add an area

Create `src/content/pillars/<id>.md` with `title`, `order`, `lead` and `illustration`, and add a matching drawing in `src/components/Illustration.astro`.

## Media

Public media (videos, posters, the compiler demo script) lives in `public/media/` and is served as is. Material that must stay private lives outside the repo in `../unpublished/`.

## Deploy

Push to `main`. GitHub Actions builds the site and publishes it to GitHub Pages. `public/CNAME` keeps the custom domain.
