import { defineCollection, z } from 'astro:content';
import { glob, file } from 'astro/loaders';
import { parse } from 'yaml';

// A pillar is one of the three research areas shown as boxes on the home page.
const pillars = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/pillars' }),
  schema: z.object({
    title: z.string(),
    order: z.number(),
    lead: z.string(),
    illustration: z.enum(['quantum', 'dqc', 'control']),
  }),
});

// A thread is a subsection inside a pillar page: description, papers, optional animation.
const threads = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/threads' }),
  schema: z.object({
    title: z.string(),
    pillar: z.string(),
    order: z.number(),
    summary: z.string(),
    media: z
      .object({
        type: z.enum(['video', 'demo', 'image']),
        src: z.string().optional(),
        poster: z.string().optional(),
        caption: z.string().optional(),
        duration: z.string().optional(),
      })
      .optional(),
    draft: z.boolean().optional(),
  }),
});

// All publications live in one YAML file. Add a paper by adding an entry there.
const publications = defineCollection({
  loader: file('src/data/publications.yaml', { parser: (text) => parse(text) }),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    authors: z.string(),
    year: z.number(),
    type: z.enum(['journal', 'conference', 'workshop', 'preprint']),
    venue: z.string(),
    pillar: z.string(),
    thread: z.string(),
    selected: z.boolean().optional(),
    badge: z.string().optional(),
    note: z.string().optional(),
    links: z
      .object({
        pdf: z.string().optional(),
        arxiv: z.string().optional(),
        doi: z.string().optional(),
        ieee: z.string().optional(),
        code: z.string().optional(),
      })
      .optional(),
  }),
});

export const collections = { pillars, threads, publications };
