import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://ppromponas.com',
  trailingSlash: 'always',
  build: { format: 'directory' },
  // Old site URLs that no longer exist. /publications/ and /cv/ keep their paths.
  redirects: {
    '/portfolio': '/cv/#awards',
    '/about': '/',
  },
});
