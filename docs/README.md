# Soma documentation

The site at [manucouto1.github.io/soma](https://manucouto1.github.io/soma/),
built with [Starlight](https://starlight.astro.build).

```bash
npm install
npm run dev     # http://localhost:4321/soma/
npm run check   # guards + production build — run this before pushing
npm run build   # production build only
```

## Layout

| Path | What it is |
|---|---|
| `src/content/docs/` | Every page. The directory structure is the URL structure. |
| `astro.config.mjs` | Site config **and the sidebar** — a new page must be added here |
| `scripts/check-sidebar.mjs` | The guards `npm run check` runs before building |
| `src/assets/logo-{light,dark}.svg` | The header mark (two files so the theme toggle works) |
| `public/favicon.svg` | Tab icon — the mark reduced to one period |
| `public/og-card.{svg,png}` | Social preview. The PNG is what `og:image` points at; regenerate it from the SVG with `sharp` after any edit. |

## Two things the guards enforce

Both exist because both failed silently once.

**Every page must be in the sidebar.** An orphaned page builds fine and is
reachable by URL, so nothing else notices. Three feature pages once stayed
invisible that way.

**Every internal link must carry the `/soma` base.** This is a project site,
so `](/design/caching/)` resolves to the wrong host root in production — but
`astro dev` serves under the base too, which means a base-less link looks
perfectly fine locally and only 404s once deployed.

```markdown
[Caching](/soma/design/caching/)     ✅
[Caching](/design/caching/)          ❌ builds, then 404s in production
```

## Writing

Pages are Markdown with Starlight frontmatter (`title`, `description`).
Use `:::caution` for anything that describes intent rather than shipped
code — several pages carry one, and the reason is that documenting an API
nobody wrote is worse than not documenting it.

The Rust API docs under `/soma/api/` are `cargo doc` output, copied into
`dist/` by the deploy workflow rather than built here.

## Regenerating the OG card

```bash
node -e "require('sharp')('public/og-card.svg',{density:144}).resize(1200,630).png().toFile('public/og-card.png')"
```
