# soma documentation

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
| `scripts/notebooks_to_docs.py` | Renders `examples/*.ipynb` into `src/content/docs/tutorials/` |
| `scripts/check-*.mjs` | The guards `npm run check` runs before building |
| `src/assets/logo-{light,dark}.svg` | The header mark (two files so the theme toggle works) |
| `public/favicon.svg` | Tab icon — the mark reduced to one period |
| `public/og-card.{svg,png}` | Social preview. The PNG is what `og:image` points at; regenerate it from the SVG with `sharp` after any edit. |

## The tutorials are not written here

They are `examples/`, executed, with their outputs. `notebooks_to_docs.py` reads
the `.ipynb` JSON — no nbconvert, no dependency — and writes the pages a `pre*`
hook before every `dev`, `build` and `check`. Both output directories are in
`.gitignore`: the notebook is the source of truth, and a committed copy of it
would eventually disagree with the notebook it came from.

Figures are stored twice in those notebooks, Plotly for a live kernel and PNG so
they render on GitHub. A static site takes the PNG.

Adding a notebook to `examples/` is **not** enough to publish it — it has to be
given a place in the sidebar, and `check-sidebar` fails until it is. That is on
purpose.

## Three things the guards enforce

All three exist because all three failed silently once.

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

**Every `crate/path.rs:line` anchor must resolve.** Dormant today — no page
cites source yet — and kept for the day one does, because a reference pointing
at a renamed file goes on reading plausibly.

**Every ```mermaid fence must parse.** The diagrams render in the browser, so a
syntax error would otherwise show up as a blank space on a deployed page and
nowhere else.

## Writing

Pages are Markdown with Starlight frontmatter (`title`, `description`).
Use `:::caution` for anything that describes intent rather than shipped
code — several pages carry one, and the reason is that documenting an API
nobody wrote is worse than not documenting it.

The Rust reference under `/soma/api/rust/` is `cargo doc` output, copied into
`dist/api/rust/` by the deploy workflow rather than built here. It briefly
occupied the site's own URL, which is what `.github/workflows/docs.yml` was
rewritten to undo.

## Regenerating the OG card

```bash
node -e "require('sharp')('public/og-card.svg',{density:144}).resize(1200,630).png().toFile('public/og-card.png')"
```
