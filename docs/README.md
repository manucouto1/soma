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
| `scripts/use_cases_to_docs.py` | Splits `use-cases.md` into one page per slice |
| `scripts/python_to_docs.py` | Renders `python-surface.json` into the Python reference |
| `scripts/python_surface.py` | Dumps that JSON. **Needs somatize installed** — see below |
| `scripts/check-*.mjs` | The guards `npm run check` runs before building |
| `src/assets/logo-{light,dark}.svg` | The header mark (two files so the theme toggle works) |
| `public/favicon.svg` | Tab icon — the mark reduced to one period |
| `public/og-card.{svg,png}` | Social preview. The PNG is what `og:image` points at; regenerate it from the SVG with `sharp` after any edit. |

## Three groups of pages are generated

The tutorials from `examples/`, the use cases from `use-cases.md`, and the
Python reference from the package's own docstrings. All three output
directories are in `.gitignore` and a `pre*` hook renders them before every
`dev`, `build` and `check`: the source is the truth, and a committed copy would
eventually disagree with it.

### The Python reference has one extra step, and CI will fail without it

It is generated from docstrings, which needs somatize **imported** — and this
site builds on a runner with nothing but a stdlib `python3`. So it goes through
a JSON dump that **is committed**:

```bash
python docs/scripts/python_surface.py           # rewrite docs/python-surface.json
python docs/scripts/python_surface.py --check    # what CI runs
```

**Edit a public docstring and the reference is stale until you run that.** The
`--check` runs in CI's Python job — the only one that has already built and
installed the extension — so a stale dump fails there rather than shipping.
`npm run check` cannot catch it: it never imports the package.

## The tutorials are not written here

They are `examples/`, executed, with their outputs. `notebooks_to_docs.py` reads
the `.ipynb` JSON — no nbconvert, no dependency.

Figures are stored twice in those notebooks, Plotly for a live kernel and PNG so
they render on GitHub. A static site takes the PNG.

Adding a notebook to `examples/` is **not** enough to publish it — it has to be
given a place in the sidebar, and `check-sidebar` fails until it is. That is on
purpose, and the same is true of a tenth `somatize` submodule.

## What the guards enforce

Every one of these exists because it failed silently once.

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

**And it must land somewhere.** Reorganising the pages once left 43 links
aimed at paths that had moved; every one carried the base, so the base check
passed them. A link with a `#` must also reach a **heading** — most of those
are generated now, on the reference pages, so renaming a class would silently
aim them all at nothing.

**Every `crate/path.rs:line` anchor must resolve.** Dormant today — no page
cites source yet — and kept for the day one does, because a reference pointing
at a renamed file goes on reading plausibly.

**Every ```mermaid fence must parse, and its caption must carry no backtick.**
The diagrams render in the browser, so a syntax error would otherwise show up
as a blank space on a deployed page and nowhere else. The caption half is the
one that actually bit: a fence's info string may not contain a backtick, so a
caption with one is not a fence at all and markdown publishes the diagram as a
paragraph of literal source — building clean, with the body parsing fine.

Parsing is as far as a guard gets. `jsdom` gives mermaid a DOM but no layout,
so a diagram that parses can still lay out badly, and the only way to know is
to look:

```bash
npm run build && ln -s "$PWD/dist" /tmp/serve/soma && python3 -m http.server 8000 -d /tmp/serve
google-chrome --headless --virtual-time-budget=20000 --screenshot=/tmp/page.png \
  --window-size=1400,3000 http://127.0.0.1:8000/soma/running/the-plan/
```

Serving `dist` at the root instead of under `/soma` looks like it works and is
not a test: the assets 404, so the renderer never loads and every diagram is
missing for a reason that is not the diagram.

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
