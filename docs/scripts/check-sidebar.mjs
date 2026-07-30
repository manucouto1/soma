// Guard: every docs page must be reachable from the sidebar, every
// sidebar slug must exist, and every internal link must carry the
// `/soma` base.
//
// Orphaned pages build fine and are linkable by URL, so nothing else
// catches them — that is how three feature pages once stayed invisible.
// Base-less links are worse: they build, they look right in `npm run
// dev` (where there is no base), and they 404 only in production — that
// is how six links to the experiment-pool page shipped broken.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const DOCS = 'src/content/docs';
const BASE = '/soma';

function walk(dir) {
	return readdirSync(dir).flatMap((entry) => {
		const full = join(dir, entry);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

const files = walk(DOCS).filter((f) => /\.mdx?$/.test(f));

const pages = files
	.map((f) => relative(DOCS, f).replace(/\.mdx?$/, '').split(sep).join('/'))
	.filter((slug) => slug !== 'index');

// Every root-relative markdown link must start with the base. Anything
// pointing at another origin, an anchor, or the base itself is fine.
const baseless = files.flatMap((file) => {
	const text = readFileSync(file, 'utf8');
	return [...text.matchAll(/]\((\/[^)\s]*)\)/g)]
		.map((m) => m[1])
		.filter((href) => href !== BASE && !href.startsWith(`${BASE}/`))
		.map((href) => ({ file: relative('.', file), href }));
});

const config = readFileSync('astro.config.mjs', 'utf8');
const slugs = [...config.matchAll(/slug: '([^']+)'/g)].map((m) => m[1]);

const orphans = pages.filter((p) => !slugs.includes(p));
const dangling = slugs.filter((s) => !pages.includes(s));

if (orphans.length || dangling.length || baseless.length) {
	if (orphans.length) {
		console.error(`Pages missing from the sidebar (${orphans.length}):`);
		for (const p of orphans) console.error(`  - ${p}`);
	}
	if (dangling.length) {
		console.error(`Sidebar slugs with no page (${dangling.length}):`);
		for (const s of dangling) console.error(`  - ${s}`);
	}
	if (baseless.length) {
		console.error(
			`Internal links missing the "${BASE}" base (${baseless.length}) — ` +
				`these 404 in production:`,
		);
		for (const { file, href } of baseless) {
			console.error(`  - ${file}: ${href}  →  ${BASE}${href}`);
		}
	}
	process.exit(1);
}
console.log(
	`sidebar OK — ${pages.length} pages, all reachable; ` +
		`links OK — every internal link carries "${BASE}"`,
);
