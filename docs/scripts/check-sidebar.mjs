// Guard: every docs page must be reachable from the sidebar, and every
// sidebar slug must exist. Orphaned pages build fine and are linkable
// by URL, so nothing else catches them — that is how three feature
// pages once stayed invisible.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const DOCS = 'src/content/docs';

function walk(dir) {
	return readdirSync(dir).flatMap((entry) => {
		const full = join(dir, entry);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

const pages = walk(DOCS)
	.filter((f) => /\.mdx?$/.test(f))
	.map((f) => relative(DOCS, f).replace(/\.mdx?$/, '').split(sep).join('/'))
	.filter((slug) => slug !== 'index');

const config = readFileSync('astro.config.mjs', 'utf8');
const slugs = [...config.matchAll(/slug: '([^']+)'/g)].map((m) => m[1]);

const orphans = pages.filter((p) => !slugs.includes(p));
const dangling = slugs.filter((s) => !pages.includes(s));

if (orphans.length || dangling.length) {
	if (orphans.length) {
		console.error(`Pages missing from the sidebar (${orphans.length}):`);
		for (const p of orphans) console.error(`  - ${p}`);
	}
	if (dangling.length) {
		console.error(`Sidebar slugs with no page (${dangling.length}):`);
		for (const s of dangling) console.error(`  - ${s}`);
	}
	process.exit(1);
}
console.log(`sidebar OK — ${pages.length} pages, all reachable`);
