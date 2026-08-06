// Regenerates the A–Z table in src/content/docs/internals/symbols.md.
//
//     node scripts/gen-symbol-index.mjs
//
// Everything above the "## A–Z" heading is hand-written and preserved; the table
// below it is replaced. Run this after adding, removing or moving a public type,
// then commit the result — check-anchors.mjs verifies every row against the
// source on `npm run check`, so a stale index fails the build.
//
// Deliberately NOT a build step: the page is committed so it is greppable from
// the repo and reviewable in a diff, and a generator that runs on every build
// would hide a type disappearing behind a silently-updated table.
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const REPO = '../';
const PAGE = 'src/content/docs/internals/symbols.md';
const MARKER = '## A–Z';

// crate directory → [internals page slug, heading anchor on that page]
const PAGES = {
	'soma-core': ['foundation', '#soma-core-somatize-core'],
	'soma-macros': ['foundation', '#soma-macros-somatize-macros'],
	soma: ['foundation', '#soma-somatize--the-facade'],
	'soma-compiler': ['execution', '#soma-compiler-somatize-compiler'],
	'soma-runtime': ['execution', '#soma-runtime-somatize-runtime'],
	'soma-llm': ['agentic', '#soma-llm-somatize-llm'],
	'soma-agent': ['agentic', '#soma-agent-somatize-agent'],
	'soma-memory': ['agentic', '#soma-memory-somatize-memory'],
	'soma-mcp': ['agentic', '#soma-mcp-somatize-mcp'],
	'soma-worker': ['distribution', '#soma-worker-somatize-worker'],
	'soma-coordinator': ['distribution', '#soma-coordinator-somatize-coordinator'],
	'soma-store': ['distribution', '#soma-store-somatize-store'],
	'soma-python': ['python', ''],
};

// `pub(crate)` is included only for `#[pyclass]` types: they are public API,
// just to Python rather than to Rust, and leaving them out would hide the entire
// user-facing surface of soma-python. Other pub(crate) types are internal.
const DECL = /^\s*pub(?:\(crate\))?\s+(trait|struct|enum)\s+([A-Z]\w*)/;
const CRATE_ONLY = /^\s*pub\(crate\)\s/;
const PYCLASS = /^\s*#\[pyclass\b/;
const LABEL = { trait: '«trait»', struct: 'struct', enum: 'enum' };

function walk(dir) {
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

// Production code is everything before the `#[cfg(test)] mod tests` block.
// It must be that pair, not a bare `#[cfg(test)]`: node_catalog.rs carries a
// test-only `use` at line 22, and cutting there hid NodeCatalog and NodeImpl —
// the workspace's central registry — from this index entirely.
function codeLines(all) {
	const end = all.findIndex((l, i) => {
		if (l.trim() !== '#[cfg(test)]') return false;
		const next = all.slice(i + 1, i + 3).find((x) => x.trim());
		return !!next && /^\s*(pub\s+)?mod\s/.test(next);
	});
	return end === -1 ? all : all.slice(0, end);
}

const rows = [];
for (const crate of Object.keys(PAGES)) {
	const src = join(REPO, crate, 'src');
	if (!existsSync(src)) continue;
	for (const file of walk(src).filter((f) => f.endsWith('.rs')).sort()) {
		const body = codeLines(readFileSync(file, 'utf8').split('\n'));
		body.forEach((line, i) => {
			const m = DECL.exec(line);
			if (!m) return;
			// Derive attributes sit between the attribute and the declaration,
			// so look back a few lines for #[pyclass].
			const py = body.slice(Math.max(0, i - 6), i).some((l) => PYCLASS.test(l));
			if (CRATE_ONLY.test(line) && !py) return;
			rows.push({
				name: m[2],
				kind: m[1],
				pyclass: py,
				crate,
				path: file.slice(REPO.length),
				line: i + 1,
			});
		});
	}
}

rows.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()) || a.crate.localeCompare(b.crate));

const out = [];
let letter = null;
for (const r of rows) {
	const first = r.name[0].toUpperCase();
	if (first !== letter) {
		letter = first;
		out.push('', `### ${letter}`, '', '| Symbol | Kind | Crate | Defined at |', '|---|---|---|---|');
	}
	const [slug, anchor] = PAGES[r.crate];
	const kind = r.pyclass ? `${LABEL[r.kind]} · pyclass` : LABEL[r.kind];
	out.push(
		`| [\`${r.name}\`](/soma/internals/${slug}/${anchor}) | ${kind} | \`${r.crate}\` | \`${r.path}:${r.line}\` |`,
	);
}

const page = readFileSync(PAGE, 'utf8');
const at = page.indexOf(MARKER);
if (at === -1) {
	console.error(`gen-symbol-index: "${MARKER}" not found in ${PAGE}; refusing to overwrite.`);
	process.exit(1);
}
writeFileSync(PAGE, `${page.slice(0, at + MARKER.length)}\n${out.join('\n')}\n`);

const counts = rows.reduce((a, r) => ({ ...a, [r.kind]: (a[r.kind] ?? 0) + 1 }), {});
console.log(
	`gen-symbol-index: ${rows.length} public types ` +
		`(${counts.trait} traits, ${counts.struct} structs, ${counts.enum} enums) → ${PAGE}`,
);
