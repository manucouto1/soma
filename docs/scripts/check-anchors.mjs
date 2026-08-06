// Guard: the `file:line` anchors in the Internals pages must still resolve.
//
// Those pages carry ~700 references of the form `soma-core/src/filter.rs:120`.
// They are the whole value of the section — a reference whose anchors point at
// deleted files is worse than no reference, which is exactly what happened to
// development/architecture-review.md before it was demoted to a historical
// document.
//
// Two checks, both hard failures:
//
//   A. Every anchor's file must exist, and its line must be inside the file.
//      Runs on all Internals pages. Catches deletion and renaming of files.
//
//   B. Every row of the Symbol Index must actually declare the symbol it names,
//      within DECL_SLACK lines of the cited position. That page is generated and
//      claims "`Foo` is declared at path:line" 270 times, so it is both the most
//      precise claim in the section and the fastest to rot.
//
// Line numbers in prose are deliberately NOT checked. They drift on every edit
// above them, and a build that goes red for a one-line shift is a build whose
// guard gets deleted. Regenerate the Symbol Index (see its header) and check B
// will re-anchor everything that matters.
//
// Contextual shorthand inside tables (`cache/memory.rs:129` under a soma-runtime
// heading) is not resolvable and is skipped by construction: ANCHOR requires the
// path to start at a crate directory.
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const PAGES = 'src/content/docs/internals';
const TRACES = 'data/traces.json';
const REPO = '../';
const DECL_SLACK = 3;

const ANCHOR = /`(soma[\w.-]*(?:\/[\w.-]+)+\.(?:rs|py|pyi|toml)):(\d+)(?:-(\d+))?`/g;
// A Symbol Index row: | [`Name`](link) | kind | `crate` | `path:line` |
// The kind column may read "struct · pyclass", so it is matched loosely.
const INDEX_ROW = /^\|\s*\[`([A-Za-z_]\w*)`\]\([^)]*\)\s*\|[^|]+\|\s*`[^`]+`\s*\|\s*`([^`:]+):(\d+)`\s*\|/;
// `pub(crate)` has no space before the paren; #[pyclass] types are declared that
// way and are indexed, so the pattern must accept both forms.
const DECL = (name) => new RegExp(`^\\s*pub(?:\\(crate\\))?\\s+(?:trait|struct|enum)\\s+${name}\\b`);

function walk(dir) {
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

if (!existsSync(PAGES)) {
	console.log('check-anchors: no internals pages, nothing to check.');
	process.exit(0);
}

const cache = new Map();
function lines(path) {
	if (!cache.has(path)) {
		const full = join(REPO, path);
		cache.set(path, existsSync(full) ? readFileSync(full, 'utf8').split('\n') : null);
	}
	return cache.get(path);
}

const missing = [];
const overrun = [];
const undeclared = [];
let anchors = 0;
let rows = 0;

for (const page of walk(PAGES).filter((f) => /\.mdx?$/.test(f))) {
	const rel = relative('.', page);
	const isIndex = /symbols\.mdx?$/.test(page);

	readFileSync(page, 'utf8')
		.split('\n')
		.forEach((text, idx) => {
			const where = `${rel}:${idx + 1}`;

			// ── Check A ──
			for (const [, path, startStr, endStr] of text.matchAll(ANCHOR)) {
				anchors++;
				const body = lines(path);
				if (body === null) {
					missing.push({ where, path });
					continue;
				}
				const line = Number(endStr ?? startStr);
				if (line > body.length) overrun.push({ where, path, line, total: body.length });
			}

			// ── Check B ──
			if (!isIndex) return;
			const row = INDEX_ROW.exec(text);
			if (!row) return;
			const [, name, path, lineStr] = row;
			const body = lines(path);
			if (body === null) return; // already reported by check A
			rows++;
			const cited = Number(lineStr);
			const re = DECL(name);
			const near = body
				.slice(Math.max(0, cited - 1 - DECL_SLACK), cited + DECL_SLACK)
				.some((l) => re.test(l));
			if (near) return;
			const actual = body.findIndex((l) => re.test(l));
			undeclared.push({ where, name, path, cited, actual: actual >= 0 ? actual + 1 : null });
		});
}

// ── Check D: the execution traces ────────────────────────────────────────────
//
// The generated ASCII writes locations short (`executor.rs:367`, `:1084`) inside
// fenced blocks, where checks A and B cannot see them. Validate the source
// instead: every `at` in data/traces.json, expanded through its `files` map.
let traceAnchors = 0;
if (existsSync(TRACES)) {
	const spec = JSON.parse(readFileSync(TRACES, 'utf8'));
	const visit = (hop, trace, inheritedFile) => {
		let file = inheritedFile;
		const m = hop.at ? /^(.*?):(\d+)$/.exec(hop.at) : null;
		if (m) {
			file = m[1] || inheritedFile;
			const full = spec.files[file];
			const where = `${TRACES} (trace ${trace}, "${hop.label.slice(0, 40)}")`;
			if (!full) {
				missing.push({ where, path: `${file} — not in the "files" map` });
			} else {
				traceAnchors++;
				const body = lines(full);
				if (body === null) missing.push({ where, path: full });
				else if (Number(m[2]) > body.length) {
					overrun.push({ where, path: full, line: Number(m[2]), total: body.length });
				}
			}
		}
		for (const k of hop.children ?? []) visit(k, trace, file);
		return file;
	};
	for (const t of spec.traces) {
		for (const b of t.blocks) {
			// A tail line closes its block, so it inherits the block root's file.
			const rootFile = visit(b.root, t.id, null);
			for (const x of b.tail ?? []) visit(x, t.id, rootFile);
		}
	}
}

if (missing.length || overrun.length || undeclared.length) {
	if (missing.length) {
		console.error(`Anchors pointing at files that no longer exist (${missing.length}):`);
		for (const { where, path } of missing) console.error(`  - ${where}: ${path}`);
	}
	if (overrun.length) {
		console.error(`Anchors past the end of their file (${overrun.length}):`);
		for (const { where, path, line, total } of overrun) {
			console.error(`  - ${where}: ${path}:${line}, but the file has ${total} lines`);
		}
	}
	if (undeclared.length) {
		console.error(`Symbol Index rows that no longer match the source (${undeclared.length}):`);
		for (const { where, name, path, cited, actual } of undeclared) {
			console.error(
				actual === null
					? `  - ${where}: ${name} is no longer declared in ${path} (cited :${cited})`
					: `  - ${where}: ${name} is at ${path}:${actual}, cited as :${cited}`,
			);
		}
		console.error('\n  Regenerate the Symbol Index — see the note at the bottom of that page.');
	}
	console.error('\nFix the references, or the pages are lying about the code.');
	process.exit(1);
}

console.log(
	`check-anchors: ${anchors} page anchors + ${traceAnchors} trace anchors resolve ` +
		`across ${cache.size} files; ${rows} Symbol Index rows match their declarations.`,
);
