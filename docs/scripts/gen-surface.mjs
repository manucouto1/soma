// Regenerates the census in src/content/docs/internals/surface.md.
//
//     node scripts/gen-surface.mjs
//
// Everything above the "## The surface" heading is hand-written and preserved;
// everything below it is replaced. Run this after adding or removing anything a
// user can import or call, then commit the result.
//
// Why this page exists. The Internals section cuts horizontally, by crate: it
// answers "what is in soma-runtime?" and never "how many ways are there to
// build a graph?". The answer to the second question was six, and nothing in
// the repo said so. This page is the answer, counted rather than argued, and it
// is the metric the simplification work is measured against.
//
// Four mechanical sources, no interpretation:
//
//   1. `__all__` in soma/__init__.py, agentic.py, library.py, viz/__init__.py
//   2. `#[pymethods]` blocks across soma-python/src/*.rs
//   3. the class body of soma/_graph.py, which assigns Python methods onto Graph
//   4. add_class / add_function / add in the #[pymodule] of soma-python/src/lib.rs
//
// Usage counts are grep, not call graphs: a name appearing in a string or a
// comment counts. They are here to find the zeroes — a symbol nothing calls is
// a symbol that can go — and a small count is a hint, not a verdict.
//
// Deliberately NOT a build step, for the same reason as gen-symbol-index: the
// page is committed so it is greppable and reviewable in a diff, and a table
// that silently regenerates hides the symbol that disappeared.
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';

const REPO = '../';
const PAGE = 'src/content/docs/internals/surface.md';
const MARKER = '## The surface';
const PY = 'soma-python/python/soma';
const RS = 'soma-python/src';

// Where a symbol being used counts from. The generated tutorials are excluded:
// they are rendered from notebooks/, and counting both doubles every number.
//
// `pkg` is the package's own Python source, and it is counted apart from the
// four user-facing corpora on purpose. It is what separates a dead symbol from
// a live one that users never touch: the twelve `run_*_json` functions have no
// user outside the repo and exactly one caller inside it, which makes them
// plumbing registered on the module rather than API.
const CORPUS = [
	['tests', 'soma-python/tests', /\.py$/],
	['nb', 'notebooks', /\.ipynb$/],
	['docs', 'docs/src/content/docs', /\.mdx?$/],
	['ex', 'examples', /\.py$/],
	['pkg', 'soma-python/python/soma', /\.pyi?$/],
];
const USER_CORPORA = 4; // the first four; `pkg` is context, not use

// A missing corpus root is an ENVIRONMENT difference, not an empty corpus.
// `examples/` is a git submodule: a checkout without `submodules: true` sees
// an empty directory, every `ex` count silently becomes 0, and the only
// symptom is this file reporting itself stale in CI while it is current on
// the machine that wrote it. Counting zero there is the wrong answer; saying
// so is the right one.
function requireCorpusRoots() {
	const missing = CORPUS.filter(([, root]) => {
		const dir = join(REPO, root);
		return !existsSync(dir) || readdirSync(dir).length === 0;
	});
	if (missing.length === 0) return;
	console.error(
		'gen-surface: these corpus roots are missing or empty, so the counts ' +
			'would be wrong:\n' +
			missing.map(([col, root]) => `  ${root}  (the "${col}" column)`).join('\n') +
			'\n\n`examples/` is a git submodule — run:\n\n' +
			'    git submodule update --init\n\n' +
			'or, in CI, check out with `submodules: true`.',
	);
	process.exit(1);
}
const CORPUS_SKIP = /(?:^|\/)(?:tutorials|node_modules|__pycache__|\.ipynb_checkpoints)(?:\/|$)/;

const read = (p) => readFileSync(join(REPO, p), 'utf8');

function walk(dir) {
	if (!existsSync(dir)) return [];
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

// ── The corpus, read once ────────────────────────────────────────────────────
requireCorpusRoots();

const corpus = CORPUS.map(([label, dir, ext]) => ({
	label,
	texts: walk(join(REPO, dir))
		.filter((f) => ext.test(f) && !CORPUS_SKIP.test(relative(REPO, f)))
		// The census must not count itself, nor the page that indexes every type.
		.filter((f) => !/internals\/(surface|symbols)\.md$/.test(f))
		.map((f) => readFileSync(f, 'utf8')),
}));

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function uses(name, asMethod) {
	const re = new RegExp(asMethod ? `\\.${escape(name)}\\b` : `\\b${escape(name)}\\b`, 'g');
	return corpus.map(({ texts }) => texts.reduce((n, t) => n + (t.match(re)?.length ?? 0), 0));
}

// ── Source 1: __all__ ────────────────────────────────────────────────────────
// A list of string literals. Matching the literals inside the brackets is
// enough — every __all__ in this package is a plain list, and a computed one
// would show up as a missing row rather than a wrong one.
function allOf(path) {
	const m = /^__all__\s*=\s*\[([\s\S]*?)\]/m.exec(read(path));
	return m ? [...m[1].matchAll(/["'](\w+)["']/g)].map((x) => x[1]) : [];
}

// ── Source 2: #[pymethods] ───────────────────────────────────────────────────
// Rust name → Python name, from #[pyclass(name = "…")].
function pyclasses(text) {
	const map = new Map();
	const lines = text.split('\n');
	lines.forEach((l, i) => {
		const attr = /^#\[pyclass\b(.*)$/.exec(l);
		if (!attr) return;
		const name = /name\s*=\s*"(\w+)"/.exec(attr[1]);
		const decl = lines.slice(i, i + 4).find((x) => /^\s*pub(?:\(crate\))?\s+struct\s+\w+/.test(x));
		if (decl) {
			const rust = /struct\s+(\w+)/.exec(decl)[1];
			map.set(rust, name ? name[1] : rust);
		}
	});
	return map;
}

// Body of the fn starting at `from`: the text between its opening brace and the
// matching close. Braces inside strings would break this; none of these bodies
// has one, and a miscount shows up as a missing alias, never a wrong row.
function bodyOf(lines, from) {
	let depth = 0;
	let started = false;
	const out = [];
	for (let i = from; i < lines.length; i++) {
		for (const ch of lines[i]) {
			if (ch === '{') {
				depth++;
				started = true;
			} else if (ch === '}') depth--;
		}
		out.push(lines[i]);
		if (started && depth === 0) break;
	}
	return out.join('\n');
}

// A method whose whole body is one call to a *sibling `#[pymethods]` method* is
// an alias — two public names for one behaviour, which is what this page exists
// to surface. Forwarding to a private helper is not that: `handoff` delegating
// to `control_edge` is a method with an implementation, and reporting it as a
// duplicate path would be a false positive that teaches a reader to skip the
// column. Hence the second pass below, once the method names are known.
const ALIAS = /\{\s*(?:self|slf)\.(\w+)\s*\([^;{}]*\)\s*;?\s*\}\s*$/;

// A grep can never see a dunder: `__len__` is reached by `len(g)`, `__getitem__`
// by `t[k]`. Counting them as unused would put seven permanent false positives
// at the top of the report.
const DUNDER = /^__\w+__$/;

function pymethods() {
	const classes = [];
	for (const file of walk(join(REPO, RS)).filter((f) => f.endsWith('.rs')).sort()) {
		const text = readFileSync(file, 'utf8');
		const names = pyclasses(text);
		const lines = text.split('\n');
		lines.forEach((line, i) => {
			if (!/^#\[pymethods\]/.test(line)) return;
			const impl = lines.slice(i, i + 4).find((x) => /^impl\s+\w+/.test(x));
			if (!impl) return;
			const rust = /^impl\s+(\w+)/.exec(impl)[1];
			const cls = { name: names.get(rust) ?? rust, methods: [] };
			// Walk the impl block: rustfmt closes it with `}` in column 0.
			for (let j = i + 2; j < lines.length && !/^\}/.test(lines[j]); j++) {
				const fn = /^\s{4}(?:pub(?:\(crate\))?\s+)?fn\s+(\w+)/.exec(lines[j]);
				if (!fn) continue;
				const attrs = [];
				for (let k = j - 1; k >= 0 && /^\s*(#\[|\/\/)/.test(lines[k]); k--) {
					if (/^\s*#\[/.test(lines[k])) attrs.unshift(lines[k]);
				}
				const joined = attrs.join(' ');
				const renamed = /name\s*=\s*"(\w+)"/.exec(joined);
				const body = bodyOf(lines, j);
				const alias = ALIAS.exec(body);
				const kind = /#\[new\]/.test(joined)
					? 'new'
					: /#\[getter\]/.test(joined)
						? 'getter'
						: /#\[setter\]/.test(joined)
							? 'setter'
							: /#\[staticmethod\]/.test(joined)
								? 'static'
								: /#\[classmethod\]/.test(joined)
									? 'class'
									: 'method';
				// pyo3 exposes `#[setter] fn set_model` as the property `model`,
				// so the Rust name is not the name a user ever writes.
				const py = renamed ? renamed[1] : kind === 'setter' ? fn[1].replace(/^set_/, '') : fn[1];
				cls.methods.push({
					name: py,
					rust: fn[1],
					line: j + 1,
					file: relative(REPO, file),
					kind,
					alias: alias && alias[1] !== fn[1] ? alias[1] : null,
				});
			}
			// Second pass: an alias only counts if its target is itself exposed.
			const exposed = new Set(cls.methods.map((m) => m.rust));
			for (const m of cls.methods) if (m.alias && !exposed.has(m.alias)) m.alias = null;
			// A getter and its setter are one name to a user — `agent.model` reads
			// and writes through the same attribute — so they collapse to one row.
			// Left apart they double-count the surface and report every read-write
			// property twice in any list keyed by name.
			const props = new Map();
			cls.methods = cls.methods.filter((m) => {
				if (m.kind !== 'getter' && m.kind !== 'setter') return true;
				const seen = props.get(m.name);
				if (!seen) {
					m.kind = m.kind === 'getter' ? 'property, read-only' : 'property, write-only';
					props.set(m.name, m);
					return true;
				}
				seen.kind = 'property';
				return false;
			});
			classes.push(cls);
		});
	}
	return classes;
}

// ── Source 3: the class body of _graph.py ────────────────────────────────────
function graphPyMethods() {
	const text = read(`${PY}/_graph.py`);
	const rows = [];
	text.split('\n').forEach((l, i) => {
		const m = /^\s{4}(\w+)\s*=\s*(_?\w+)\.(\w+)\s*$/.exec(l);
		if (m) rows.push({ name: m[1], from: `${m[2]}.${m[3]}`, line: i + 1 });
	});
	// The two classmethods are declared with `def`, not assigned.
	text.split('\n').forEach((l, i) => {
		const m = /^\s{4}def\s+(\w+)\s*\(cls/.exec(l);
		if (m) rows.push({ name: m[1], from: 'classmethod', line: i + 1 });
	});
	return rows;
}

// ── Source 4: the #[pymodule] ────────────────────────────────────────────────
function moduleSurface() {
	const text = read(`${RS}/lib.rs`);
	return {
		classes: [...text.matchAll(/add_class::<(?:crate::\w+::)?(\w+)>/g)].map((m) => m[1]),
		functions: [...text.matchAll(/wrap_pyfunction!\((?:[\w:]+::)?(\w+)\s*,/g)].map((m) => m[1]),
		values: [...text.matchAll(/m\.add\(\s*"(\w+)"/g)].map((m) => m[1]),
	};
}

// ── Assemble ─────────────────────────────────────────────────────────────────
const out = [];
const H = '| Symbol | Defined at | Notes | tests | nb | docs | ex | pkg |';
const SEP = '|---|---|---|---|---|---|---|---|';
const row = (sym, at, notes, c) => `| \`${sym}\` | ${at} | ${notes} | ${c.join(' | ')} |`;
// Unused means no *user* names it. The package naming its own plumbing is not
// use; it is the reason the symbol looks alive.
const zero = (c) => c.slice(0, USER_CORPORA).every((n) => n === 0);

const modules = [
	['soma', `${PY}/__init__.py`],
	['soma.agentic', `${PY}/agentic.py`],
	['soma.library', `${PY}/library.py`],
	['soma.viz', `${PY}/viz/__init__.py`],
];

let exported = 0;
let dead = [];

out.push('', '### Exported names', '');
for (const [mod, path] of modules) {
	const names = allOf(path);
	exported += names.length;
	out.push('', `#### \`${mod}\` — ${names.length} names`, '', H, SEP);
	for (const n of names) {
		const c = uses(n, false);
		if (zero(c)) dead.push({ sym: `${mod}.${n}`, pkg: c[USER_CORPORA] });
		out.push(row(n, `\`${path}\``, '', c));
	}
}

const classes = pymethods();
const graphPy = graphPyMethods();
let methodCount = 0;
const aliases = [];

out.push('', '### Methods on the extension classes', '');
for (const cls of classes.sort((a, b) => b.methods.length - a.methods.length)) {
	const extra = cls.name === 'Graph' ? graphPy.length : 0;
	methodCount += cls.methods.length + extra;
	const total = cls.methods.length + extra;
	out.push(
		'',
		`#### \`${cls.name}\` — ${total} methods` +
			(extra ? ` (${cls.methods.length} in Rust, ${extra} in Python)` : ''),
		'',
		H,
		SEP,
	);
	for (const m of cls.methods) {
		const c = uses(m.name, true);
		const notes = [m.kind === 'method' ? '' : m.kind, m.alias ? `**alias of \`${m.alias}\`**` : '']
			.filter(Boolean)
			.join(' · ');
		if (m.alias) aliases.push(`${cls.name}.${m.name} → ${m.alias}`);
		if (zero(c) && m.kind !== 'new' && !DUNDER.test(m.name)) {
			dead.push({ sym: `${cls.name}.${m.name}`, pkg: c[USER_CORPORA] });
		}
		out.push(row(m.name, `\`${m.file}:${m.line}\``, notes, c));
	}
	if (cls.name === 'Graph') {
		for (const m of graphPy) {
			const c = uses(m.name, true);
			if (zero(c)) dead.push({ sym: `Graph.${m.name}`, pkg: c[USER_CORPORA] });
			out.push(row(m.name, `\`${PY}/_graph.py:${m.line}\``, `python · \`${m.from}\``, c));
		}
	}
}

const mod = moduleSurface();
out.push(
	'',
	'### Free functions in `soma._soma`',
	'',
	`${mod.functions.length} functions, ${mod.classes.length} classes and ${mod.values.length} values are registered on the extension module.`,
	'',
	H,
	SEP,
);
for (const f of mod.functions.sort()) {
	const c = uses(f, false);
	if (zero(c)) dead.push({ sym: `_soma.${f}`, pkg: c[USER_CORPORA] });
	out.push(row(f, `\`${RS}/lib.rs\``, '', c));
}

// ── The numbers this page exists to report ───────────────────────────────────
const summary = [
	'',
	'### Totals',
	'',
	'| Measure | Count |',
	'|---|---|',
	`| Exported names across the four modules | ${exported} |`,
	`| Methods on the extension classes | ${methodCount} |`,
	`| Free functions in \`soma._soma\` | ${mod.functions.length} |`,
	`| Literal aliases | ${aliases.length} |`,
	`| Named by no user, and by nothing in the package either | ${dead.filter((d) => !d.pkg).length} |`,
	`| Named by no user, but live inside the package | ${dead.filter((d) => d.pkg).length} |`,
	'',
];
if (aliases.length) {
	summary.push(
		'**Aliases** — a second public name whose whole body is a call to the first:',
		'',
	);
	for (const a of aliases) summary.push(`- \`${a}\``);
	summary.push('');
}
const orphans = dead.filter((d) => !d.pkg);
const plumbing = dead.filter((d) => d.pkg);
if (orphans.length) {
	summary.push(
		'**Unused** — nothing in tests, notebooks, docs, examples or the package',
		'itself names these. Dunders are excluded, since no grep can see `len(g)`',
		'reaching `__len__`. Treat each as a question worth answering:',
		'',
	);
	for (const d of orphans) summary.push(`- \`${d.sym}\``);
	summary.push('');
}
if (plumbing.length) {
	summary.push(
		'**Plumbing** — no user names these, but the package does. They are registered',
		'on the public module and carry its maintenance cost while serving one caller:',
		'',
	);
	for (const d of plumbing) summary.push(`- \`${d.sym}\` — ${d.pkg} uses inside the package`);
	summary.push('');
}

const page = readFileSync(PAGE, 'utf8');
const at = page.indexOf(MARKER);
if (at === -1) {
	console.error(`gen-surface: "${MARKER}" not found in ${PAGE}; refusing to overwrite.`);
	process.exit(1);
}
const next = `${page.slice(0, at + MARKER.length)}\n${[...summary, ...out].join('\n')}\n`;
const tally =
	`${exported} exported names, ${methodCount} methods, ` +
	`${mod.functions.length} free functions, ${aliases.length} aliases, ${dead.length} unused`;

// `--check` is what `npm run check` runs: it compares without writing, so the
// guard needs no clean working tree and says nothing about unrelated edits.
// Deleting a method and not regenerating is the failure this catches — the page
// would go on reporting a surface that no longer exists, which is worse than
// having no page, since this one is meant to be the metric.
if (process.argv.includes('--check')) {
	if (page !== next) {
		console.error(`gen-surface: ${PAGE} is stale. Run:\n\n    node scripts/gen-surface.mjs\n`);
		console.error(`  the source now has ${tally}`);
		process.exit(1);
	}
	console.log(`gen-surface: ${PAGE} is current — ${tally}.`);
	process.exit(0);
}

writeFileSync(PAGE, next);
console.log(`gen-surface: ${tally} → ${PAGE}`);
