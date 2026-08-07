// Renders data/capabilities.json into src/content/docs/internals/capabilities.md.
//
//     node scripts/gen-capabilities.mjs
//     node scripts/gen-capabilities.mjs --check     # fail if the page is stale
//
// The Internals section cuts by crate, and the domain folders answer "where
// does this live". Neither answers the two questions you actually arrive with:
// *what happens when I call `g.fit()`*, and *what debt hangs off the cache*.
// One row per capability answers both, by crossing five things that already
// exist separately — the entry points (surface census), the execution trace
// (data/traces.json), the types (Symbol Index), the debt (the register) and
// the tests.
//
// Every column is checked against its source, which is the point: a table of
// hand-typed cross-references rots in a week. Four checks, and the third would
// have caught the broken register anchors that commit 8622270 had to go find:
//
//   1. every `entry` is a symbol the surface census lists
//   2. every `trace` names a trace in data/traces.json
//   3. every `debt` is a real `### D-nn` heading in the register
//   4. every `tests` path resolves on disk (a glob needs one match)
//
// `hops` are written short (`executor.rs:816`) and expanded here to full repo
// paths, so check-anchors.mjs verifies them like every other anchor — the same
// arrangement gen-traces.mjs uses, and for the same reason.
//
// Deliberately NOT a build step, like its siblings: the page is committed so it
// is greppable and reviewable in a diff.
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';

const REPO = '../';
const DATA = 'data/capabilities.json';
const PAGE = 'src/content/docs/internals/capabilities.md';
const SURFACE = 'src/content/docs/internals/surface.md';
const SYMBOLS = 'src/content/docs/internals/symbols.md';
const DEBT = 'src/content/docs/internals/debt.md';
const TRACES = 'data/traces.json';
const MARKER = '## The capabilities';

const spec = JSON.parse(readFileSync(DATA, 'utf8'));

// ── the four sources we check against ────────────────────────────────────────

// The surface census, as `{ exported: Set, methods: Map<class, Set> }`.
// Parsed from the generated page rather than re-derived: it is committed, and
// `gen-surface.mjs --check` is what keeps it honest.
function readSurface() {
	const text = readFileSync(SURFACE, 'utf8');
	const exported = new Set();
	const methods = new Map();
	let cls = null;
	let section = null;
	for (const line of text.split('\n')) {
		const head = /^###\s+(.*)$/.exec(line);
		if (head) {
			section = head[1];
			cls = null;
		}
		const sub = /^####\s+`([^`]+)`/.exec(line);
		if (sub) {
			cls = sub[1];
			if (!methods.has(cls)) methods.set(cls, new Set());
		}
		const row = /^\|\s*`([^`]+)`\s*\|/.exec(line);
		if (!row || !section) continue;
		// A `####` under "Exported names" is a module (`soma.agentic`), under
		// "Methods on the extension classes" a class. Rows go into `exported`
		// either way — a name is a name — and additionally under their owner,
		// so `Graph.fit` can be checked as a pair.
		exported.add(row[1]);
		if (cls) methods.get(cls).add(row[1]);
	}
	return { exported, methods };
}

const surface = readSurface();
const symbols = new Set(
	[...readFileSync(SYMBOLS, 'utf8').matchAll(/^\|\s*\[`([^`]+)`\]/gm)].map((m) => m[1]),
);
// The heading slug Starlight generates, so the links below are derived from
// the headings rather than kept in a second table that can disagree with them.
const slug = (heading) =>
	heading
		.toLowerCase()
		.replace(/[`'’]/g, '')
		.replace(/[^a-z0-9_ -]/g, '')
		.trim()
		.replace(/ /g, '-');

const debtSlugs = new Map(
	[...readFileSync(DEBT, 'utf8').matchAll(/^###\s+(D-\d+)\s(.*)$/gm)].map((m) => [
		m[1],
		slug(`${m[1]} ${m[2]}`),
	]),
);
const traceSlugs = new Map(
	[
		...readFileSync('src/content/docs/internals/execution.md', 'utf8').matchAll(
			/^###\s+\(([a-z])\)\s(.*)$/gm,
		),
	].map((m) => [m[1], slug(`(${m[1]}) ${m[2]}`)]),
);
const traceIds = new Set(JSON.parse(readFileSync(TRACES, 'utf8')).traces.map((t) => t.id));

// ── checks ───────────────────────────────────────────────────────────────────

const problems = [];
const complain = (id, msg) => problems.push(`  ${id}: ${msg}`);

// A `Class.method` entry needs both halves; a bare name is an exported symbol.
// `Graph(cache=)` names a constructor argument, so only the class is checked —
// the census counts names, not parameters.
function checkEntry(id, entry) {
	const bare = entry.replace(/\(.*\)$/, '');
	const [head, tail] = bare.split('.');
	if (!surface.exported.has(head) && !surface.methods.has(head)) {
		return complain(id, `entry \`${entry}\`: the surface census has no \`${head}\``);
	}
	if (tail === undefined) return;
	const own = surface.methods.get(head);
	if (!own?.has(tail)) {
		complain(id, `entry \`${entry}\`: \`${head}\` has no method \`${tail}\``);
	}
}

// A glob is one `*` in the last segment; anything else is a literal path.
function resolves(pattern) {
	const full = join(REPO, pattern);
	if (!pattern.includes('*')) return existsSync(full);
	const dir = dirname(full);
	if (!existsSync(dir)) return false;
	const re = new RegExp('^' + basename(pattern).replace(/[.]/g, '\\.').replace(/\*/g, '.*') + '$');
	return readdirSync(dir).some((f) => re.test(f));
}

for (const cap of spec.capabilities) {
	for (const entry of cap.entry) checkEntry(cap.id, entry);
	if (cap.trace !== null && !traceIds.has(cap.trace)) {
		complain(cap.id, `trace \`${cap.trace}\` is not in ${TRACES}`);
	}
	for (const type of cap.types) {
		if (!symbols.has(type)) complain(cap.id, `type \`${type}\` is not in the Symbol Index`);
	}
	for (const d of cap.debt) {
		if (!debtSlugs.has(d)) complain(cap.id, `debt \`${d}\` is not a heading in the register`);
	}
	for (const t of cap.tests) {
		if (!resolves(t)) complain(cap.id, `tests \`${t}\` matches nothing on disk`);
	}
	for (const hop of cap.hops) {
		const [file] = hop.split(':');
		if (!spec.files[file]) complain(cap.id, `hop \`${hop}\`: \`${file}\` is not in "files"`);
	}
}

if (problems.length) {
	console.error(`gen-capabilities: ${problems.length} broken cross-reference(s):`);
	console.error(problems.join('\n'));
	console.error('\n  Fix the data, or the page is lying about the code.');
	process.exit(1);
}

// ── render ───────────────────────────────────────────────────────────────────

const expand = (hop) => {
	const [file, line] = hop.split(':');
	return `\`${spec.files[file]}:${line}\``;
};
const traceLink = (t) =>
	t === null ? '—' : `[(${t})](/soma/internals/execution/#${traceSlugs.get(t)})`;
const debtLink = (d) => `[${d}](/soma/internals/debt/#${debtSlugs.get(d)})`;
const code = (xs) => xs.map((x) => `\`${x}\``).join(' · ');

const out = [];
out.push(MARKER, '');
out.push(
	`${spec.capabilities.length} rows. **Entry** is what a user writes; **trace** links the`,
	'execution path when one is written down; **hops** are where the work actually',
	'happens; **types** are what to look up in the',
	'[Symbol Index](/soma/internals/symbols/); **debt** is what is known to be wrong',
	'with it.',
	'',
);

for (const cap of spec.capabilities) {
	out.push(`### ${cap.title}`, '');
	out.push(cap.summary, '');
	out.push('| | |', '|---|---|');
	out.push(`| **Entry** | ${code(cap.entry)} |`);
	out.push(`| **Trace** | ${traceLink(cap.trace)} |`);
	out.push(`| **Hops** | ${cap.hops.map(expand).join(' → ')} |`);
	out.push(`| **Types** | ${code(cap.types)} |`);
	out.push(`| **Debt** | ${cap.debt.length ? cap.debt.map(debtLink).join(' · ') : 'none recorded'} |`);
	out.push(`| **Tests** | ${code(cap.tests)} |`);
	out.push('');
}

const page = readFileSync(PAGE, 'utf8');

// The preamble states how many rows are untraced. It is prose, above the
// marker, so this script cannot own it — but it can refuse to let it lie.
// It said "eight of the fifteen" for exactly as long as it took to write one
// more trace, which is the rot this page exists to argue against.
const CLAIM = /\*\*(\d+) of the (\d+)\*\*\s+rows below are untraced/;
const claim = CLAIM.exec(page);
const untraced = spec.capabilities.filter((c) => c.trace === null).length;
if (!claim) {
	console.error(
		`gen-capabilities: the preamble of ${PAGE} no longer states "**N of the M** rows below are untraced".\n` +
			'Keep the sentence in that form, or drop this check with it.',
	);
	process.exit(1);
}
if (Number(claim[1]) !== untraced || Number(claim[2]) !== spec.capabilities.length) {
	console.error(
		`gen-capabilities: the preamble of ${PAGE} says ${claim[1]} of ${claim[2]} rows are untraced; ` +
			`the table says ${untraced} of ${spec.capabilities.length}.`,
	);
	process.exit(1);
}

const at = page.indexOf(MARKER);
if (at === -1) {
	console.error(`gen-capabilities: "${MARKER}" not found in ${PAGE}; refusing to overwrite.`);
	process.exit(1);
}
const rendered = page.slice(0, at) + out.join('\n').trimEnd() + '\n';

if (process.argv.includes('--check')) {
	if (rendered !== page) {
		console.error(`gen-capabilities: ${PAGE} is stale. Run:\n\n    node scripts/gen-capabilities.mjs\n`);
		process.exit(1);
	}
	console.log(`gen-capabilities: ${PAGE} is current (${spec.capabilities.length} capabilities).`);
} else {
	writeFileSync(PAGE, rendered);
	const traced = spec.capabilities.filter((c) => c.trace !== null).length;
	console.log(
		`gen-capabilities: ${spec.capabilities.length} capabilities ` +
			`(${traced} traced, ${spec.capabilities.length - traced} not yet), ` +
			`${spec.capabilities.reduce((n, c) => n + c.hops.length, 0)} hops → ${PAGE}`,
	);
}
