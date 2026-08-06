// Renders data/traces.json into its two outputs:
//
//   1. the ASCII blocks between the `traces:begin/end` markers in
//      src/content/docs/internals/execution.md
//   2. the `soma-call-paths` JSON blob in src/content/docs/internals/paths.md
//
//     node scripts/gen-traces.mjs
//
// One source, two renderings. Hand-editing either output is how the two drift
// apart, which is the D-11 smell the debt register documents — writing the same
// logic twice and letting the copies diverge. Not in the section that denounces it.
//
// A side benefit that turned out to matter more than expected: the traces used
// to carry their `file:line` references in short form (`executor.rs:367`,
// `:1084`) that check-anchors.mjs could not see, because it only matches full
// repo paths in backticks. `files` here expands every one, so all 99 anchors are
// verified for the first time — check D in that script reads this data directly.
import { readFileSync, writeFileSync } from 'node:fs';

const DATA = 'data/traces.json';
const EXEC = 'src/content/docs/internals/execution.md';
const PATHS = 'src/content/docs/internals/paths.md';
const BEGIN = '<!-- traces:begin -->';
const END = '<!-- traces:end -->';
const BLOB_OPEN = '<script type="application/json" id="soma-call-paths">';
const BLOB_CLOSE = '</script>';

const LOC_COL_MIN = 50;
const LOC_COL_MAX = 66;
const PRIMITIVE = { P1: 'output_key', P2: 'compute_node', P3: 'store_output' };

const spec = JSON.parse(readFileSync(DATA, 'utf8'));

// ── resolution ───────────────────────────────────────────────────────────────

// `at` is written short for display. Expand it to a repo path so the anchor
// guard can check it, inheriting the file from the nearest ancestor when the
// hop only carries a line (`:1084`).
function resolve(at, inheritedFile) {
	if (!at) return { loc: null, file: inheritedFile };
	const m = /^(.*?):(\d+)$/.exec(at);
	if (!m) return { loc: null, file: inheritedFile };
	const file = m[1] || inheritedFile;
	if (!file) return { loc: null, file: inheritedFile };
	const full = spec.files[file];
	return { loc: full ? `${full}:${m[2]}` : null, file };
}

// The symbol a hop refers to, used to detect junctions between traces.
//
// Declared, never guessed. Parsing the first `ident(` out of the label looked
// tempting and produced `set` and `Ok` as junctions — `ctx.set(…)` and
// `→ Ok(())` appear on several paths and mean nothing. A junction is a claim
// about the architecture, so it is written down: either explicitly via `sym`,
// or implied by the P1/P2/P3 mark that names one of the three shared primitives.
function symbolOf(hop) {
	if (hop.sym) return hop.sym;
	if (hop.mark && PRIMITIVE[hop.mark]) return PRIMITIVE[hop.mark];
	return null;
}

// ── ASCII ────────────────────────────────────────────────────────────────────

function flatten(hop, prefix, isLast, isRoot, out) {
	const branch = isRoot ? '' : isLast ? '└─ ' : '├─ ';
	let text = prefix + branch + hop.label;
	if (hop.trail) text += `  ${hop.trail}`;
	if (hop.ref) text += `  → (${hop.ref})`;
	out.push({ text, at: hop.at ?? null });

	const kidPrefix = isRoot ? '' : prefix + (isLast ? '   ' : '│  ');
	for (const line of hop.note ?? []) out.push({ text: `${kidPrefix}   ${line}`, at: null });
	const kids = hop.children ?? [];
	kids.forEach((k, i) => flatten(k, kidPrefix, i === kids.length - 1, false, out));
}

function renderBlock(block) {
	const rows = [];
	flatten(block.root, '', true, true, rows);
	for (const t of block.tail ?? []) rows.push({ text: t.label, at: t.at ?? null });
	// One location column per block. Sized to the 90th percentile rather than
	// the longest line: a single wide hop would otherwise open a gutter across
	// the whole block. Anything past the column simply takes two spaces.
	const lens = rows.filter((r) => r.at).map((r) => r.text.length).sort((a, b) => a - b);
	const p90 = lens.length ? lens[Math.min(lens.length - 1, Math.floor(lens.length * 0.9))] : 0;
	const col = Math.max(LOC_COL_MIN, Math.min(LOC_COL_MAX, p90 + 2));
	return rows
		.map((r) => (r.at ? (r.text.length >= col ? r.text + '  ' : r.text.padEnd(col)) + r.at : r.text))
		.map((l) => l.replace(/\s+$/, ''))
		.join('\n');
}

const ascii = [];
for (const t of spec.traces) {
	ascii.push(`### (${t.id}) ${t.title}`, '');
	if (t.before) ascii.push(t.before, '');
	ascii.push('```');
	ascii.push(t.blocks.map(renderBlock).join('\n\n'));
	ascii.push('```', '');
	if (t.after) ascii.push(t.after, '');
}

const execPage = readFileSync(EXEC, 'utf8');
const bStart = execPage.indexOf(BEGIN);
const bEnd = execPage.indexOf(END);
if (bStart === -1 || bEnd === -1) {
	console.error(`gen-traces: ${BEGIN} / ${END} markers not found in ${EXEC}; refusing to overwrite.`);
	process.exit(1);
}
writeFileSync(
	EXEC,
	execPage.slice(0, bStart + BEGIN.length) + '\n\n' + ascii.join('\n').trimEnd() + '\n\n' + execPage.slice(bEnd),
);

// ── graph data ───────────────────────────────────────────────────────────────

let nextId = 0;
const symbolTraces = new Map(); // symbol -> Set(trace id)
const anchors = [];

function build(hop, traceId, inheritedFile) {
	const { loc, file } = resolve(hop.at, inheritedFile);
	if (loc) anchors.push(loc);
	const sym = symbolOf(hop);
	if (sym) {
		if (!symbolTraces.has(sym)) symbolTraces.set(sym, new Set());
		symbolTraces.get(sym).add(traceId);
	}
	return {
		id: `n${nextId++}`,
		label: hop.label,
		trail: hop.trail ?? null,
		at: hop.at ?? null,
		loc,
		sym,
		note: hop.note ?? [],
		mark: hop.mark ?? null,
		ref: hop.ref ?? null,
		debt: hop.debt ?? null,
		dyn: hop.dyn ?? null,
		children: (hop.children ?? []).map((k) => build(k, traceId, file)),
	};
}

const traces = spec.traces.map((t) => ({
	id: t.id,
	title: t.title.replace(/`/g, ''),
	entry: t.entry,
	blocks: t.blocks.map((b) => ({
		root: build(b.root, t.id, null),
		tail: (b.tail ?? []).map((x) => x.label),
	})),
}));

// A junction is a symbol reached from more than one trace — the thing the ASCII
// blocks structurally cannot show, because they sit hundreds of lines apart.
const junctions = [...symbolTraces.entries()]
	.filter(([, set]) => set.size > 1)
	.map(([sym, set]) => ({ sym, traces: [...set].sort() }))
	.sort((a, b) => b.traces.length - a.traces.length || a.sym.localeCompare(b.sym));

const blob = { traces, junctions };

const pathsPage = readFileSync(PATHS, 'utf8');
const pStart = pathsPage.indexOf(BLOB_OPEN);
if (pStart === -1) {
	console.error(`gen-traces: "${BLOB_OPEN}" not found in ${PATHS}; refusing to overwrite.`);
	process.exit(1);
}
const pEnd = pathsPage.indexOf(BLOB_CLOSE, pStart);
writeFileSync(
	PATHS,
	pathsPage.slice(0, pStart + BLOB_OPEN.length) + JSON.stringify(blob) + pathsPage.slice(pEnd),
);

console.log(
	`gen-traces: ${traces.length} traces, ${nextId} hops, ${anchors.length} resolved anchors, ` +
		`${junctions.length} junctions (${junctions.map((j) => j.sym).join(', ')})`,
);
