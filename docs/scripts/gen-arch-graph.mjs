// Regenerates the graph data embedded in src/content/docs/internals/graph.md.
//
//     node scripts/gen-arch-graph.mjs
//
// Extracts, from the Rust sources only (no cargo, no toolchain, no network):
//
//   nodes  — 13 crates, every public trait / struct / enum, plus #[pyclass]
//            types (pub(crate) in Rust, public to Python)
//   impls  — `impl Trait for Type`, the realization edges
//   owns   — struct fields whose type names another node, split into
//            composition (owned by value / Box / Vec) and aggregation
//            (Arc / Rc / & — shared, may outlive the owner)
//   deps   — path dependencies between workspace crates
//
// Everything from the first `#[cfg(test)]` in a file is ignored: test fixtures
// implement Filter and Step dozens of times and would swamp the real graph.
//
// The result replaces the `soma-arch-graph` JSON blob in the page — the same
// "<script type=application/json id=soma-data-*>" idiom `soma report` uses, so
// the page is a static artifact with no build-time data dependency.
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const REPO = '../';
const PAGE = 'src/content/docs/internals/graph.md';
const OPEN = '<script type="application/json" id="soma-arch-graph">';
const CLOSE = '</script>';

const CRATES = [
	'soma-macros',
	'soma-core',
	'soma-store',
	'soma-compiler',
	'soma-runtime',
	'soma-llm',
	'soma-agent',
	'soma-memory',
	'soma-worker',
	'soma-coordinator',
	'soma-mcp',
	'soma-python',
	'soma',
];

// Which Internals page documents each crate.
const PAGES = {
	'soma-core': ['foundation', 'soma-core-somatize-core'],
	'soma-macros': ['foundation', 'soma-macros-somatize-macros'],
	soma: ['foundation', 'soma-somatize--the-facade'],
	'soma-compiler': ['execution', 'soma-compiler-somatize-compiler'],
	'soma-runtime': ['execution', 'soma-runtime-somatize-runtime'],
	'soma-llm': ['agentic', 'soma-llm-somatize-llm'],
	'soma-agent': ['agentic', 'soma-agent-somatize-agent'],
	'soma-memory': ['agentic', 'soma-memory-somatize-memory'],
	'soma-mcp': ['agentic', 'soma-mcp-somatize-mcp'],
	'soma-worker': ['distribution', 'soma-worker-somatize-worker'],
	'soma-coordinator': ['distribution', 'soma-coordinator-somatize-coordinator'],
	'soma-store': ['distribution', 'soma-store-somatize-store'],
	'soma-python': ['python', ''],
};

// Anchored at column 0: module-level items only, so a `struct` declared inside
// a function body is not mistaken for a type. Private and pub(crate) items are
// matched too — they are held provisionally and kept only if they implement a
// workspace trait (PyStepBridge and PyPbtExecutor are plain `struct`, yet they
// are how a Python object becomes a Step).
const DECL = /^(?:pub(?:\(crate\))?\s+)?(trait|struct|enum)\s+([A-Z]\w*)/;
const NOT_PUBLIC = /^(?!pub\s)/;
const PYCLASS = /^\s*#\[pyclass\b/;
// `impl Trait for Type`, tolerating generics, lifetimes and a module path on
// either side. Trailing `{` or `where` is not required — some span two lines.
const IMPL = /^\s*impl(?:\s*<[^>]*>)?\s+((?:\w+::)*)([A-Z]\w*)(?:\s*<[^>]*>)?\s+for\s+((?:\w+::)*)([A-Z]\w*)/;
const FIELD = /^\s{4}(?:pub(?:\(crate\))?\s+)?(\w+)\s*:\s*(.+?),?\s*$/;

function walk(dir) {
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

// Production code is everything before the `#[cfg(test)] mod tests` block.
// It must be that pair, not a bare `#[cfg(test)]`: node_catalog.rs carries a
// test-only `use` at line 22, and cutting there dropped NodeCatalog, NodeImpl
// and `impl NodeRegistry for NodeCatalog` — the workspace's central seam.
function codeLines(all) {
	const end = all.findIndex((l, i) => {
		if (l.trim() !== '#[cfg(test)]') return false;
		const next = all.slice(i + 1, i + 3).find((x) => x.trim());
		return !!next && /^\s*(pub\s+)?mod\s/.test(next);
	});
	return end === -1 ? all : all.slice(0, end);
}

// ── Pass 1: nodes ────────────────────────────────────────────────────────────

const nodes = new Map(); // name -> node
const provisional = new Set(); // pub(crate), kept only if it implements a trait
const files = []; // { crate, path, lines }

for (const crate of CRATES) {
	const src = join(REPO, crate, 'src');
	if (!existsSync(src)) continue;
	for (const file of walk(src).filter((f) => f.endsWith('.rs')).sort()) {
		const lines = codeLines(readFileSync(file, 'utf8').split('\n'));
		files.push({ crate, path: file.slice(REPO.length), lines });

		lines.forEach((line, i) => {
			const m = DECL.exec(line);
			if (!m) return;
			const py = lines.slice(Math.max(0, i - 6), i).some((l) => PYCLASS.test(l));
			const name = m[2];
			// Non-`pub` types are kept provisionally: a #[pyclass] is public to
			// Python, and a type that realizes a workspace trait is public to
			// the architecture whatever Rust's visibility says — that is how the
			// four FFI bridges (PyFilterBridge, PyStepBridge, PyToolAdapter,
			// PyPbtExecutor) earn their place. Pass 2 drops the rest.
			if (NOT_PUBLIC.test(line) && !py) provisional.add(name);
			// First declaration wins; a duplicate name across crates is rare
			// and the graph keys on the name.
			if (nodes.has(name)) return;
			const [page, anchor] = PAGES[crate];
			nodes.set(name, {
				id: name,
				kind: m[1],
				crate,
				pyclass: py,
				file: file.slice(REPO.length),
				line: i + 1,
				page,
				anchor,
				impls: [],
				implementedBy: [],
				owns: [],
				ownedBy: [],
			});
		});
	}
}

// ── Pass 2: impl edges ───────────────────────────────────────────────────────

const impls = [];
const seenImpl = new Set();

for (const { crate, path, lines } of files) {
	lines.forEach((line, i) => {
		const m = IMPL.exec(line);
		if (!m) return;
		const [, , traitName, , typeName] = m;
		// Only edges between two nodes we know about. `impl Debug for X` and
		// `impl From<Foo> for Bar` where Foo is std are dropped by this.
		if (!nodes.has(traitName) || !nodes.has(typeName)) return;
		if (nodes.get(traitName).kind !== 'trait') return;
		const key = `${traitName}|${typeName}`;
		if (seenImpl.has(key)) return;
		seenImpl.add(key);
		impls.push({ trait: traitName, type: typeName, crate, file: path, line: i + 1 });
		nodes.get(traitName).implementedBy.push(typeName);
		nodes.get(typeName).impls.push(traitName);
	});
}

// A provisional pub(crate) type that implements nothing is genuinely internal.
// Drop it, then drop any edge that pointed at it.
for (const name of provisional) {
	if (nodes.get(name)?.impls.length) continue;
	nodes.delete(name);
}
for (let i = impls.length - 1; i >= 0; i--) {
	if (!nodes.has(impls[i].type) || !nodes.has(impls[i].trait)) impls.splice(i, 1);
}
for (const n of nodes.values()) {
	n.implementedBy = n.implementedBy.filter((x) => nodes.has(x));
	n.impls = n.impls.filter((x) => nodes.has(x));
}

// ── Pass 3: ownership edges from struct fields ───────────────────────────────

const owns = [];
const seenOwn = new Set();

for (const { lines } of files) {
	let current = null;
	let depth = 0;
	lines.forEach((line) => {
		if (current === null) {
			const m = DECL.exec(line);
			if (m && m[1] === 'struct' && line.includes('{') && nodes.has(m[2])) {
				current = m[2];
				depth = 1;
			}
			return;
		}
		depth += (line.match(/\{/g) ?? []).length - (line.match(/\}/g) ?? []).length;
		if (depth <= 0) {
			current = null;
			return;
		}
		const f = FIELD.exec(line);
		if (!f) return;
		const ty = f[2];
		// `Arc<dyn T>`, `Arc<T>`, `Rc<..>` and references are shared: the
		// pointee may outlive the owner. Everything else is owned outright.
		const shared = /\b(Arc|Rc)\s*</.test(ty) || ty.trimStart().startsWith('&');
		for (const [name, node] of nodes) {
			if (name === current) continue;
			if (!new RegExp(`\\b${name}\\b`).test(ty)) continue;
			const key = `${current}|${name}`;
			if (seenOwn.has(key)) continue;
			seenOwn.add(key);
			owns.push({
				from: current,
				to: name,
				field: f[1],
				ty: ty.replace(/\s+/g, ' ').slice(0, 80),
				shared,
				dyn: /\bdyn\s/.test(ty),
			});
			nodes.get(current).owns.push(name);
			node.ownedBy.push(current);
		}
	});
}

// ── Pass 4: crate dependencies ───────────────────────────────────────────────

const deps = [];
for (const crate of CRATES) {
	const manifest = join(REPO, crate, 'Cargo.toml');
	if (!existsSync(manifest)) continue;
	const text = readFileSync(manifest, 'utf8');
	for (const other of CRATES) {
		if (other === crate) continue;
		if (new RegExp(`^\\s*somatize-${other.replace(/^soma-?/, '')}\\s*=`, 'm').test(text)) {
			deps.push({ from: crate, to: other });
		} else if (new RegExp(`path\\s*=\\s*"\\.\\./${other}"`).test(text)) {
			deps.push({ from: crate, to: other });
		}
	}
}

// ── Emit ─────────────────────────────────────────────────────────────────────

const data = {
	crates: CRATES.filter((c) => [...nodes.values()].some((n) => n.crate === c)),
	nodes: [...nodes.values()].sort((a, b) => a.id.localeCompare(b.id)),
	impls,
	owns,
	deps,
};

const page = readFileSync(PAGE, 'utf8');
const start = page.indexOf(OPEN);
if (start === -1) {
	console.error(`gen-arch-graph: "${OPEN}" not found in ${PAGE}; refusing to overwrite.`);
	process.exit(1);
}
const end = page.indexOf(CLOSE, start);
writeFileSync(
	PAGE,
	page.slice(0, start + OPEN.length) + JSON.stringify(data) + page.slice(end),
);

const traits = data.nodes.filter((n) => n.kind === 'trait').length;
console.log(
	`gen-arch-graph: ${data.nodes.length} nodes (${traits} traits), ` +
		`${impls.length} impl edges, ${owns.length} ownership edges, ` +
		`${deps.length} crate deps → ${PAGE}`,
);
