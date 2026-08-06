// Guard: every `D-nn` cross-reference must point at the entry it names.
//
// The Debt Register is 74 numbered entries that the rest of the Internals
// section links into, and its own top-ten table links back down. check-anchors
// verifies `file:line` against the source; nothing verified that a link *inside*
// the docs resolves. It did not, in four places: the entries were renumbered in
// blocks of ten and the anchors kept the old ids, so `[D-51]` pointed at the
// text of D-61. Both halves of such a link look right in isolation — the number
// reads plausible and the slug reads plausible — which is why it survived.
//
// Two checks, both hard failures:
//
//   A. Every `#d-nn-…` anchor must name an entry that exists in debt.md.
//   B. The anchor's slug must match that entry's heading, and where the link
//      text says `D-mm`, mm must equal nn.
//
// Slug rules are not re-derived here. Astro, rehype and GitHub disagree about
// what happens to `::` and `'`, and a guard that reimplements a slugifier fails
// on the slugifier's next release rather than on a real error. Comparison is on
// alphanumerics only, which is enough to tell D-61's heading from D-51's and
// immune to how punctuation is spelled.
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const DOCS = 'src/content/docs';
const REGISTER = 'src/content/docs/internals/debt.md';

// `### D-01 · `PyGraph` is the workspace's god object`
const HEADING = /^###\s+(D-\d+)\s*(?:·\s*)?(.*)$/;
// A markdown link whose target is a debt anchor, same-page or cross-page.
// Captures the visible text and the anchor's id + slug tail.
const REF = /\[([^\]]*)\]\((?:[^)#]*\/debt\/)?#d-(\d+)-*([^)]*)\)/g;

const squash = (s) =>
	s
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '')
		.trim();

function walk(dir) {
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

if (!existsSync(REGISTER)) {
	console.log('check-debt-refs: no debt register, nothing to check.');
	process.exit(0);
}

// ── The entries the register actually declares ───────────────────────────────
const entries = new Map(); // "D-61" → { title, squashed, line }
readFileSync(REGISTER, 'utf8')
	.split('\n')
	.forEach((text, idx) => {
		const m = HEADING.exec(text);
		if (m) entries.set(m[1], { title: m[2].trim(), squashed: squash(m[2]), line: idx + 1 });
	});

const unknown = [];
const mismatched = [];
const contradictory = [];
let refs = 0;

for (const page of walk(DOCS).filter((f) => /\.mdx?$/.test(f))) {
	const rel = relative('.', page);
	readFileSync(page, 'utf8')
		.split('\n')
		.forEach((text, idx) => {
			const where = `${rel}:${idx + 1}`;
			for (const [, label, num, slug] of text.matchAll(REF)) {
				refs++;
				const id = `D-${num}`;
				const entry = entries.get(id);
				if (!entry) {
					unknown.push({ where, id });
					continue;
				}
				// ── Check B ──
				// The slug is the heading with punctuation eaten; an entry whose
				// title is a prefix of another's would still be caught by the
				// number, so a prefix match is enough and tolerates truncation.
				const want = entry.squashed;
				const got = squash(slug);
				if (got && !want.startsWith(got) && !got.startsWith(want)) {
					mismatched.push({ where, id, slug, title: entry.title });
				}
				const said = /\bD-(\d+)\b/.exec(label);
				if (said && said[1] !== num) {
					contradictory.push({ where, said: `D-${said[1]}`, points: id });
				}
			}
		});
}

if (unknown.length || mismatched.length || contradictory.length) {
	if (unknown.length) {
		console.error(`References to debt entries that do not exist (${unknown.length}):`);
		for (const { where, id } of unknown) console.error(`  - ${where}: ${id}`);
	}
	if (mismatched.length) {
		console.error(`Anchors whose slug belongs to a different entry (${mismatched.length}):`);
		for (const { where, id, slug, title } of mismatched) {
			console.error(`  - ${where}: #d-${id.slice(2)}-${slug}`);
			console.error(`      but ${id} is "${title}"`);
		}
	}
	if (contradictory.length) {
		console.error(`Links whose text and target disagree (${contradictory.length}):`);
		for (const { where, said, points } of contradictory) {
			console.error(`  - ${where}: reads ${said}, links to ${points}`);
		}
	}
	console.error('\nFix the references, or the register is lying about itself.');
	process.exit(1);
}

console.log(
	`check-debt-refs: ${refs} references resolve across ${entries.size} debt entries.`,
);
