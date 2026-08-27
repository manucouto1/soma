// Guard: every ```mermaid fence in the docs must parse.
//
// The diagrams render in the browser, so a syntax error would otherwise show
// up as a blank space on a deployed page and nowhere else. Mermaid needs a DOM
// even to parse — hence jsdom — but not to *lay out*, which is the part that
// would need a real browser.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { JSDOM } from 'jsdom';

const PAGES = 'src/content/docs';
const FENCE = /^```mermaid[^\n]*\n([\s\S]*?)^```/gm;

const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
// `navigator` is a getter-only global on modern Node; define over it.
Object.defineProperty(globalThis, 'navigator', {
	value: dom.window.navigator,
	configurable: true,
});

const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });

function walk(dir) {
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
}

const bad = [];
let count = 0;

for (const page of walk(PAGES).filter((f) => /\.mdx?$/.test(f))) {
	const text = readFileSync(page, 'utf8');
	for (const [, body] of text.matchAll(FENCE)) {
		count++;
		// The page escapes `<` on the way out; the browser hands mermaid the
		// decoded text, so parse what the browser will actually see.
		const src = body.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
		try {
			await mermaid.parse(src);
		} catch (e) {
			bad.push({ page: relative('.', page), message: String(e.message ?? e).split('\n')[0] });
		}
	}
}

if (bad.length) {
	console.error(`Mermaid diagrams that do not parse (${bad.length}):`);
	for (const b of bad) console.error(`  - ${b.page}: ${b.message}`);
	console.error('\nA diagram that does not parse renders as nothing at all.');
	process.exit(1);
}

// ── Inline SVG figures ───────────────────────────────────────────────────────
//
// A blank line inside a raw HTML block ends the block: markdown auto-closes the
// `<svg>` and renders the rest of the drawing as a paragraph of attribute text.
// It builds clean and looks catastrophic, so it is checked rather than
// remembered.
const FIGURE = /<figure class="soma-figure">[\s\S]*?<\/figure>/g;
const split = [];
let figures = 0;

for (const page of walk(PAGES).filter((f) => /\.mdx?$/.test(f))) {
	for (const [block] of readFileSync(page, 'utf8').matchAll(FIGURE)) {
		figures++;
		if (/\n[ \t]*\n/.test(block)) split.push(relative('.', page));
	}
}

if (split.length) {
	console.error(`Inline SVG figures with a blank line inside (${split.length}):`);
	for (const p of [...new Set(split)]) console.error(`  - ${p}`);
	console.error('\nA blank line closes the HTML block; the drawing renders as text.');
	process.exit(1);
}
console.log(`check-mermaid: ${count} diagrams parse, ${figures} inline figures unbroken.`);
