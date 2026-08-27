// Guard: every `file:line` anchor in the docs must still resolve.
//
// A page that cites `soma-core/src/fact.rs:41` is making a claim about the
// repository, and that is the kind of claim that rots without a word: the file
// gets renamed, the page keeps reading plausibly, and the reference points at
// nothing. Legacy carried ~700 of them in an Internals section that described
// code this repository does not have; the section is gone and the guard stays,
// widened from that one directory to every page, because the next anchor
// somebody writes should be checked on the day it is written.
//
// Line numbers in prose are deliberately NOT checked, only the file and whether
// the line is inside it. Prose numbers drift on every edit above them, and a
// build that goes red for a one-line shift is a build whose guard gets deleted.
//
// Contextual shorthand inside tables (`cache/memory.rs:129` under a crate
// heading) is not resolvable and is skipped by construction: the pattern
// requires the path to start at a crate directory.
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const PAGES = 'src/content/docs';
const REPO = '../';

const ANCHOR = /`(soma[\w.-]*(?:\/[\w.-]+)+\.(?:rs|py|pyi|toml)):(\d+)(?:-(\d+))?`/g;

function walk(dir) {
	return readdirSync(dir).flatMap((e) => {
		const full = join(dir, e);
		return statSync(full).isDirectory() ? walk(full) : [full];
	});
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
let anchors = 0;

for (const page of walk(PAGES).filter((f) => /\.mdx?$/.test(f))) {
	const rel = relative('.', page);
	readFileSync(page, 'utf8')
		.split('\n')
		.forEach((text, idx) => {
			const where = `${rel}:${idx + 1}`;
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
		});
}

if (missing.length || overrun.length) {
	if (missing.length) {
		console.error(`Anchors pointing at files that do not exist (${missing.length}):`);
		for (const { where, path } of missing) console.error(`  - ${where}: ${path}`);
	}
	if (overrun.length) {
		console.error(`Anchors past the end of their file (${overrun.length}):`);
		for (const { where, path, line, total } of overrun) {
			console.error(`  - ${where}: ${path}:${line}, but the file has ${total} lines`);
		}
	}
	console.error('\nFix the references, or the pages are lying about the code.');
	process.exit(1);
}

console.log(
	anchors === 0
		? 'check-anchors: no page cites source yet; nothing to resolve.'
		: `check-anchors: ${anchors} anchors resolve across ${cache.size} files.`,
);
