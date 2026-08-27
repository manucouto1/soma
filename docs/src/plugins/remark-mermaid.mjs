// Turn ```mermaid fences into `<pre class="mermaid">` so Shiki leaves them
// alone and the client renderer can find them.
//
// The text after the language — ```mermaid What this shows — becomes the
// figure's caption.
//
// **Write `<` raw inside labels**, as in `["Arc<dyn Filter>"]`. The source is
// escaped once here and the HTML serializer round-trips it, so an author who
// escapes it too ends up with a literal `&lt;` on the page.
import { visit } from 'unist-util-visit';

const escapeText = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export function remarkMermaid() {
	return (tree) => {
		visit(tree, 'code', (node, index, parent) => {
			if (node.lang !== 'mermaid' || !parent) return;
			const caption = (node.meta ?? '').trim();
			parent.children[index] = {
				type: 'html',
				value:
					`<figure class="mermaid-figure">` +
					`<pre class="mermaid">${escapeText(node.value)}</pre>` +
					(caption ? `<figcaption>${escapeText(caption)}</figcaption>` : '') +
					`</figure>`,
			};
		});
	};
}
