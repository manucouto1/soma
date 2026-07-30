// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// GitHub Pages project site: https://manucouto1.github.io/soma/
// `base` must be set or every internal link resolves to the domain
// root and 404s once deployed. `astro dev` also serves under the base,
// so links that include it work identically in dev and in production.
const SITE = 'https://manucouto1.github.io';
const BASE = '/soma';

// https://astro.build/config
export default defineConfig({
	site: SITE,
	base: BASE,
	integrations: [
		starlight({
			title: 'Soma',
			description: 'A computational graph runtime for research pipelines, agent orchestration, and data virtualization.',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/manucouto1/soma' }],
			defaultLocale: 'en',
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'getting-started/introduction' },
						{ label: 'Quickstart', slug: 'getting-started/quickstart' },
						{ label: 'Tutorials (notebooks)', slug: 'getting-started/notebooks' },
						{ label: 'Problem & Solution', slug: 'getting-started/problem-solution' },
						{ label: 'Philosophy', slug: 'getting-started/philosophy' },
					],
				},
				{
					label: 'Architecture',
					items: [
						{ label: 'Overview', slug: 'architecture/overview' },
						{ label: 'Crate Structure', slug: 'architecture/crates' },
						{ label: 'Data Virtualization', slug: 'architecture/data-virtualization' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Execution Modes & Data Transport', slug: 'guides/execution-modes' },
						{ label: 'Gradient Health Audit', slug: 'guides/gradient-audit' },
						{ label: 'Checkpoints', slug: 'guides/checkpoints' },
					],
				},
				{
					label: 'Design',
					items: [
						{ label: 'Filter Model', slug: 'design/filter-model' },
						{ label: 'Caching System', slug: 'design/caching' },
						{ label: 'Streaming', slug: 'design/streaming' },
						{ label: 'Gradient Propagation', slug: 'design/gradients' },
						{ label: 'Compiler & Execution Plans', slug: 'design/compiler' },
						{ label: 'Event System', slug: 'design/events' },
						{ label: 'Experiment Tracking', slug: 'design/tracking' },
						{ label: 'Visualization', slug: 'design/visualization' },
						{ label: 'Hyperparameter Optimization', slug: 'design/optimization' },
						{ label: 'DataStore & S3', slug: 'design/data-store' },
						{ label: 'Scheduler', slug: 'design/scheduler' },
					],
				},
				{
					label: 'Platform',
					items: [
						{ label: 'Agents & Memory', slug: 'platform/agents' },
						{ label: 'Knowledge Base', slug: 'platform/knowledge-base' },
						{ label: 'Workers & Remote Execution', slug: 'platform/workers' },
						{ label: 'Environment Manager', slug: 'platform/env-manager' },
						{ label: 'Graph Integration', slug: 'platform/graph-integration' },
					],
				},
				{
					label: 'API Reference',
					items: [
						{
							// cargo-doc output, copied into dist/api/ by the
							// deploy workflow — a raw link, so it carries the
							// base explicitly (slug: entries get it for free).
							label: 'Rust API (cargo doc)',
							link: `${BASE}/api/soma_core/`,
							attrs: { target: '_blank' },
						},
						{ label: 'Python API', slug: 'api/python' },
					],
				},
				{
					label: 'Development',
					items: [
						{ label: 'Gitflow & Workflow', slug: 'development/gitflow' },
						{ label: 'TDD Strategy', slug: 'development/tdd' },
						{ label: 'Implementation Roadmap', slug: 'development/roadmap' },
						{ label: 'Architecture Review', slug: 'development/architecture-review' },
					],
				},
			],
		}),
	],
});
