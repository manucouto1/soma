// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
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
					label: 'Design',
					items: [
						{ label: 'Filter Model', slug: 'design/filter-model' },
						{ label: 'Caching System', slug: 'design/caching' },
						{ label: 'Streaming', slug: 'design/streaming' },
						{ label: 'Gradient Propagation', slug: 'design/gradients' },
						{ label: 'Compiler & Execution Plans', slug: 'design/compiler' },
						{ label: 'Event System', slug: 'design/events' },
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
							label: 'Rust API (cargo doc)',
							link: '/api/soma_core/',
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
