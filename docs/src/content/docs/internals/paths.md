---
title: Call Paths
description: The five documented execution traces as one explorable graph, with the hops they share.
---

The [execution traces](/soma/internals/execution/#execution-traces) read as five
separate call chains. They are not. They are **five entry points into one
graph**, and the places where they meet are where the architecture's load-bearing
claims live.

That is the one thing two ASCII blocks three hundred lines apart structurally
cannot show, and it is why this page exists.

<style>
#cp { --cp-p: #7c6cf0; --cp-warn: #c9762b; --cp-dyn: #2b8a6f; border: 1px solid var(--sl-color-gray-5); border-radius: .5rem; overflow: hidden; margin: 1.5rem 0; background: var(--sl-color-black); }
:root[data-theme='light'] #cp { --cp-p: #5b4bd6; --cp-warn: #9c5a1a; --cp-dyn: #1d6b54; }
#cp-bar { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; padding: .6rem .75rem; border-bottom: 1px solid var(--sl-color-gray-5); background: var(--sl-color-gray-6); }
#cp-bar button { font: inherit; font-size: .78rem; padding: .25rem .55rem; border-radius: .3rem; border: 1px solid var(--sl-color-gray-5); background: transparent; color: var(--sl-color-gray-2); cursor: pointer; }
#cp-bar button[aria-pressed=true] { background: var(--sl-color-accent-low); color: var(--sl-color-white); border-color: var(--sl-color-accent); }
#cp-bar label { font-size: .78rem; color: var(--sl-color-gray-3); display: inline-flex; align-items: center; gap: .3rem; cursor: pointer; }
#cp-stage { display: flex; min-height: 26rem; }
#cp-tree { flex: 1 1 auto; overflow: auto; max-height: 36rem; padding: .7rem .9rem; font-family: var(--sl-font-mono); font-size: .74rem; line-height: 1.55; }
#cp-tree .row { white-space: pre; cursor: pointer; border-radius: .2rem; padding: 0 .2rem; }
#cp-tree .row:hover { background: var(--sl-color-gray-6); }
#cp-tree .row.sel { background: var(--sl-color-accent-low); }
#cp-tree .row.junction { font-weight: 700; }
#cp-tree .lbl { color: var(--sl-color-white); }
#cp-tree .tree { color: var(--sl-color-gray-4); }
#cp-tree .loc { color: var(--sl-color-gray-4); }
#cp-tree .note { color: var(--sl-color-gray-3); white-space: pre; }
#cp-tree .badge { font-size: .68rem; padding: 0 .3rem; border-radius: .25rem; margin-left: .4rem; }
#cp-tree .b-p { background: var(--cp-p); color: #fff; }
#cp-tree .b-warn { background: var(--cp-warn); color: #fff; }
#cp-tree .b-dyn { background: var(--cp-dyn); color: #fff; }
#cp-tree .b-ref { border: 1px solid var(--sl-color-gray-4); color: var(--sl-color-gray-2); }
#cp-tree h4 { font-family: var(--sl-font); font-size: .8rem; margin: 1rem 0 .3rem; color: var(--sl-color-gray-2); }
#cp-tree h4:first-child { margin-top: 0; }
#cp-side { flex: 0 0 16rem; border-left: 1px solid var(--sl-color-gray-5); padding: .8rem; overflow-y: auto; max-height: 36rem; font-size: .78rem; line-height: 1.5; }
#cp-side h3 { font-size: .9rem; margin: 0 0 .3rem; overflow-wrap: anywhere; }
#cp-side h4 { font-size: .7rem; text-transform: uppercase; letter-spacing: .04em; color: var(--sl-color-gray-3); margin: .8rem 0 .2rem; }
#cp-side code { font-size: .72rem; overflow-wrap: anywhere; }
#cp-side ul { margin: 0; padding-left: 1rem; }
#cp-side .empty { color: var(--sl-color-gray-4); font-style: italic; }
#cp-junc { padding: .5rem .75rem; border-top: 1px solid var(--sl-color-gray-5); font-size: .74rem; color: var(--sl-color-gray-3); }
#cp-junc b { color: var(--sl-color-white); }
@media (max-width: 50rem) { #cp-stage { flex-direction: column; } #cp-side { flex: 1 1 auto; border-left: 0; border-top: 1px solid var(--sl-color-gray-5); max-height: 18rem; } }
</style>

<div id="cp"><div id="cp-bar"><span id="cp-tabs"></span><label><input type="checkbox" id="cp-only-junc" /> only junctions</label><label><input type="checkbox" id="cp-notes" checked /> notes</label></div><div id="cp-stage"><div id="cp-tree"></div><div id="cp-side"><p class="empty">Select a hop.</p></div></div><div id="cp-junc"></div></div>

<script type="application/json" id="soma-call-paths">{"traces":[{"id":"a","title":"GraphSession::forward","entry":"GraphSession::forward","blocks":[{"root":{"id":"n0","label":"GraphSession::forward(x)","trail":null,"at":"graph_session.rs:333","loc":"soma-runtime/src/graph_session.rs:333","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n1","label":"forward_with(x, &Standard)","trail":null,"at":"graph_session.rs:334","loc":"soma-runtime/src/graph_session.rs:334","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n2","label":"run_driver()  → driver.clone().with_catalog(…)","trail":null,"at":"graph_session.rs:145","loc":"soma-runtime/src/graph_session.rs:145","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n3","label":"Standard::forward(graph, &ForwardEnv{…}, x)","trail":null,"at":"forward.rs:49","loc":"soma-runtime/src/forward.rs:49","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n4","label":"compile(graph, catalog, Inference, Some(cache))","trail":null,"at":"forward.rs:51","loc":"soma-runtime/src/forward.rs:51","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n5","label":"run_forward(graph, &plan, env, x)","trail":null,"at":"forward.rs:77","loc":"soma-runtime/src/forward.rs:77","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n6","label":"timestamp_id(\"forward\")","trail":null,"at":"forward.rs:83","loc":"soma-runtime/src/forward.rs:83","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n7","label":"RunContext::new(…, GraphInfo::from_graph(graph))","trail":null,"at":"forward.rs:84","loc":"soma-runtime/src/forward.rs:84","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n8","label":"LocalRunner.forward(plan, &ctx, x)","trail":null,"at":"forward.rs:94","loc":"soma-runtime/src/forward.rs:94","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n9","label":"walk(plan, ctx, input, RunMode::Forward)","trail":null,"at":"runner/local.rs:26","loc":"soma-runtime/src/runner/local.rs:26","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n10","label":"Context::new(…).with_graph_info(…).with_seed(…)","trail":null,"at":"runner/local.rs:33","loc":"soma-runtime/src/runner/local.rs:33","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n11","label":"exec.set(input_key(first), input)","trail":null,"at":"runner/local.rs:40","loc":"soma-runtime/src/runner/local.rs:40","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n12","label":"executor::execute(…)","trail":null,"at":"runner/local.rs:44","loc":"soma-runtime/src/runner/local.rs:44","sym":"execute","note":[],"mark":null,"ref":"b","debt":null,"dyn":null,"children":[]},{"id":"n13","label":"last_output(&exec)","trail":null,"at":"runner/local.rs:51","loc":"soma-runtime/src/runner/local.rs:51","sym":null,"note":["execution_order().rev().find(!reserved)"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]}]}]}]}]}]},"tail":[]}]},{"id":"b","title":"execute → run_node → the three primitives","entry":"execute","blocks":[{"root":{"id":"n14","label":"execute(plan, ctx, catalog, cache)","trail":null,"at":"executor.rs:367","loc":"soma-runtime/src/executor.rs:367","sym":"execute","note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n15","label":"Empty                → Ok(())","trail":null,"at":":374","loc":"soma-runtime/src/executor.rs:374","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n16","label":"Execute{id}          → execute_node(id, &[], …)","trail":null,"at":":377","loc":"soma-runtime/src/executor.rs:377","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n17","label":"Step{id, handoffs}   → execute_node(id, handoffs, …)","trail":null,"at":":379","loc":"soma-runtime/src/executor.rs:379","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n18","label":"Sequence(v)          → for each: execute(…)","trail":null,"at":":383","loc":"soma-runtime/src/executor.rs:383","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n19","label":"Parallel(b)          → execute_parallel","trail":null,"at":":1084","loc":"soma-runtime/src/executor.rs:1084","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n20","label":"Loop{…}              → execute_loop","trail":null,"at":":460","loc":"soma-runtime/src/executor.rs:460","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n21","label":"Branch{…}            → execute_branch","trail":null,"at":":531","loc":"soma-runtime/src/executor.rs:531","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n22","label":"Remote{…, target: _} → execute_remote","trail":"(! target discarded)","at":":601","loc":"soma-runtime/src/executor.rs:601","sym":null,"note":[],"mark":"warn","ref":null,"debt":"D-42","dyn":null,"children":[]},{"id":"n23","label":"Composite{ids}       → composite_fit (fit) | per-node (fwd)","trail":null,"at":":953","loc":"soma-runtime/src/executor.rs:953","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n24","label":"Stream{ids, size}    → execute_stream","trail":null,"at":":1208","loc":"soma-runtime/src/executor.rs:1208","sym":"execute_stream","note":[],"mark":null,"ref":"d","debt":null,"dyn":null,"children":[]},{"id":"n25","label":"other                → Err(\"newer compiler\")","trail":null,"at":":445","loc":"soma-runtime/src/executor.rs:445","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]},"tail":[]},{"root":{"id":"n26","label":"run_node(node_id, ctx, catalog, cache)","trail":null,"at":"executor.rs:816","loc":"soma-runtime/src/executor.rs:816","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n27","label":"node = catalog.node(id)?.clone(); meta = node.meta()","trail":null,"at":":824","loc":"soma-runtime/src/executor.rs:824","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n28","label":"input = resolve_input(node_id, ctx)","trail":null,"at":":1173","loc":"soma-runtime/src/executor.rs:1173","sym":null,"note":["0 preds → execution_order.last()  (!)  D-44","1 pred  → that pred","n preds → merged JSON object, keyed by predecessor"],"mark":"warn","ref":null,"debt":"D-44","dyn":null,"children":[]},{"id":"n29","label":"fitted = fit_state_if_needed(…)","trail":null,"at":":1002","loc":"soma-runtime/src/executor.rs:1002","sym":null,"note":["guard: mode.is_fit() && meta.trainable()","key = salt_with_seed(CacheKey::for_state(config, x, y), seed)","hit → reuse | miss → catch_unwind(filter.fit) → put_computed"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n30","label":"▸ PRIMITIVE 1  output_key(node, meta, state, input_hash, seed)","trail":null,"at":":670","loc":"soma-runtime/src/executor.rs:670","sym":"output_key","note":["guard: !(meta.cacheable && meta.deterministic) → None"],"mark":"P1","ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n31","label":"cache.get_located(key) hit → emit NodeCacheHit, return Produced","trail":null,"at":":862","loc":"soma-runtime/src/executor.rs:862","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n32","label":"miss → emit NodeCacheMiss, emit NodeStarted","trail":null,"at":":876","loc":"soma-runtime/src/executor.rs:876","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n33","label":"▸ PRIMITIVE 2  compute_node(…) = catch_unwind(run_node_inner)","trail":null,"at":":692","loc":"soma-runtime/src/executor.rs:692","sym":"compute_node","note":[],"mark":"P2","ref":null,"debt":null,"dyn":null,"children":[{"id":"n34","label":"run_node_inner","trail":null,"at":":1058","loc":"soma-runtime/src/executor.rs:1058","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"NodeImpl","children":[{"id":"n35","label":"NodeImpl::Filter(f) → f.forward(input, state) → Produced","trail":null,"at":":1066","loc":"soma-runtime/src/executor.rs:1066","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"Filter","children":[]},{"id":"n36","label":"NodeImpl::Step(s)   → driver.run(s, run_id, node_id, input)","trail":null,"at":":1069","loc":"soma-runtime/src/executor.rs:1069","sym":"EffectDriver::run","note":[],"mark":null,"ref":"c","debt":null,"dyn":"Step","children":[]}]}]},{"id":"n37","label":"match outcome","trail":null,"at":":905","loc":"soma-runtime/src/executor.rs:905","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n38","label":"Produced(out) → ▸ PRIMITIVE 3  store_output(…)","trail":null,"at":":717","loc":"soma-runtime/src/executor.rs:717","sym":"store_output","note":["→ maybe_spill → set_virtual → NodeCompleted"],"mark":"P3","ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n39","label":"HandOff{target, carry} → ctx.set(node, carry); NodeCompleted","trail":null,"at":":931","loc":"soma-runtime/src/executor.rs:931","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n40","label":"Paused{turn, reason}   → nothing stored","trail":null,"at":":942","loc":"soma-runtime/src/executor.rs:942","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]}]},"tail":[]},{"root":{"id":"n41","label":"back in execute_node","trail":null,"at":"executor.rs:748","loc":"soma-runtime/src/executor.rs:748","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n42","label":"Produced → Ok(())","trail":null,"at":null,"loc":null,"sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n43","label":"HandOff  → select_handoff(:770) → execute(that plan)","trail":null,"at":null,"loc":null,"sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n44","label":"Paused   → Err(SomaError::Suspended{…})","trail":null,"at":null,"loc":null,"sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]},"tail":[]}]},{"id":"c","title":"The EffectDriver turn loop","entry":"EffectDriver::run","blocks":[{"root":{"id":"n45","label":"EffectDriver::run(step, run_id, node_id, input)","trail":null,"at":"effects/mod.rs:107","loc":"soma-runtime/src/effects/mod.rs:107","sym":"EffectDriver::run","note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n46","label":"journal = self.journal.with_enabled(enabled && meta.journal)","trail":null,"at":":116","loc":"soma-runtime/src/effects/mod.rs:116","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n47","label":"for turn in 0..meta.max_turns","trail":null,"at":":127","loc":"soma-runtime/src/effects/mod.rs:127","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n48","label":"emit AgentTurnStarted","trail":null,"at":null,"loc":null,"sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n49","label":"ctx = StepCtx::new(…).with_history(&history)","trail":null,"at":":134","loc":"soma-runtime/src/effects/mod.rs:134","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n50","label":"transition = step.poll(&ctx)?","trail":null,"at":":138","loc":"soma-runtime/src/effects/mod.rs:138","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"Step","children":[]},{"id":"n51","label":"match transition","trail":null,"at":":146","loc":"soma-runtime/src/effects/mod.rs:146","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n52","label":"Await(effects)  → perform_all(…)","trail":null,"at":":440","loc":"soma-runtime/src/effects/mod.rs:440","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n53","label":"emit EffectRequested per effect","trail":null,"at":null,"loc":null,"sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n54","label":"thread::scope: one thread per effect","trail":null,"at":":459","loc":"soma-runtime/src/effects/mod.rs:459","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n55","label":"perform_one(journal, EffectSite{run,node,turn,i})","trail":null,"at":":519","loc":"soma-runtime/src/effects/mod.rs:519","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n56","label":"journal.lookup(site, effect)? → replayed","trail":null,"at":":527","loc":"soma-runtime/src/effects/mod.rs:527","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n57","label":"handlers.iter().find(|h| h.handles(effect))","trail":null,"at":":531","loc":"soma-runtime/src/effects/mod.rs:531","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"EffectHandler","children":[]},{"id":"n58","label":"handler.perform(effect)?","trail":null,"at":":543","loc":"soma-runtime/src/effects/mod.rs:543","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"EffectHandler","children":[]},{"id":"n59","label":"journal.record(…)   (Failed is never recorded)","trail":null,"at":":545","loc":"soma-runtime/src/effects/mod.rs:545","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]}]},{"id":"n60","label":"usage += …; emit ToolCalled / EffectCompleted","trail":"→ history.push(results)","at":":159","loc":"soma-runtime/src/effects/mod.rs:159","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]},{"id":"n61","label":"Done(v)             → Ok(NodeOutcome::Produced(v))","trail":null,"at":":167","loc":"soma-runtime/src/effects/mod.rs:167","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n62","label":"Goto{target, carry} → Ok(NodeOutcome::HandOff{…})","trail":null,"at":":172","loc":"soma-runtime/src/effects/mod.rs:172","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n63","label":"Suspend{reason}","trail":null,"at":":187","loc":"soma-runtime/src/effects/mod.rs:187","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n64","label":"journal.lookup(site, suspension_effect(reason))?","trail":null,"at":null,"loc":null,"sym":null,"note":["Some → emit Resumed, continue   « the resume path »"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n65","label":"None → emit Suspended → Ok(NodeOutcome::Paused{…})","trail":null,"at":":206","loc":"soma-runtime/src/effects/mod.rs:206","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]},{"id":"n66","label":"Spawn{specs, join} → spawn_all(…)","trail":null,"at":":316","loc":"soma-runtime/src/effects/mod.rs:316","sym":null,"note":["child ids \"{node_id}/{label|turn.index}\"; thread::scope;","RECURSES into self.run per child                         :365","JoinPolicy: All | AllSettled | First  (! First still joins all)"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]}]}]},"tail":["loop exhausted → Err(\"did not finish within N turns\")"]}]},{"id":"d","title":"StreamRun","entry":"execute_stream","blocks":[{"root":{"id":"n67","label":"execute_stream","trail":null,"at":"executor.rs:1208","loc":"soma-runtime/src/executor.rs:1208","sym":"execute_stream","note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n68","label":"refuse if mode is Fit","trail":null,"at":":1220","loc":"soma-runtime/src/executor.rs:1220","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n69","label":"chunks = chunk_value(input, chunk_size)","trail":null,"at":":1264","loc":"soma-runtime/src/executor.rs:1264","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n70","label":"run = StreamRun::new(node_ids, catalog)   (steps → Err)","trail":null,"at":"stream.rs:83","loc":"soma-runtime/src/executors/stream.rs:83","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n71","label":"per chunk: run.process_chunk(chunk, ctx, cache)","trail":null,"at":"stream.rs:130","loc":"soma-runtime/src/executors/stream.rs:130","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n72","label":"per node i:","trail":null,"at":null,"loc":null,"sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n73","label":"Barrier  → buffer the value, stop the cascade","trail":null,"at":":140","loc":"soma-runtime/src/executors/stream.rs:140","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n74","label":"else     → current = run_compute(i, current, …)","trail":null,"at":":194","loc":"soma-runtime/src/executors/stream.rs:194","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n75","label":"first touch → emit NodeStarted","trail":null,"at":":203","loc":"soma-runtime/src/executors/stream.rs:203","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n76","label":"state = evolving.or(base_state)","trail":null,"at":":213","loc":"soma-runtime/src/executors/stream.rs:213","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n77","label":"▸ output_key(…)","trail":null,"at":":217","loc":"soma-runtime/src/executors/stream.rs:217","sym":"output_key","note":[],"mark":"P1","ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n78","label":"cache hit → counters only","trail":"(!) no event","at":":220","loc":"soma-runtime/src/executors/stream.rs:220","sym":null,"note":[],"mark":"warn","ref":null,"debt":"D-11","dyn":null,"children":[]},{"id":"n79","label":"▸ compute_node(…)","trail":null,"at":":233","loc":"soma-runtime/src/executors/stream.rs:233","sym":"compute_node","note":[],"mark":"P2","ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n80","label":"▸ store_output(…); evolving update","trail":null,"at":":239","loc":"soma-runtime/src/executors/stream.rs:239","sym":"store_output","note":[],"mark":"P3","ref":null,"debt":null,"dyn":null,"children":[]}]}]}]},{"id":"n81","label":"run.flush(ctx, cache)  → materialize_buffer per barrier node","trail":null,"at":"stream.rs:152","loc":"soma-runtime/src/executors/stream.rs:152","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n82","label":"run.finish(ctx)  → one NodeCompleted per node,","trail":null,"at":"stream.rs:167","loc":"soma-runtime/src/executors/stream.rs:167","sym":null,"note":["\"stream: N chunks, H hits, M misses\""],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n83","label":"ctx.set(last_id, output.finish())","trail":null,"at":"stream.rs:310","loc":"soma-runtime/src/executors/stream.rs:310","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]},"tail":[]}]},{"id":"e","title":"StudyRunner and PbtRunner","entry":"StudyRunner::run","blocks":[{"root":{"id":"n84","label":"StudyRunner::run(study, sampler, executor)","trail":null,"at":"executors/study.rs:187","loc":"soma-runtime/src/executors/study.rs:187","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n85","label":"sampler.prepare(&study.search_space)","trail":null,"at":":193","loc":"soma-runtime/src/executors/study.rs:193","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"Sampler","children":[]},{"id":"n86","label":"RESUME: replay completed trials into sampler.record_result","trail":null,"at":":205","loc":"soma-runtime/src/executors/study.rs:205","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n87","label":"pruner = build_pruner(&study.pruning)","trail":null,"at":":407","loc":"soma-runtime/src/executors/study.rs:407","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":"Pruner","children":[]},{"id":"n88","label":"trial_index = study.trials.len()          « the resume point »","trail":null,"at":":218","loc":"soma-runtime/src/executors/study.rs:218","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n89","label":"loop","trail":null,"at":":228","loc":"soma-runtime/src/executors/study.rs:228","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n90","label":"config_index = i / n_seeds ; seed_slot = i % n_seeds","trail":null,"at":":229","loc":"soma-runtime/src/executors/study.rs:229","sym":null,"note":["seed_slot > 0 → reuse the previous trial's params minus \"seed\"","else          → sampler.sample(space, config_index)?"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n91","label":"params += {\"seed\": …} += study.frozen","trail":null,"at":":250","loc":"soma-runtime/src/executors/study.rs:250","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n92","label":"ctx = TrialContext{objective, pruner, history, bus, shared}","trail":null,"at":":270","loc":"soma-runtime/src/executors/study.rs:270","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n93","label":"outcome = executor.execute_trial(&params, &ctx)","trail":null,"at":":281","loc":"soma-runtime/src/executors/study.rs:281","sym":null,"note":["user code calls ctx.report(name, value, step)             :70","  → push metric, emit TrialMetric, ask the pruner"],"mark":null,"ref":null,"debt":null,"dyn":"TrialExecutor","children":[]},{"id":"n94","label":"match (outcome, pruned)","trail":null,"at":":287","loc":"soma-runtime/src/executors/study.rs:287","sym":null,"note":["(Ok(_), Some(..)) | (Ok(Pruned{..}), None) → Pruned","(Ok(Completed(m)), None)                   → Completed","(Err(e), _)                                → Failed"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n95","label":"sampler.record_result(…); best-trial check; StudyProgress","trail":null,"at":":335","loc":"soma-runtime/src/executors/study.rs:335","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n96","label":"save_study(study)","trail":"(! rewrites the whole file per trial)","at":":361","loc":"soma-runtime/src/executors/study.rs:361","sym":null,"note":[],"mark":"warn","ref":null,"debt":"D-64","dyn":null,"children":[]}]}]},"tail":[]},{"root":{"id":"n97","label":"PbtRunner::run(config, executor)","trail":null,"at":"executors/pbt.rs:97","loc":"soma-runtime/src/executors/pbt.rs:97","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n98","label":"rng_state = 42","trail":"(! hardcoded, no seed field)","at":":103","loc":"soma-runtime/src/executors/pbt.rs:103","sym":null,"note":[],"mark":"warn","ref":null,"debt":"D-49","dyn":null,"children":[]},{"id":"n99","label":"initialize_population","trail":null,"at":":195","loc":"soma-runtime/src/executors/pbt.rs:195","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n100","label":"for generation in 0..generations","trail":null,"at":":108","loc":"soma-runtime/src/executors/pbt.rs:108","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[{"id":"n101","label":"TRAIN each member","trail":"(! failure → warn, keeps stale state)","at":":116","loc":"soma-runtime/src/executors/pbt.rs:116","sym":null,"note":[],"mark":"warn","ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n102","label":"EVAL each member","trail":"(failure → NEG_INFINITY, counted)","at":":135","loc":"soma-runtime/src/executors/pbt.rs:135","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n103","label":"sort by fitness desc","trail":null,"at":":156","loc":"soma-runtime/src/executors/pbt.rs:156","sym":null,"note":[],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]},{"id":"n104","label":"evolve: exploit (truncation | binary tournament)","trail":null,"at":":215","loc":"soma-runtime/src/executors/pbt.rs:215","sym":null,"note":["then explore (perturbation | resample)"],"mark":null,"ref":null,"debt":null,"dyn":null,"children":[]}]}]},"tail":[]}]}],"junctions":[{"sym":"compute_node","traces":["b","d"]},{"sym":"EffectDriver::run","traces":["b","c"]},{"sym":"execute","traces":["a","b"]},{"sym":"execute_stream","traces":["b","d"]},{"sym":"output_key","traces":["b","d"]},{"sym":"store_output","traces":["b","d"]}]}</script>

<script>
(() => {
	const root = document.getElementById('cp');
	if (!root) return;
	const D = JSON.parse(document.getElementById('soma-call-paths').textContent);
	const tree = document.getElementById('cp-tree');
	const side = document.getElementById('cp-side');
	const tabs = document.getElementById('cp-tabs');
	const juncBar = document.getElementById('cp-junc');
	const onlyJunc = document.getElementById('cp-only-junc');
	const showNotes = document.getElementById('cp-notes');

	const juncSyms = new Map(D.junctions.map((j) => [j.sym, j.traces]));
	const flat = new Map(); // hop id -> { hop, trace }
	for (const t of D.traces) {
		const walk = (h) => { flat.set(h.id, { hop: h, trace: t }); (h.children ?? []).forEach(walk); };
		for (const b of t.blocks) walk(b.root);
	}
	let active = 'all';
	let selected = null;

	const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

	function rowHtml(hop, prefix, isLast, isRoot) {
		const lean = onlyJunc.checked;
		// Filtering to junctions leaves the surviving rows with tree prefixes
		// drawn for parents that are no longer on screen, which reads as
		// corruption. In that mode the shape is not the point — drop it.
		const branch = lean || isRoot ? '' : isLast ? '└─ ' : '├─ ';
		const drawn = lean ? '' : prefix;
		const isJ = hop.sym && juncSyms.has(hop.sym);
		if (lean && !isJ && !isRoot) return '';
		let badges = '';
		if (hop.mark && hop.mark.startsWith('P')) badges += `<span class="badge b-p">${hop.mark}</span>`;
		if (hop.mark === 'warn') badges += `<span class="badge b-warn">${hop.debt ?? '!'}</span>`;
		if (hop.dyn) badges += `<span class="badge b-dyn">dyn ${esc(hop.dyn)}</span>`;
		if (hop.ref) badges += `<span class="badge b-ref">→ (${hop.ref})</span>`;
		if (isJ) badges += `<span class="badge b-ref">in ${juncSyms.get(hop.sym).length} traces</span>`;
		const loc = hop.at ? `<span class="loc">  ${esc(hop.at)}</span>` : '';
		let html =
			`<div class="row${isJ ? ' junction' : ''}" data-id="${hop.id}">` +
			`<span class="tree">${drawn}${branch}</span><span class="lbl">${esc(hop.label)}</span>${loc}${badges}</div>`;
		const kidPrefix = isRoot ? '' : prefix + (isLast ? '   ' : '│  ');
		if (showNotes.checked && !lean) {
			for (const n of hop.note ?? []) html += `<div class="note">${esc(kidPrefix + '   ' + n)}</div>`;
		}
		const kids = hop.children ?? [];
		kids.forEach((k, i) => { html += rowHtml(k, kidPrefix, i === kids.length - 1, false); });
		return html;
	}

	function render() {
		const shown = D.traces.filter((t) => active === 'all' || t.id === active);
		tree.innerHTML = shown
			.map((t) => `<h4>(${t.id}) ${esc(t.title)}</h4>` + t.blocks.map((b) => rowHtml(b.root, '', true, true)).join(''))
			.join('');
		if (selected) markSelected();
	}

	function markSelected() {
		for (const r of tree.querySelectorAll('.row')) r.classList.toggle('sel', r.dataset.id === selected);
	}

	function show(id) {
		const entry = flat.get(id);
		if (!entry) return;
		const { hop, trace } = entry;
		selected = id;
		markSelected();
		const alsoIn = hop.sym && juncSyms.has(hop.sym) ? juncSyms.get(hop.sym).filter((x) => x !== trace.id) : [];
		side.innerHTML =
			`<h3>${esc(hop.label)}</h3>` +
			(hop.loc ? `<p><code>${esc(hop.loc)}</code></p>` : '<p class="empty">no source anchor</p>') +
			`<h4>Trace</h4><p>(${trace.id}) ${esc(trace.title)}</p>` +
			(alsoIn.length
				? `<h4>Also reached from</h4><ul>${alsoIn.map((x) => `<li>trace (${x})</li>`).join('')}</ul>` +
					`<p class="empty">A shared hop: the same code on more than one path.</p>`
				: '') +
			(hop.dyn
				? `<h4>Dynamic dispatch</h4><p>through <code>dyn ${esc(hop.dyn)}</code> — ` +
					`<a href="/soma/internals/graph/">see its implementors</a></p>`
				: '') +
			(hop.ref ? `<h4>Continues in</h4><p>trace (${hop.ref})</p>` : '') +
			(hop.debt
				? `<h4>Known debt</h4><p><a href="/soma/internals/debt/">${hop.debt}</a> — this hop is on the path of a documented problem.</p>`
				: '') +
			(hop.note?.length ? `<h4>Notes</h4><ul>${hop.note.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '');
	}

	tree.addEventListener('click', (ev) => {
		const r = ev.target.closest('.row');
		if (r) show(r.dataset.id);
	});

	for (const [id, label] of [['all', 'All five'], ...D.traces.map((t) => [t.id, `(${t.id})`])]) {
		const b = document.createElement('button');
		b.textContent = label;
		b.setAttribute('aria-pressed', String(active === id));
		b.onclick = () => {
			active = id;
			for (const x of tabs.children) x.setAttribute('aria-pressed', String(x === b));
			render();
		};
		tabs.appendChild(b);
	}
	onlyJunc.onchange = render;
	showNotes.onchange = render;

	juncBar.innerHTML =
		'<b>Junctions:</b> ' +
		D.junctions.map((j) => `<code>${j.sym}</code> → ${j.traces.map((t) => `(${t})`).join(' ')}`).join(' · ');

	render();
})();
</script>

## What the junctions mean

The three primitives — `output_key`, `compute_node`, `store_output` — appear in
trace **(b)** and trace **(d)**, and nowhere else. That is the entire design
claim of the streaming implementation: batch and stream execution share their
centre.

It is also the entire bug. Everything *around* those three is written twice, and
the copies have drifted — the stream path emits no cache events at all, so
`RunReader::cache_activity` reports zero cache activity for every streamed run.
That is [D-11](/soma/internals/debt/#d-11--the-stream-path-re-implements-run_node-and-has-drifted),
and this page is the shortest way to see why it happened: the shared part is
three function calls, and the duplicated part is everything you can see around
them in both trees at once.

The other junctions are structural rather than suspicious. `execute` joins (a)
to (b) because a forward is a plan walk. `EffectDriver` appears in (c) and in the
ownership spine because it is the one component a session holds only when the
graph contains steps.

## Reading the badges

| Badge | Meaning |
|---|---|
| `P1` `P2` `P3` | One of the three shared primitives — `output_key`, `compute_node`, `store_output` |
| `dyn T` | The call leaves through a trait object. This is where a static call-graph tool stops and the [architecture graph](/soma/internals/graph/) takes over |
| `D-nn` | The hop is on the path of a [documented problem](/soma/internals/debt/) |
| `→ (x)` | Control continues in another trace |
| `in N traces` | A junction: the same code reached from more than one entry point |

## How this is built

`docs/data/traces.json` is the source. `docs/scripts/gen-traces.mjs` renders it
twice — as the ASCII blocks in
[execution.md](/soma/internals/execution/#execution-traces) and as the JSON blob
behind this page:

```bash
cd docs && node scripts/gen-traces.mjs
```

They are generated from one file rather than written twice on purpose. Keeping
two hand-maintained copies of the same call chains is precisely the smell the
[debt register](/soma/internals/debt/#duplicated-logic) is about, and writing it
into the section that denounces it would be hard to defend.

The traces are **hand-authored, not extracted**, and that is the point. A static
analyser cannot tell you that `run_node_inner` is *the* branch distinguishing a
filter from a step, or that the `Suspend` arm's journal lookup is the whole
resume mechanism. What the generator adds is bookkeeping: it expands every short
`file:line` into a repo path, so `npm run check` verifies the 99 of them that
carry one. In their hand-written form the guard could see none.
