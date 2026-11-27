# Learning views

The KV cache note is the first pilot. Its existing Chinese and English URLs, Markdown source, note ID, search index and full article stay unchanged. `learning_views.py` adds authored views when assembling this note; other notes do not load the view assets.

- **Quick read:** one small example, the mechanism, its costs and assumptions.
- **Visual lab:** the same example, with manual steps and optional playback. Counts describe causal attention scores per layer/head, not measured speedups. Under the lanes, a short `prefill` / `decode` sketch in Python marks the lines the current step runs and says what the cache holds afterwards; each line names the steps it belongs to in `data-code-steps`. One check question follows; its answer is shown in place and nothing is stored.
- **Full note:** the original article, with all examples, derivations, references and anchors.

An unfragmented URL opens Quick read when JavaScript works. `?view=lab&step=1` links to a particular lab state; `?view=full` opens the complete article. Any existing fragment opens Full note, including search results and saved reading positions. View changes use browser history; steps replace the current history entry. Language links keep the selected view and step.

The full article is visible in server-rendered HTML. The tab controls and shorter views are hidden until initialization succeeds. Without JavaScript, readers get the full note. Printing also uses the full article. Reduced-motion readers can step manually; playback is never automatic and stops when hidden or offscreen.

These views do not write learning progress. They use no storage keys and do not introduce completion scores. Quick and lab headings have no heading IDs, so visiting them cannot replace a saved full-note heading. Bookmarks, reading history, backups and legacy imports continue to use the existing note identity and storage implementation.

To extend the pilot, author and review both languages, preserve the complete note, scope assets to supported pages, and test the mathematics separately from the interface. Only offer a lab when there is a real explanation to explore. Do not add empty tabs or turn a shorter view into a replacement for the full article.

Run the Python and Node tests, then check both languages, small screens, large text, both themes, keyboard navigation, back/forward, old fragments, JavaScript-disabled loading, print, and backup round trips in isolated browser contexts. The public pages must not contain personal learning records or private planning material.
