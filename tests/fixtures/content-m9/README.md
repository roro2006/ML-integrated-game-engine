# M9 exit fixture

New content, added to a copy of `assets/content/` by the test
`m9_new_content_plays_to_completion` — a new quest, an enemy variant whose
ability applies a new status, an encounter in two waves, a new location and
speaker, and a consumable. Nothing here needs Rust: the test merges these
files beside the shipped ones (a family may be a directory of files) and plays
the quest through.
