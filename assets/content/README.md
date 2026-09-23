# assets/content/

Human-authored RON content: the shipped Cinderwake slice. Filenames and IDs use
lowercase kebab-case. Schemas live in `game_core::content`; the loader and its
cross-family reference checks live in `content_runtime`.

Validate after any change here — a dangling reference is a build failure, not a
runtime surprise:

```bash
cargo run -p content_runtime --bin validate-content -- assets/content
```

In a development run (`--slice --dev`), **F5** reloads this folder at the next
safe point — no fight, no panel open. Invalid content, or content that would
orphan the running world, is refused and nothing changes. Models, effects,
and name meshes reload on restart.

Deliberately broken content lives in `tests/fixtures/content-invalid/`, which the
loader's own tests use to prove bad content is rejected. Do not "fix" it.
`tests/fixtures/unfinishable/` does the same for quests (M9): a small valid
world, `finishable/`, and one copy per way a quest can load yet never be
finished, each broken in exactly one place.

Since M9 the loader refuses a quest no player could finish: every objective's
NPC, clue, choice, fight, item, or place must exist somewhere a door path
reaches from where a conversation gives the quest. Optional text for players:

- `QuestNode.text` — what the objective tracker says for that step; without
  it the tracker words the objective itself ("Talk to Warden Sela").
- `Clue(clue, text)` — what examining the clue shows.
- `Choice(choice, prompt)` — what the confirm panel asks before settling.
- `statuses.ron` — what each status means: a name, a chip colour, and
  bounded modifiers. Every `ApplyStatus` must name one.
- `regions.ron` — each region's name and biome; every location names one.
- `ClassDef.combat` — a class's own swing, stance, and roll; fields left out
  are the slice's defaults.
- `ItemDef.on_use` — what using a consumable does to the user, and how long
  using it takes.
- `DialogueDef.conditions` and `topic` — a conversation opens only while its
  conditions hold; when a speaker has several open, the player picks by
  `topic`, which the loader requires on every conversation of such a speaker.

The presence of this directory marks the asset root: the engine searches upward from
the executable for `assets/content/`, or uses `<APP_ID>_ASSET_ROOT` when set.
