//! Headless content validation.
//!
//! Parses and cross-checks a content root, printing every problem it finds.
//! CI runs this so a dangling reference or a malformed file fails the build
//! rather than the game.
//!
//! Usage: `validate-content [CONTENT_ROOT]` (default: `assets/content`)

// A command-line tool's output is its purpose, unlike engine code where a
// stray print is a debugging leftover.
#![expect(clippy::print_stdout, reason = "this binary's output is its interface")]

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let root = match args.next() {
        Some(argument) if argument == "--help" || argument == "-h" => {
            println!("usage: validate-content [CONTENT_ROOT]");
            println!("  CONTENT_ROOT defaults to assets/content");
            return ExitCode::SUCCESS;
        }
        Some(argument) => PathBuf::from(argument),
        None => PathBuf::from("assets/content"),
    };
    if let Some(unexpected) = args.next() {
        eprintln!("unexpected argument: {unexpected}");
        eprintln!("usage: validate-content [CONTENT_ROOT]");
        return ExitCode::FAILURE;
    }

    match content_runtime::load(&root) {
        Ok(registry) => {
            println!("{}: {} definitions valid", root.display(), registry.len());
            // Itemised so the log says what was actually validated. A summary
            // that omits families hides a loader silently reading none of them.
            for (family, count) in [
                ("items", registry.items.len()),
                ("recipes", registry.recipes.len()),
                ("abilities", registry.abilities.len()),
                ("classes", registry.classes.len()),
                ("enemies", registry.enemies.len()),
                ("encounters", registry.encounters.len()),
                ("quests", registry.quests.len()),
                ("locations", registry.locations.len()),
                ("dialogues", registry.dialogues.len()),
                ("loot-tables", registry.loot_tables.len()),
                ("upgrades", registry.upgrades.len()),
                ("body-plans", registry.body_plans.len()),
                ("palettes", registry.palettes.len()),
                ("models", registry.models.len()),
                ("clips", registry.clips.len()),
                ("vfx", registry.vfx.len()),
                ("statuses", registry.statuses.len()),
                ("regions", registry.regions.len()),
                ("level-curves", registry.level_curves.len()),
            ] {
                println!("  {count:>4} {family}");
            }
            // The revision identity saves record (ADR-0013, ADR-0014), printed
            // so a CI log says which content a build actually validated.
            match registry.content_hash() {
                Ok(hash) => println!("content hash: {hash}"),
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Err(report) => {
            eprintln!("{}: {} problem(s)", root.display(), report.len());
            for problem in &report.problems {
                eprintln!("  {problem}");
            }
            ExitCode::FAILURE
        }
    }
}
