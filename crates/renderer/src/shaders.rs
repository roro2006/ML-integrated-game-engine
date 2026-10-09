//! Shader source loading.
//!
//! Debug builds read WGSL from the asset root so shaders hot-reload; release
//! builds embed the sources at compile time (ADR-0011).
//!
//! The split exists because the two builds have opposite priorities. During
//! renderer iteration, re-reading a file beats restarting the game to see a
//! one-line change. In a shipped build, a missing or edited file must not be
//! able to break rendering at all, so the sources travel inside the binary and
//! the hot-reload directory is ignored entirely.
//!
//! Nothing here panics. A failed read is a typed error, because a hot-reload
//! failure is an ordinary event — a half-saved file, a typo, a deleted
//! shader — and the right response is to report it on the debug overlay and
//! keep drawing with the last pipeline that compiled, not to take the engine
//! down mid-session.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// The basic lit material: directional light plus ambient, no textures.
pub const LIT: &str = "lit.wgsl";

/// World-space debug lines: bounds boxes, axis gizmos, rays.
pub const DEBUG_LINE: &str = "debug_line.wgsl";

/// The fullscreen pass that tonemaps the HDR scene onto the swapchain.
pub(crate) const TONEMAP: &str = "tonemap.wgsl";

/// Shared colour maths: tonemapping curves. Included, never compiled alone.
pub(crate) const COLOUR: &str = "colour.wgsl";

macro_rules! shader_file {
    ($name:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/",
            $name
        ))
    };
}

/// Every shader the binary ships with, keyed by file name, snippets included.
///
/// This table is the authority on which shaders exist, in both build flavours.
/// A name missing from it is an error even when a matching file sits in the
/// hot-reload directory, so a shader cannot be developed in debug and then be
/// absent from the release binary.
///
/// `include_str!` resolves relative to this source file, so the path is
/// anchored at the crate manifest instead — that survives the file moving
/// within `src/`.
const EMBEDDED: &[(&str, &str)] = &[
    (LIT, shader_file!("lit.wgsl")),
    (DEBUG_LINE, shader_file!("debug_line.wgsl")),
    (TONEMAP, shader_file!("tonemap.wgsl")),
    (COLOUR, shader_file!("colour.wgsl")),
];

fn embedded_source(name: &str) -> Option<&'static str> {
    EMBEDDED
        .iter()
        .find(|(shader, _)| *shader == name)
        .map(|(_, source)| *source)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderError {
    /// Not a shader this binary ships; see `EMBEDDED`.
    UnknownShader { name: String },
    /// The file could not be read. The message is the OS error, flattened to a
    /// `String` so callers can compare and log errors without an `io` import.
    Read { path: PathBuf, message: String },
    /// An `#include` names itself, directly or through another file.
    IncludeCycle { chain: Vec<String> },
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownShader { name } => write!(f, "unknown shader {name}"),
            Self::Read { path, message } => {
                write!(f, "cannot read shader {}: {message}", path.display())
            }
            Self::IncludeCycle { chain } => write!(f, "shader include cycle: {}", chain.join(" -> ")),
        }
    }
}

impl std::error::Error for ShaderError {}

/// A run of consecutive lines in expanded source that came from one place in
/// one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Segment {
    /// First line of the run in the expanded source, from 1.
    pub(crate) expanded_line: usize,
    pub(crate) file: String,
    /// The same line in `file`, from 1.
    pub(crate) file_line: usize,
    pub(crate) lines: usize,
}

/// Source with its `#include` lines replaced by the files they name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expanded {
    pub(crate) text: String,
    pub(crate) segments: Vec<Segment>,
    /// Every file the text came from, the root first.
    pub(crate) files: Vec<String>,
}

impl Expanded {
    /// The file and line an expanded line came from.
    pub(crate) fn locate(&self, expanded_line: usize) -> Option<(&str, usize)> {
        self.segments
            .iter()
            .find(|segment| {
                (segment.expanded_line..segment.expanded_line + segment.lines).contains(&expanded_line)
            })
            .map(|segment| {
                (
                    segment.file.as_str(),
                    segment.file_line + expanded_line - segment.expanded_line,
                )
            })
    }

    /// Appends the original file and line to a compiler message that names an
    /// expanded line as `wgsl:LINE:COLUMN`, which is how naga reports them.
    pub(crate) fn annotate(&self, message: &str) -> String {
        let located = message.match_indices("wgsl:").find_map(|(index, marker)| {
            let digits: String = message[index + marker.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            let line = digits.parse().ok()?;
            self.locate(line)
        });
        match located {
            Some((file, line)) => format!("{message}\n(that line is {file}:{line})"),
            None => message.to_owned(),
        }
    }
}

/// The file a line includes, if it is an include directive:
/// `#include "colour.wgsl"`, alone on its line.
fn include_target(line: &str) -> Option<&str> {
    line.trim()
        .strip_prefix("#include")?
        .trim()
        .strip_prefix('"')?
        .strip_suffix('"')
}

/// Expands the includes in `root`, reading files through `read`. Line endings
/// come out as `\n` whatever the files used.
///
/// Each file is included at most once however many files include it, so
/// shared snippets need no guards. A cycle is an error naming the chain.
pub(crate) fn expand(
    root: &str,
    read: &mut dyn FnMut(&str) -> Result<String, ShaderError>,
) -> Result<Expanded, ShaderError> {
    let mut expanded = Expanded {
        text: String::new(),
        segments: Vec::new(),
        files: Vec::new(),
    };
    let mut stack = Vec::new();
    expand_into(root, read, &mut stack, &mut expanded)?;
    Ok(expanded)
}

fn expand_into(
    name: &str,
    read: &mut dyn FnMut(&str) -> Result<String, ShaderError>,
    stack: &mut Vec<String>,
    out: &mut Expanded,
) -> Result<(), ShaderError> {
    if stack.iter().any(|open| open == name) {
        let mut chain = stack.clone();
        chain.push(name.to_owned());
        return Err(ShaderError::IncludeCycle { chain });
    }
    if out.files.iter().any(|done| done == name) {
        return Ok(());
    }
    let source = read(name)?;
    out.files.push(name.to_owned());
    stack.push(name.to_owned());

    let mut run_start: Option<(usize, usize)> = None;
    for (index, line) in source.lines().enumerate() {
        let file_line = index + 1;
        if let Some(target) = include_target(line) {
            close_run(out, name, &mut run_start);
            expand_into(target, read, stack, out)?;
            continue;
        }
        let expanded_line = out.text.lines().count() + 1;
        run_start.get_or_insert((expanded_line, file_line));
        out.text.push_str(line);
        out.text.push('\n');
    }
    close_run(out, name, &mut run_start);
    stack.pop();
    Ok(())
}

fn close_run(out: &mut Expanded, file: &str, run_start: &mut Option<(usize, usize)>) {
    if let Some((expanded_line, file_line)) = run_start.take() {
        let lines = out.text.lines().count() + 1 - expanded_line;
        out.segments.push(Segment {
            expanded_line,
            file: file.to_owned(),
            file_line,
            lines,
        });
    }
}

/// Resolves shader names to WGSL source, from disk in debug builds and from
/// the binary in release builds.
#[derive(Debug)]
pub struct ShaderLibrary {
    /// `None` disables hot reload: either the caller supplied no directory, or
    /// this is a release build. Release drops the directory here, at
    /// construction, so that every later decision has one obvious answer and
    /// cannot drift from the ADR.
    shader_dir: Option<PathBuf>,
    /// The last source successfully read from disk, per file name. Used to
    /// detect edits; ordered so `changed` reports names deterministically.
    loaded: BTreeMap<String, String>,
    /// For each shader returned by `source`, every file it was built from.
    dependencies: BTreeMap<String, Vec<String>>,
}

impl ShaderLibrary {
    /// `shader_dir` is the directory holding the WGSL files, typically
    /// `<asset root>/shaders`. It is ignored in release builds.
    pub fn new(shader_dir: Option<PathBuf>) -> Self {
        Self {
            shader_dir: if cfg!(debug_assertions) { shader_dir } else { None },
            loaded: BTreeMap::new(),
            dependencies: BTreeMap::new(),
        }
    }

    /// The directory shaders hot-reload from, or `None` when they are embedded.
    /// Reported at startup so it is obvious which mode the build is in.
    pub fn hot_reload_dir(&self) -> Option<&Path> {
        self.shader_dir.as_deref()
    }

    /// WGSL source for `name`, e.g. [`LIT`], with its includes expanded.
    ///
    /// Reads from disk on every call when hot reload is on: pipelines are built
    /// rarely, and caching here would serve stale text for the one workflow the
    /// directory exists to support. A read failure is returned rather than
    /// silently falling back to the embedded copy, so an edit with a typo in
    /// the file name is visible instead of quietly rendering the old shader.
    pub fn source(&mut self, name: &str) -> Result<Cow<'_, str>, ShaderError> {
        self.expanded(name).map(|expanded| Cow::Owned(expanded.text))
    }

    /// Like [`source`](Self::source), keeping the line map so compiler errors
    /// can name the file a line came from.
    pub(crate) fn expanded(&mut self, name: &str) -> Result<Expanded, ShaderError> {
        let dir = self.shader_dir.clone();
        let loaded = &mut self.loaded;
        let mut read = |file: &str| -> Result<String, ShaderError> {
            let embedded = embedded_source(file).ok_or_else(|| ShaderError::UnknownShader {
                name: file.to_owned(),
            })?;
            let Some(dir) = &dir else {
                return Ok(embedded.to_owned());
            };
            let path = dir.join(file);
            let source = fs::read_to_string(&path).map_err(|error| ShaderError::Read {
                path,
                message: error.to_string(),
            })?;
            loaded.insert(file.to_owned(), source.clone());
            Ok(source)
        };
        let expanded = expand(name, &mut read)?;
        self.dependencies.insert(name.to_owned(), expanded.files.clone());
        Ok(expanded)
    }

    /// Shaders that need rebuilding because a file they were built from
    /// changed on disk since it was last read: the shader's own file or any
    /// file it includes. Always empty in release builds, and empty for shaders
    /// [`source`] has never returned — a shader with no pipeline has nothing to
    /// rebuild.
    ///
    /// Compares content rather than modification time on purpose: mtime
    /// granularity is coarse on some filesystems, so two saves within the same
    /// tick would be missed, and `touch` with no edit would force a needless
    /// pipeline rebuild. These files are a few kilobytes and this only runs in
    /// debug builds.
    ///
    /// A file that has become unreadable is not reported: the caller keeps the
    /// pipeline it has, and the next [`source`] call surfaces the error.
    ///
    /// [`source`]: Self::source
    pub fn changed(&mut self) -> Vec<String> {
        let Some(dir) = &self.shader_dir else {
            return Vec::new();
        };
        let edits: Vec<(String, String)> = self
            .loaded
            .iter()
            .filter_map(|(name, loaded)| match fs::read_to_string(dir.join(name)) {
                Ok(source) if &source != loaded => Some((name.clone(), source)),
                _ => None,
            })
            .collect();

        let mut edited = BTreeSet::new();
        for (name, source) in edits {
            self.loaded.insert(name.clone(), source);
            edited.insert(name);
        }
        self.dependencies
            .iter()
            .filter(|(_, files)| files.iter().any(|file| edited.contains(file)))
            .map(|(shader, _)| shader.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A unique directory under the system temp dir, removed on drop. Avoids a
    /// `tempfile` dependency for what is a handful of test files.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "renderer-shaders-{}-{nanos}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("temp dir is creatable");
            Self(path)
        }

        fn write(&self, name: &str, contents: &str) {
            fs::write(self.0.join(name), contents).expect("temp file is writable");
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// `text` with `\n` line endings, as expansion produces whatever the
    /// checkout used.
    fn normalised(text: &str) -> String {
        text.lines().map(|line| format!("{line}\n")).collect()
    }

    /// Reads from an in-memory set of files, for include tests.
    fn files<'a>(entries: &'a [(&'a str, &'a str)]) -> impl FnMut(&str) -> Result<String, ShaderError> + 'a {
        move |name: &str| {
            entries
                .iter()
                .find(|(file, _)| *file == name)
                .map(|(_, source)| (*source).to_owned())
                .ok_or_else(|| ShaderError::UnknownShader {
                    name: name.to_owned(),
                })
        }
    }

    #[test]
    fn an_include_is_replaced_by_the_file_it_names() {
        let mut read = files(&[
            ("main.wgsl", "a\n#include \"lib.wgsl\"\nb\n"),
            ("lib.wgsl", "l1\nl2\n"),
        ]);
        let expanded = expand("main.wgsl", &mut read).expect("valid includes");
        assert_eq!(expanded.text, "a\nl1\nl2\nb\n");
        assert_eq!(expanded.files, ["main.wgsl", "lib.wgsl"]);
    }

    #[test]
    fn a_file_included_twice_appears_once() {
        let mut read = files(&[
            ("main.wgsl", "#include \"a.wgsl\"\n#include \"b.wgsl\"\nmain\n"),
            ("a.wgsl", "#include \"shared.wgsl\"\na\n"),
            ("b.wgsl", "#include \"shared.wgsl\"\nb\n"),
            ("shared.wgsl", "shared\n"),
        ]);
        let expanded = expand("main.wgsl", &mut read).expect("valid includes");
        assert_eq!(expanded.text, "shared\na\nb\nmain\n");
    }

    #[test]
    fn an_include_cycle_is_an_error_naming_the_chain() {
        let mut read = files(&[
            ("main.wgsl", "#include \"a.wgsl\"\n"),
            ("a.wgsl", "#include \"b.wgsl\"\n"),
            ("b.wgsl", "#include \"a.wgsl\"\n"),
        ]);
        assert_eq!(
            expand("main.wgsl", &mut read),
            Err(ShaderError::IncludeCycle {
                chain: vec![
                    "main.wgsl".to_owned(),
                    "a.wgsl".to_owned(),
                    "b.wgsl".to_owned(),
                    "a.wgsl".to_owned()
                ],
            })
        );
    }

    #[test]
    fn an_unknown_include_is_reported() {
        let mut read = files(&[("main.wgsl", "#include \"missing.wgsl\"\n")]);
        assert!(matches!(
            expand("main.wgsl", &mut read),
            Err(ShaderError::UnknownShader { name }) if name == "missing.wgsl"
        ));
    }

    #[test]
    fn expanded_lines_map_back_to_their_file_and_line() {
        let mut read = files(&[
            ("main.wgsl", "m1\n#include \"lib.wgsl\"\nm3\nm4\n"),
            ("lib.wgsl", "l1\nl2\n"),
        ]);
        let expanded = expand("main.wgsl", &mut read).expect("valid includes");
        // Expanded: m1, l1, l2, m3, m4.
        assert_eq!(expanded.locate(1), Some(("main.wgsl", 1)));
        assert_eq!(expanded.locate(2), Some(("lib.wgsl", 1)));
        assert_eq!(expanded.locate(3), Some(("lib.wgsl", 2)));
        assert_eq!(
            expanded.locate(4),
            Some(("main.wgsl", 3)),
            "the include line itself is skipped"
        );
        assert_eq!(expanded.locate(5), Some(("main.wgsl", 4)));
        assert_eq!(expanded.locate(6), None);
    }

    #[test]
    fn a_compiler_message_gains_the_original_location() {
        let mut read = files(&[
            ("main.wgsl", "m1\n#include \"lib.wgsl\"\n"),
            ("lib.wgsl", "l1\nbroken\n"),
        ]);
        let expanded = expand("main.wgsl", &mut read).expect("valid includes");
        let message = "error: expected expression\n  ┌─ wgsl:3:1\n";
        assert!(expanded.annotate(message).ends_with("(that line is lib.wgsl:2)"));
        assert_eq!(expanded.annotate("no location"), "no location");
    }

    #[test]
    fn the_tonemap_shader_carries_its_colour_maths() {
        let mut library = ShaderLibrary::new(None);
        let source = library.source(TONEMAP).expect("embedded");
        assert!(
            source.lines().all(|line| include_target(line).is_none()),
            "every include is expanded"
        );
        assert!(source.contains("fn aces_fitted"));
        assert!(source.contains("fn agx"));
        assert!(source.contains("fn fs_main"));
    }

    #[test]
    fn embedded_source_is_a_complete_shader() {
        let source = embedded_source(LIT).expect("lit.wgsl is embedded");
        assert!(!source.is_empty());
        assert!(source.contains("vs_main"), "missing vertex entry point");
        assert!(source.contains("fs_main"), "missing fragment entry point");
        assert!(
            source.contains("@group(0) @binding(0)"),
            "missing globals binding"
        );
        assert!(
            source.contains("@group(1) @binding(0)"),
            "missing per-draw binding"
        );
    }

    /// The debug shader binds group 0 and nothing else: its vertices arrive in
    /// world space, so a per-draw group would have nothing to carry.
    #[test]
    fn the_debug_line_shader_is_embedded_and_binds_globals_only() {
        let source = embedded_source(DEBUG_LINE).expect("debug_line.wgsl is embedded");
        assert!(source.contains("vs_main"), "missing vertex entry point");
        assert!(source.contains("fs_main"), "missing fragment entry point");
        assert!(
            source.contains("@group(0) @binding(0)"),
            "missing globals binding"
        );
        assert!(
            !source.contains("@group(1)"),
            "the debug pass must not declare a per-draw bind group"
        );
    }

    #[test]
    fn without_a_shader_dir_the_embedded_source_is_used() {
        let mut library = ShaderLibrary::new(None);
        assert_eq!(library.hot_reload_dir(), None);
        let source = library.source(LIT).expect("embedded source is always available");
        assert_eq!(source, normalised(embedded_source(LIT).unwrap()));
        assert!(library.changed().is_empty());
    }

    #[test]
    fn unknown_names_are_rejected_without_panicking() {
        let mut library = ShaderLibrary::new(None);
        let error = library.source("no-such.wgsl").expect_err("unknown shader");
        assert_eq!(
            error,
            ShaderError::UnknownShader {
                name: "no-such.wgsl".to_owned()
            }
        );
        assert!(error.to_string().contains("no-such.wgsl"));
    }

    /// Debug-only: release builds ignore the directory by design, so these
    /// assertions would be false there.
    #[cfg(debug_assertions)]
    mod hot_reload {
        use super::*;

        #[test]
        fn a_file_on_disk_takes_precedence_over_the_embedded_source() {
            let dir = TempDir::new();
            dir.write(LIT, "// edited\nfn vs_main() {}\n");
            let mut library = ShaderLibrary::new(Some(dir.0.clone()));

            assert_eq!(library.hot_reload_dir(), Some(dir.0.as_path()));
            let source = library.source(LIT).expect("the file is readable");
            assert!(source.starts_with("// edited"));
            assert_ne!(source, normalised(embedded_source(LIT).unwrap()));
        }

        #[test]
        fn a_missing_file_is_an_error_not_a_panic() {
            let dir = TempDir::new();
            let mut library = ShaderLibrary::new(Some(dir.0.clone()));

            let error = library.source(LIT).expect_err("no file was written");
            match error {
                ShaderError::Read { path, message } => {
                    assert_eq!(path, dir.0.join(LIT));
                    assert!(!message.is_empty());
                }
                other => panic!("expected a read error, got {other:?}"),
            }
        }

        #[test]
        fn changed_reports_edited_shaders_once() {
            let dir = TempDir::new();
            dir.write(LIT, "// first\n");
            let mut library = ShaderLibrary::new(Some(dir.0.clone()));
            library.source(LIT).expect("the file is readable");

            // Nothing has been edited yet.
            assert!(library.changed().is_empty());

            // Distinctly different content, so detection never depends on
            // filesystem timestamp granularity.
            dir.write(LIT, "// second, and materially longer than the first\n");
            assert_eq!(library.changed(), vec![LIT.to_owned()]);
            // Reported once: the caller has rebuilt by now.
            assert!(library.changed().is_empty());
            assert!(
                library
                    .source(LIT)
                    .expect("still readable")
                    .starts_with("// second")
            );
        }

        #[test]
        fn editing_an_include_rebuilds_every_shader_that_uses_it() {
            let dir = TempDir::new();
            dir.write(
                TONEMAP,
                "#include \"colour.wgsl\"
// tonemap
",
            );
            dir.write(
                COLOUR,
                "// colour, first
",
            );
            dir.write(
                LIT, "// lit
",
            );
            let mut library = ShaderLibrary::new(Some(dir.0.clone()));
            library.source(TONEMAP).expect("readable");
            library.source(LIT).expect("readable");

            dir.write(
                COLOUR,
                "// colour, second and materially longer
",
            );
            assert_eq!(
                library.changed(),
                vec![TONEMAP.to_owned()],
                "the includer rebuilds; an unrelated shader does not"
            );
        }

        #[test]
        fn changed_ignores_shaders_that_were_never_loaded_or_became_unreadable() {
            let dir = TempDir::new();
            dir.write(LIT, "// first\n");
            let mut library = ShaderLibrary::new(Some(dir.0.clone()));

            // Never read through `source`, so there is no pipeline to rebuild.
            assert!(library.changed().is_empty());

            library.source(LIT).expect("the file is readable");
            fs::remove_file(dir.0.join(LIT)).expect("temp file is removable");
            assert!(library.changed().is_empty());
            assert!(library.source(LIT).is_err());
        }
    }

    /// Release builds must never touch the filesystem, whatever the caller passes.
    #[cfg(not(debug_assertions))]
    #[test]
    fn release_builds_ignore_the_shader_dir() {
        let dir = TempDir::new();
        dir.write(LIT, "// edited\n");
        let mut library = ShaderLibrary::new(Some(dir.0.clone()));

        assert_eq!(library.hot_reload_dir(), None);
        let source = library.source(LIT).expect("embedded source is always available");
        assert_eq!(source, normalised(embedded_source(LIT).unwrap()));
        assert!(library.changed().is_empty());
    }
}
