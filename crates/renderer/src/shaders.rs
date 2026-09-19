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
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// The basic lit material: directional light plus ambient, no textures.
pub const LIT: &str = "lit.wgsl";

/// World-space debug lines: bounds boxes, axis gizmos, rays.
pub const DEBUG_LINE: &str = "debug_line.wgsl";

/// Every shader the binary ships with, keyed by file name.
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
    (
        LIT,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/lit.wgsl"
        )),
    ),
    (
        DEBUG_LINE,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/debug_line.wgsl"
        )),
    ),
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
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownShader { name } => write!(f, "unknown shader {name}"),
            Self::Read { path, message } => {
                write!(f, "cannot read shader {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for ShaderError {}

/// Resolves shader names to WGSL source, from disk in debug builds and from
/// the binary in release builds.
#[derive(Debug)]
pub struct ShaderLibrary {
    /// `None` disables hot reload: either the caller supplied no directory, or
    /// this is a release build. Release drops the directory here, at
    /// construction, so that every later decision has one obvious answer and
    /// cannot drift from the ADR.
    shader_dir: Option<PathBuf>,
    /// The last source successfully read from disk, per shader name. Used to
    /// detect edits; ordered so `changed` reports names deterministically.
    loaded: BTreeMap<String, String>,
}

impl ShaderLibrary {
    /// `shader_dir` is the directory holding the WGSL files, typically
    /// `<asset root>/shaders`. It is ignored in release builds.
    pub fn new(shader_dir: Option<PathBuf>) -> Self {
        Self {
            shader_dir: if cfg!(debug_assertions) { shader_dir } else { None },
            loaded: BTreeMap::new(),
        }
    }

    /// The directory shaders hot-reload from, or `None` when they are embedded.
    /// Reported at startup so it is obvious which mode the build is in.
    pub fn hot_reload_dir(&self) -> Option<&Path> {
        self.shader_dir.as_deref()
    }

    /// WGSL source for `name`, e.g. [`LIT`].
    ///
    /// Reads from disk on every call when hot reload is on: pipelines are built
    /// rarely, and caching here would serve stale text for the one workflow the
    /// directory exists to support. A read failure is returned rather than
    /// silently falling back to the embedded copy, so an edit with a typo in
    /// the file name is visible instead of quietly rendering the old shader.
    pub fn source(&mut self, name: &str) -> Result<Cow<'_, str>, ShaderError> {
        let embedded = embedded_source(name).ok_or_else(|| ShaderError::UnknownShader {
            name: name.to_owned(),
        })?;
        let path = match &self.shader_dir {
            Some(dir) => dir.join(name),
            None => return Ok(Cow::Borrowed(embedded)),
        };
        let source = fs::read_to_string(&path).map_err(|error| ShaderError::Read {
            path,
            message: error.to_string(),
        })?;
        // Borrowed from the cache rather than returned by value: pipeline
        // creation only needs to look at the text.
        let entry = self.loaded.entry(name.to_owned()).or_default();
        *entry = source;
        Ok(Cow::Borrowed(entry.as_str()))
    }

    /// Names whose files changed on disk since they were last read. Always
    /// empty in release builds, and empty for shaders [`source`] has never
    /// returned — a shader with no pipeline has nothing to rebuild.
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

        let mut names = Vec::with_capacity(edits.len());
        for (name, source) in edits {
            self.loaded.insert(name.clone(), source);
            names.push(name);
        }
        names
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
        assert_eq!(source, Cow::Borrowed(embedded_source(LIT).unwrap()));
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
            assert_ne!(source, Cow::Borrowed(embedded_source(LIT).unwrap()));
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
        assert_eq!(source, Cow::Borrowed(embedded_source(LIT).unwrap()));
        assert!(library.changed().is_empty());
    }
}
