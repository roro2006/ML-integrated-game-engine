//! Engine path services.
//!
//! Resolves per-user config, data, save, cache, and log directories and the
//! read-only asset root without relying on the process working directory.
//! Resolution is a pure function of a captured [`PathEnvironment`] so both
//! the Linux and Windows rules are testable on any host.
//!
//! | Kind   | Linux (XDG)                           | Windows                         |
//! |--------|---------------------------------------|---------------------------------|
//! | config | `$XDG_CONFIG_HOME/<app>`              | `%APPDATA%\<app>\config`        |
//! | data   | `$XDG_DATA_HOME/<app>`                | `%LOCALAPPDATA%\<app>`          |
//! | saves  | `<data>/saves`                        | `<data>\saves`                  |
//! | cache  | `$XDG_CACHE_HOME/<app>`               | `%LOCALAPPDATA%\<app>\cache`    |
//! | logs   | `$XDG_STATE_HOME/<app>/logs`          | `%LOCALAPPDATA%\<app>\logs`     |
//!
//! XDG variables fall back to `$HOME/.config`, `.local/share`, `.cache`, and
//! `.local/state`; Windows variables fall back to `%USERPROFILE%\AppData`.
//!
//! The asset root is, in order: the `<APP_ID>_ASSET_ROOT` override variable;
//! otherwise the nearest `assets/` directory containing `content/` found at or
//! above the executable's directory (covers both installed layouts and
//! `target/<profile>/` during development).

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// Number of directories (starting with the executable's) searched for assets.
const ASSET_SEARCH_DEPTH: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostOs {
    Linux,
    Windows,
}

impl HostOs {
    pub const fn current() -> Self {
        if cfg!(windows) { Self::Windows } else { Self::Linux }
    }
}

impl fmt::Display for HostOs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Linux => "linux",
            Self::Windows => "windows",
        })
    }
}

/// The process inputs path resolution depends on.
#[derive(Debug, Clone)]
pub struct PathEnvironment {
    pub os: HostOs,
    pub vars: BTreeMap<String, OsString>,
    pub exe_dir: Option<PathBuf>,
}

const BASE_VARS: &[&str] = &[
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
];

impl PathEnvironment {
    /// Captures only the variables path resolution uses for `app_id`.
    pub fn from_process(app_id: &str) -> Self {
        let override_var = asset_root_override_var(app_id);
        let vars = BASE_VARS
            .iter()
            .copied()
            .chain(std::iter::once(override_var.as_str()))
            .filter_map(|key| std::env::var_os(key).map(|value| (key.to_owned(), value)))
            .collect();
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf));
        Self {
            os: HostOs::current(),
            vars,
            exe_dir,
        }
    }

    fn var_path(&self, key: &str) -> Option<PathBuf> {
        self.vars
            .get(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    }

    /// XDG requires absolute paths; relative values are ignored per the spec.
    ///
    /// Absoluteness is judged by POSIX rules rather than [`Path::is_absolute`],
    /// which answers for the platform this was *compiled* for rather than for
    /// [`PathEnvironment::os`]. In a Windows build `/home/u` reads as relative
    /// — Windows wants a drive letter or a UNC prefix — so resolving Linux
    /// paths from a Windows host discarded every XDG value and then reported
    /// `HOME` unset. Resolution takes [`HostOs`] as a value precisely so it can
    /// answer for either host from either host, and this has to honour that.
    ///
    /// XDG is a POSIX specification, so a leading `/` is the definition rather
    /// than an approximation of it.
    fn xdg_var(&self, key: &str) -> Option<PathBuf> {
        self.var_path(key)
            .filter(|path| path.to_string_lossy().starts_with('/'))
    }
}

#[derive(Debug)]
pub enum PathError {
    InvalidAppId(String),
    NoBaseDirectory { os: HostOs, variable: &'static str },
    CreateDirectory { path: PathBuf, source: io::Error },
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAppId(id) => {
                write!(f, "invalid app id {id:?}: use lowercase kebab-case [a-z0-9-]")
            }
            Self::NoBaseDirectory { os, variable } => {
                write!(
                    f,
                    "cannot resolve user directories on {os}: {variable} is not set"
                )
            }
            Self::CreateDirectory { path, source } => {
                write!(f, "cannot create directory {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for PathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateDirectory { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Resolved directories for one application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnginePaths {
    pub config: PathBuf,
    pub data: PathBuf,
    pub saves: PathBuf,
    pub cache: PathBuf,
    pub logs: PathBuf,
    /// Read-only shipped assets; `None` if no asset root was found.
    pub assets: Option<PathBuf>,
}

impl EnginePaths {
    pub fn resolve(app_id: &str, env: &PathEnvironment) -> Result<Self, PathError> {
        Self::resolve_with(app_id, env, &|path| path.is_dir())
    }

    /// Like [`resolve`](Self::resolve) with an injectable directory probe.
    pub fn resolve_with(
        app_id: &str,
        env: &PathEnvironment,
        is_dir: &dyn Fn(&Path) -> bool,
    ) -> Result<Self, PathError> {
        validate_app_id(app_id)?;
        let os = env.os;
        let (config, data, cache, logs) = match os {
            HostOs::Linux => {
                let home = env.xdg_var("HOME");
                let base = |xdg: &str, fallback: &[&str]| {
                    env.xdg_var(xdg)
                        .or_else(|| {
                            home.as_ref()
                                .map(|home| fallback.iter().fold(home.clone(), |path, part| path.join(part)))
                        })
                        .ok_or(PathError::NoBaseDirectory { os, variable: "HOME" })
                };
                (
                    base("XDG_CONFIG_HOME", &[".config"])?.join(app_id),
                    base("XDG_DATA_HOME", &[".local", "share"])?.join(app_id),
                    base("XDG_CACHE_HOME", &[".cache"])?.join(app_id),
                    base("XDG_STATE_HOME", &[".local", "state"])?
                        .join(app_id)
                        .join("logs"),
                )
            }
            HostOs::Windows => {
                let appdata = env.var_path("USERPROFILE").map(|profile| profile.join("AppData"));
                let roaming = env
                    .var_path("APPDATA")
                    .or_else(|| appdata.as_ref().map(|dir| dir.join("Roaming")))
                    .ok_or(PathError::NoBaseDirectory {
                        os,
                        variable: "APPDATA",
                    })?;
                let local = env
                    .var_path("LOCALAPPDATA")
                    .or_else(|| appdata.as_ref().map(|dir| dir.join("Local")))
                    .ok_or(PathError::NoBaseDirectory {
                        os,
                        variable: "LOCALAPPDATA",
                    })?
                    .join(app_id);
                (
                    roaming.join(app_id).join("config"),
                    local.clone(),
                    local.join("cache"),
                    local.join("logs"),
                )
            }
        };
        Ok(Self {
            saves: data.join("saves"),
            assets: find_asset_root(app_id, env, is_dir),
            config,
            data,
            cache,
            logs,
        })
    }

    /// Creates every writable directory (not the asset root).
    pub fn create_writable_dirs(&self) -> Result<(), PathError> {
        for dir in [&self.config, &self.data, &self.saves, &self.cache, &self.logs] {
            std::fs::create_dir_all(dir).map_err(|source| PathError::CreateDirectory {
                path: dir.clone(),
                source,
            })?;
        }
        Ok(())
    }
}

/// App IDs name directories on case-sensitive filesystems: lowercase kebab-case only.
pub fn validate_app_id(app_id: &str) -> Result<(), PathError> {
    let valid_chars = app_id
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    let valid = !app_id.is_empty()
        && app_id.len() <= 64
        && valid_chars
        && !app_id.starts_with('-')
        && !app_id.ends_with('-')
        && !app_id.contains("--");
    if valid {
        Ok(())
    } else {
        Err(PathError::InvalidAppId(app_id.to_owned()))
    }
}

/// Environment variable that overrides the asset root, e.g. `CINDERWAKE_ASSET_ROOT`.
pub fn asset_root_override_var(app_id: &str) -> String {
    let mut name: String = app_id
        .chars()
        .map(|c| if c == '-' { '_' } else { c.to_ascii_uppercase() })
        .collect();
    name.push_str("_ASSET_ROOT");
    name
}

fn find_asset_root(app_id: &str, env: &PathEnvironment, is_dir: &dyn Fn(&Path) -> bool) -> Option<PathBuf> {
    if let Some(path) = env.var_path(&asset_root_override_var(app_id)) {
        return is_dir(&path).then_some(path);
    }
    env.exe_dir
        .as_deref()?
        .ancestors()
        .take(ASSET_SEARCH_DEPTH)
        .map(|dir| dir.join("assets"))
        .find(|candidate| is_dir(&candidate.join("content")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(os: HostOs, vars: &[(&str, &str)], exe_dir: Option<&str>) -> PathEnvironment {
        PathEnvironment {
            os,
            vars: vars
                .iter()
                .map(|(k, v)| ((*k).to_owned(), OsString::from(v)))
                .collect(),
            exe_dir: exe_dir.map(PathBuf::from),
        }
    }

    fn no_dirs(_: &Path) -> bool {
        false
    }

    #[test]
    fn linux_uses_xdg_variables() {
        let env = env(
            HostOs::Linux,
            &[
                ("HOME", "/home/u"),
                ("XDG_CONFIG_HOME", "/cfg"),
                ("XDG_DATA_HOME", "/data"),
                ("XDG_CACHE_HOME", "/cache"),
                ("XDG_STATE_HOME", "/state"),
            ],
            None,
        );
        let paths = EnginePaths::resolve_with("cinderwake", &env, &no_dirs).unwrap();
        assert_eq!(paths.config, PathBuf::from("/cfg").join("cinderwake"));
        assert_eq!(paths.data, PathBuf::from("/data").join("cinderwake"));
        assert_eq!(
            paths.saves,
            PathBuf::from("/data").join("cinderwake").join("saves")
        );
        assert_eq!(paths.cache, PathBuf::from("/cache").join("cinderwake"));
        assert_eq!(
            paths.logs,
            PathBuf::from("/state").join("cinderwake").join("logs")
        );
        assert_eq!(paths.assets, None);
    }

    #[test]
    fn linux_falls_back_to_home_and_ignores_relative_xdg() {
        let env = env(
            HostOs::Linux,
            &[("HOME", "/home/u"), ("XDG_CONFIG_HOME", "relative/cfg")],
            None,
        );
        let paths = EnginePaths::resolve_with("cinderwake", &env, &no_dirs).unwrap();
        let home = PathBuf::from("/home/u");
        assert_eq!(paths.config, home.join(".config").join("cinderwake"));
        assert_eq!(paths.data, home.join(".local").join("share").join("cinderwake"));
        assert_eq!(paths.cache, home.join(".cache").join("cinderwake"));
        assert_eq!(
            paths.logs,
            home.join(".local").join("state").join("cinderwake").join("logs")
        );
    }

    #[test]
    fn linux_without_home_is_an_error() {
        let env = env(HostOs::Linux, &[("XDG_CONFIG_HOME", "/cfg")], None);
        assert!(matches!(
            EnginePaths::resolve_with("cinderwake", &env, &no_dirs),
            Err(PathError::NoBaseDirectory { variable: "HOME", .. })
        ));
    }

    #[test]
    fn windows_uses_appdata_and_localappdata() {
        let env = env(
            HostOs::Windows,
            &[
                ("APPDATA", "C:/Users/u/AppData/Roaming"),
                ("LOCALAPPDATA", "C:/Users/u/AppData/Local"),
            ],
            None,
        );
        let paths = EnginePaths::resolve_with("cinderwake", &env, &no_dirs).unwrap();
        let local = PathBuf::from("C:/Users/u/AppData/Local").join("cinderwake");
        assert_eq!(
            paths.config,
            PathBuf::from("C:/Users/u/AppData/Roaming")
                .join("cinderwake")
                .join("config")
        );
        assert_eq!(paths.data, local);
        assert_eq!(paths.saves, local.join("saves"));
        assert_eq!(paths.cache, local.join("cache"));
        assert_eq!(paths.logs, local.join("logs"));
    }

    #[test]
    fn windows_falls_back_to_userprofile() {
        let profile_only = env(HostOs::Windows, &[("USERPROFILE", "C:/Users/u")], None);
        let paths = EnginePaths::resolve_with("cinderwake", &profile_only, &no_dirs).unwrap();
        let appdata = PathBuf::from("C:/Users/u").join("AppData");
        assert_eq!(
            paths.config,
            appdata.join("Roaming").join("cinderwake").join("config")
        );
        assert_eq!(paths.data, appdata.join("Local").join("cinderwake"));

        let empty = env(HostOs::Windows, &[], None);
        assert!(matches!(
            EnginePaths::resolve_with("cinderwake", &empty, &no_dirs),
            Err(PathError::NoBaseDirectory {
                variable: "APPDATA",
                ..
            })
        ));
    }

    #[test]
    fn app_id_must_be_lowercase_kebab_case() {
        for valid in ["cinderwake", "rust-engine", "game2"] {
            assert!(validate_app_id(valid).is_ok(), "{valid}");
        }
        for invalid in [
            "",
            "Cinderwake",
            "my game",
            "a/b",
            "../x",
            "-a",
            "a-",
            "a--b",
            "a_b",
        ] {
            assert!(validate_app_id(invalid).is_err(), "{invalid}");
        }
        let env = env(HostOs::Linux, &[("HOME", "/home/u")], None);
        assert!(matches!(
            EnginePaths::resolve_with("../evil", &env, &no_dirs),
            Err(PathError::InvalidAppId(_))
        ));
    }

    #[test]
    fn asset_root_override_takes_precedence() {
        assert_eq!(asset_root_override_var("cinder-wake"), "CINDER_WAKE_ASSET_ROOT");
        let env = env(
            HostOs::Linux,
            &[("HOME", "/home/u"), ("CINDERWAKE_ASSET_ROOT", "/opt/assets")],
            Some("/ws/assets/content/x"),
        );
        let found = EnginePaths::resolve_with("cinderwake", &env, &|_| true).unwrap();
        assert_eq!(found.assets, Some(PathBuf::from("/opt/assets")));

        let missing = EnginePaths::resolve_with("cinderwake", &env, &no_dirs).unwrap();
        assert_eq!(
            missing.assets, None,
            "invalid override must not silently fall back"
        );
    }

    #[test]
    fn asset_root_found_above_executable() {
        let env = env(HostOs::Linux, &[("HOME", "/home/u")], Some("/ws/target/debug"));
        let expected = PathBuf::from("/ws").join("assets");
        let marker = expected.join("content");
        let paths = EnginePaths::resolve_with("cinderwake", &env, &|path| path == marker).unwrap();
        assert_eq!(paths.assets, Some(expected));
    }

    #[test]
    fn asset_search_depth_is_bounded() {
        let env = env(HostOs::Linux, &[("HOME", "/home/u")], Some("/ws/a/b/c/d/e"));
        let marker = PathBuf::from("/ws").join("assets").join("content");
        let paths = EnginePaths::resolve_with("cinderwake", &env, &|path| path == marker).unwrap();
        assert_eq!(paths.assets, None);
    }
}
