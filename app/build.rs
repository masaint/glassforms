//! Which build this is, worked out when it is compiled: the branch (or "detached"), the commit, whether
//! the tree had changes not committed, and when it was built (UTC). Read by `src/build_info.rs`.
//!
//! Without git, or outside a checkout (a `git archive` snapshot), GF_BUILD_ID names the build instead;
//! with neither, the build is "unknown". The script runs again when HEAD moves, the branch's ref changes,
//! the index changes, or any source file does, so a new commit shows without a clean build.
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Days since 1970-01-01 as a civil date (proleptic Gregorian).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=GF_BUILD_ID");
    // (a source change can make the tree dirty without touching git's files)
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../core/src");
    println!("cargo:rerun-if-changed=build.rs");
    // The program's main thread: 32 MB of stack reserved instead of Windows' 1 MB. An unoptimised build
    // gives every branch of `run` (main.rs) its own slots for the engine and state it builds (60 and 20 KB
    // each) and its callees as much again, which passes 1 MB on the still path. Reserving costs address
    // space only; pages are committed as they are touched.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
            Ok("msvc") => println!("cargo:rustc-link-arg-bins=/STACK:33554432"),
            Ok("gnu") => println!("cargo:rustc-link-arg-bins=-Wl,--stack,33554432"),
            _ => {}
        }
    }

    let sha = git(&here, &["rev-parse", "--short", "HEAD"]);
    if let (Some(_), Some(git_dir), Some(common)) = (&sha, git(&here, &["rev-parse", "--absolute-git-dir"]), git(&here, &["rev-parse", "--path-format=absolute", "--git-common-dir"])) {
        let (git_dir, common) = (PathBuf::from(git_dir), PathBuf::from(common));
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
        println!("cargo:rerun-if-changed={}", common.join("packed-refs").display());
        if let Some(r) = git(&here, &["symbolic-ref", "-q", "HEAD"]) {
            println!("cargo:rerun-if-changed={}", common.join(r).display());
        }
    }
    let branch = sha.as_ref().and_then(|_| git(&here, &["rev-parse", "--abbrev-ref", "HEAD"]))
        .map(|b| if b == "HEAD" { "detached".to_owned() } else { b });
    let dirty = sha.as_ref().is_some_and(|_| git(&here, &["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty()));
    let id = match (sha, branch) {
        (Some(sha), Some(branch)) => format!("{branch}@{sha}{}", if dirty { "+dirty" } else { "" }),
        _ => std::env::var("GF_BUILD_ID").ok().filter(|s| !s.trim().is_empty()).map_or("unknown".to_owned(), |s| s.trim().to_owned()),
    };

    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (y, m, d) = civil(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    println!("cargo:rustc-env=GF_BUILD_ID={id}");
    println!("cargo:rustc-env=GF_BUILD_TIME={y:04}-{m:02}-{d:02} {:02}:{:02} UTC", s / 3600, s % 3600 / 60);
}
