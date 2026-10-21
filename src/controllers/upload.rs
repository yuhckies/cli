use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use gzp::{ZBuilder, deflate::Gzip};
use ignore::WalkBuilder;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use synchronized_writer::SynchronizedWriter;
use tar::Builder;
use url::Url;

use crate::errors::RailwayError;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpResponse {
    pub deployment_id: String,
    pub url: String,
    pub logs_url: String,
    pub deployment_domain: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpErrorResponse {
    pub message: String,
}

/// Create a gzipped tarball from a project directory, respecting .railwayignore and .gitignore.
///
/// Reject symlinks resolving outside `project_path` before traversal and archiving.
///
/// `on_progress` is called with `(current, total)` after each entry is processed.
/// The first call is `(0, total)` once indexing is complete (before compression begins).
pub fn create_deploy_tarball(
    project_path: &Path,
    archive_prefix_path: &Path,
    no_gitignore: bool,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>> {
    let project_root = project_path
        .canonicalize()
        .context("Failed to resolve deployment directory")?;
    let bytes = Vec::<u8>::new();
    let arc = Arc::new(Mutex::new(bytes));
    let mut parz = ZBuilder::<Gzip, _>::new()
        .num_threads(num_cpus::get())
        .from_writer(SynchronizedWriter::new(arc.clone()));

    let ignore_paths = [
        std::ffi::OsStr::new(".git"),
        std::ffi::OsStr::new("node_modules"),
    ];

    {
        let mut archive = Builder::new(&mut parz);
        let mut builder = WalkBuilder::new(project_path);
        builder.add_custom_ignore_filename(".railwayignore");
        if no_gitignore {
            builder.git_ignore(false);
        }

        // Reject escaping links before descending into their targets. Retain the
        // lexical paths for archive names, including project-relative subpaths.
        let walk_root = project_root.clone();
        let walk_ignore_paths = ignore_paths;
        let (rejected_tx, rejected_rx) = std::sync::mpsc::channel();
        builder.filter_entry(move |entry| {
            if entry
                .path()
                .components()
                .any(|c| walk_ignore_paths.contains(&c.as_os_str()))
            {
                return false;
            }
            match resolve_deploy_path(entry.path(), &walk_root) {
                Ok(_) => true,
                Err(error) => {
                    let _ = rejected_tx.send(error);
                    false
                }
            }
        });
        let walker = builder.follow_links(true).hidden(false);
        let walked = walker.build().collect::<Vec<_>>();
        if let Ok(error) = rejected_rx.try_recv() {
            return Err(error);
        }
        let total = walked.len();
        on_progress(0, total);

        for (i, entry) in walked.into_iter().enumerate() {
            let entry = entry?;
            let path = entry.path();
            if path
                .components()
                .any(|c| ignore_paths.contains(&c.as_os_str()))
            {
                continue;
            }
            let stripped =
                std::path::PathBuf::from(".").join(path.strip_prefix(archive_prefix_path)?);
            // Resolve again after indexing and archive the checked target, not
            // the original symlink path.
            let resolved = resolve_deploy_path(path, &project_root)?;
            archive.append_path_with_name(resolved, stripped)?;
            on_progress(i + 1, total);
        }
    }
    parz.finish()?;

    let body = Arc::try_unwrap(arc)
        .map_err(|_| {
            anyhow::anyhow!("internal error: tarball buffer still has references after compression")
        })?
        .into_inner()
        .map_err(|e| anyhow::anyhow!("internal error: failed to unwrap tarball buffer: {e}"))?;
    Ok(body)
}

/// Resolve against the selected deployment directory, not the archive prefix:
/// the prefix may be an ancestor when deploying a monorepo subdirectory.
fn resolve_deploy_path(path: &Path, project_root: &Path) -> Result<PathBuf> {
    let resolved = path
        .canonicalize()
        .with_context(|| format!("Failed to resolve deployment path {}", path.display()))?;
    if !resolved.starts_with(project_root) {
        bail!(
            "Refusing to deploy {}: its target {} is outside the deployment directory {}. Remove the symlink or select a deployment directory containing its target.",
            path.display(),
            resolved.display(),
            project_root.display(),
        );
    }
    Ok(resolved)
}

/// Upload a deploy tarball to Railway's backboard API.
pub async fn upload_deploy_tarball(
    client: &Client,
    hostname: &str,
    project_id: &str,
    environment_id: &str,
    service_id: Option<&str>,
    message: Option<&str>,
    body: Vec<u8>,
) -> Result<UpResponse> {
    let mut url = Url::parse(&format!(
        "https://backboard.{hostname}/project/{project_id}/environment/{environment_id}/up",
    ))?;

    url.query_pairs_mut()
        .append_pair("serviceId", service_id.unwrap_or_default());

    if let Some(message) = message {
        url.query_pairs_mut().append_pair("message", message);
    }

    let body_len = body.len();
    let mut request = client
        .post(url.to_string())
        .header("Content-Type", "application/gzip");

    // Agent attribution: present only when an agent harness is driving the
    // CLI. Backboard threads these into the deployment-create event so an
    // agentic `up` deploy is distinguishable from a human CLI deploy
    // (`source = "CLI Agent"`), closing the signup → deploy funnel.
    if let Some((caller, agent_session_id)) = crate::telemetry::deploy_attribution() {
        request = request.header("x-railway-caller", caller);
        if let Some(agent_session_id) = agent_session_id {
            request = request.header("x-railway-agent-session", agent_session_id);
        }
    }

    let res = request.body(body).send().await?;

    let status = res.status();
    if status != 200 {
        if status == 400 {
            let body = res.json::<UpErrorResponse>().await?;
            return Err(RailwayError::FailedToUpload(body.message).into());
        }

        if status == 413 {
            let err = res.text().await?;
            return Err(RailwayError::FailedToUpload(format!(
                "Failed to upload code. File too large ({body_len} bytes): {err}",
            ))
            .into());
        }

        return Err(RailwayError::FailedToUpload(format!(
            "Failed to upload code with status code {status}"
        ))
        .into());
    }

    let mut response = res.json::<UpResponse>().await?;
    // Backboard can leave an empty query parameter at the end of these URLs.
    for url in [&mut response.url, &mut response.logs_url] {
        *url = url.trim_end_matches('&').to_owned();
    }
    Ok(response)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{collections::HashMap, fs, io::Read, os::unix::fs::symlink, path::PathBuf};

    fn archive_files(root: &Path, prefix: &Path, no_gitignore: bool) -> HashMap<PathBuf, Vec<u8>> {
        let body = create_deploy_tarball(root, prefix, no_gitignore, |_, _| {}).unwrap();
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(body.as_slice()));
        archive
            .entries()
            .unwrap()
            .map(|entry| {
                let mut entry = entry.unwrap();
                let path = entry.path().unwrap().into_owned();
                let mut contents = Vec::new();
                entry.read_to_end(&mut contents).unwrap();
                (path, contents)
            })
            .collect()
    }

    #[test]
    fn rejects_external_file_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir(&root).unwrap();
        fs::write(dir.path().join("secret"), "private credential").unwrap();
        for target in [PathBuf::from("../secret"), dir.path().join("secret")] {
            let link = root.join("innocent.txt");
            symlink(target, &link).unwrap();
            assert!(create_deploy_tarball(&root, &root, false, |_, _| {}).is_err());
            fs::remove_file(link).unwrap();
        }
    }

    #[test]
    fn rejects_external_directory_and_chained_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        let outside = dir.path().join("app-other");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret"), "private credential").unwrap();
        symlink(&outside, root.join("assets")).unwrap();
        assert!(create_deploy_tarball(&root, &root, false, |_, _| {}).is_err());
        fs::remove_file(root.join("assets")).unwrap();
        symlink(&outside, dir.path().join("intermediate")).unwrap();
        symlink("../intermediate/secret", root.join("asset")).unwrap();
        assert!(create_deploy_tarball(&root, &root, false, |_, _| {}).is_err());
    }

    #[test]
    fn selected_subdirectory_is_the_content_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir(&root).unwrap();
        fs::write(dir.path().join("secret"), "private credential").unwrap();
        symlink("../secret", root.join("asset")).unwrap();
        assert!(create_deploy_tarball(&root, dir.path(), false, |_, _| {}).is_err());
        assert!(create_deploy_tarball(&root, dir.path(), true, |_, _| {}).is_err());
    }

    #[test]
    fn preserves_internal_symlinks_and_project_relative_names() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.txt"), "source code").unwrap();
        symlink("src/main.txt", root.join("relative.txt")).unwrap();
        symlink(root.join("src/main.txt"), root.join("absolute.txt")).unwrap();
        symlink("src", root.join("assets")).unwrap();
        let files = archive_files(&root, dir.path(), false);
        for name in [
            "app/src/main.txt",
            "app/relative.txt",
            "app/absolute.txt",
            "app/assets/main.txt",
        ] {
            assert_eq!(files.get(Path::new(name)).unwrap(), b"source code");
        }
    }

    #[test]
    fn permits_symlinked_project_roots() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        let alias = dir.path().join("alias");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("main.txt"), "source code").unwrap();
        symlink(&root, &alias).unwrap();
        let files = archive_files(&alias, &alias, false);
        assert_eq!(files.get(Path::new("main.txt")).unwrap(), b"source code");
    }

    #[test]
    fn rechecks_symlinks_after_indexing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("main.txt"), "source code").unwrap();
        fs::write(dir.path().join("secret"), "private credential").unwrap();
        let link = root.join("asset");
        symlink("main.txt", &link).unwrap();
        let result = create_deploy_tarball(&root, &root, false, |current, _| {
            if current == 0 {
                fs::remove_file(&link).unwrap();
                symlink("../secret", &link).unwrap();
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn respects_gitignore_unless_explicitly_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(dir.path().join("secret"), "private credential").unwrap();
        fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
        symlink("../secret", root.join("ignored.txt")).unwrap();
        let files = archive_files(&root, &root, false);
        assert!(!files.contains_key(Path::new("ignored.txt")));
        assert!(create_deploy_tarball(&root, &root, true, |_, _| {}).is_err());
    }

    #[test]
    fn excluded_directory_symlinks_do_not_block_deployment() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        let outside = dir.path().join("outside");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret"), "private credential").unwrap();
        fs::write(root.join("main.txt"), "source code").unwrap();
        symlink(&outside, root.join("node_modules")).unwrap();
        let files = archive_files(&root, &root, true);
        assert!(!files.contains_key(Path::new("node_modules/secret")));
        assert_eq!(files.get(Path::new("main.txt")).unwrap(), b"source code");
    }

    #[test]
    fn ignored_external_symlinks_do_not_block_deployment() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir(&root).unwrap();
        fs::write(dir.path().join("secret"), "private credential").unwrap();
        fs::write(root.join(".railwayignore"), "ignored.txt\n").unwrap();
        fs::write(root.join("main.txt"), "source code").unwrap();
        symlink("../secret", root.join("ignored.txt")).unwrap();
        let files = archive_files(&root, &root, false);
        assert!(!files.contains_key(Path::new("ignored.txt")));
        assert_eq!(files.get(Path::new("main.txt")).unwrap(), b"source code");
    }
}
