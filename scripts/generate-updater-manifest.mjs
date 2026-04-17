#!/usr/bin/env node

import { Buffer } from "node:buffer";

const token = process.env.GITHUB_TOKEN || "";
const repository = process.env.GITHUB_REPOSITORY || "";
const releaseId = process.env.RELEASE_ID || "";
const releaseTag = (process.env.RELEASE_TAG || "").trim();

if (!token) throw new Error("GITHUB_TOKEN is required.");
if (!repository) throw new Error("GITHUB_REPOSITORY is required.");
if (!releaseId) throw new Error("RELEASE_ID is required.");
if (!releaseTag) throw new Error("RELEASE_TAG is required.");

const [owner, repo] = repository.split("/");
if (!owner || !repo) throw new Error(`Invalid GITHUB_REPOSITORY: ${repository}`);

const apiBase = `https://api.github.com/repos/${owner}/${repo}`;
const downloadBase = `https://github.com/${owner}/${repo}/releases/download/${encodeURIComponent(releaseTag)}`;

const headers = {
  Accept: "application/vnd.github+json",
  Authorization: `Bearer ${token}`,
  "X-GitHub-Api-Version": "2022-11-28",
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  const release = await githubJson(`${apiBase}/releases/${releaseId}`);
  const assets = Array.isArray(release.assets) ? release.assets : [];
  const byName = new Map(assets.map((asset) => [asset.name, asset]));
  const version = releaseTag.replace(/^v/, "");

  const platforms = {};
  const missing = [];

  const macAsset = byName.get("ChaosEngineAI_aarch64.app.tar.gz");
  const macSigAsset = byName.get("ChaosEngineAI_aarch64.app.tar.gz.sig");
  if (macAsset && macSigAsset) {
    const signature = await downloadAssetText(macSigAsset.id);
    const url = `${downloadBase}/${encodeURIComponent(macAsset.name)}`;
    platforms["darwin-aarch64"] = { signature, url };
    platforms["darwin-aarch64-app"] = { signature, url };
  } else {
    missing.push("macOS updater bundle/signature");
  }

  const linuxAssetName = `ChaosEngineAI_${version}_amd64.AppImage`;
  const linuxSigName = `${linuxAssetName}.sig`;
  const linuxAsset = byName.get(linuxAssetName);
  const linuxSigAsset = byName.get(linuxSigName);
  const linuxDebAssetName = `ChaosEngineAI_${version}_amd64.deb`;
  const linuxDebSigName = `${linuxDebAssetName}.sig`;
  const linuxDebAsset = byName.get(linuxDebAssetName);
  const linuxDebSigAsset = byName.get(linuxDebSigName);
  if (linuxAsset && linuxSigAsset) {
    const signature = await downloadAssetText(linuxSigAsset.id);
    const url = `${downloadBase}/${encodeURIComponent(linuxAsset.name)}`;
    platforms["linux-x86_64"] = { signature, url };
    platforms["linux-x86_64-appimage"] = { signature, url };
  } else {
    missing.push(`${linuxAssetName} / ${linuxSigName}`);
  }

  if (linuxDebAsset && linuxDebSigAsset) {
    const signature = await downloadAssetText(linuxDebSigAsset.id);
    const url = `${downloadBase}/${encodeURIComponent(linuxDebAsset.name)}`;
    platforms["linux-x86_64-deb"] = { signature, url };
  } else {
    missing.push(`${linuxDebAssetName} / ${linuxDebSigName}`);
  }

  const windowsAssetName = `ChaosEngineAI_${version}_x64-setup.exe`;
  const windowsSigName = `${windowsAssetName}.sig`;
  const windowsAsset = byName.get(windowsAssetName);
  const windowsSigAsset = byName.get(windowsSigName);
  if (windowsAsset && windowsSigAsset) {
    const signature = await downloadAssetText(windowsSigAsset.id);
    const url = `${downloadBase}/${encodeURIComponent(windowsAsset.name)}`;
    platforms["windows-x86_64"] = { signature, url };
    platforms["windows-x86_64-nsis"] = { signature, url };
  } else {
    missing.push(`${windowsAssetName} / ${windowsSigName}`);
  }

  if (missing.length) {
    throw new Error(`Cannot generate updater manifest; missing updater assets: ${missing.join(", ")}`);
  }

  const manifest = {
    version,
    notes: "",
    pub_date: new Date().toISOString(),
    platforms,
  };

  const payload = `${JSON.stringify(manifest, null, 2)}\n`;
  const latestAssetIds = assets.filter((asset) => asset.name === "latest.json").map((asset) => asset.id);
  for (const assetId of latestAssetIds) {
    await githubDelete(`${apiBase}/releases/assets/${assetId}`);
  }

  const uploadUrl = new URL(`https://uploads.github.com/repos/${owner}/${repo}/releases/${releaseId}/assets`);
  uploadUrl.searchParams.set("name", "latest.json");
  const response = await fetch(uploadUrl, {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: payload,
  });
  if (!response.ok) {
    throw new Error(`Failed to upload latest.json: ${response.status} ${await response.text()}`);
  }

  console.log("Uploaded combined latest.json");
  console.log(payload);
}

async function githubJson(url) {
  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`GitHub API request failed: ${response.status} ${await response.text()}`);
  }
  return response.json();
}

async function githubDelete(url) {
  const response = await fetch(url, {
    method: "DELETE",
    headers,
  });
  if (!response.ok) {
    throw new Error(`GitHub API delete failed: ${response.status} ${await response.text()}`);
  }
}

async function downloadAssetText(assetId) {
  const response = await fetch(`${apiBase}/releases/assets/${assetId}`, {
    headers: {
      ...headers,
      Accept: "application/octet-stream",
    },
    redirect: "follow",
  });
  if (!response.ok) {
    throw new Error(`Asset download failed: ${response.status} ${await response.text()}`);
  }
  const bytes = await response.arrayBuffer();
  return Buffer.from(bytes).toString("utf8").trim();
}
