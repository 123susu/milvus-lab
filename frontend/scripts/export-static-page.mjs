import { copyFile, mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const serverEntry = resolve("dist/server/index.js");
const workerUrl = pathToFileURL(serverEntry);
workerUrl.searchParams.set("static-export", Date.now().toString());
const { default: worker } = await import(workerUrl.href);

const response = await worker.fetch(
  new Request("https://123susu.github.io/", {
    headers: {
      accept: "text/html",
      "x-forwarded-host": "123susu.github.io",
      "x-forwarded-proto": "https",
    },
  }),
  {
    ASSETS: {
      fetch: async () => new Response("Not found", { status: 404 }),
    },
  },
  {
    waitUntil() {},
    passThroughOnException() {},
  },
);

if (!response.ok) {
  throw new Error(`Static page render failed with HTTP ${response.status}`);
}

const outputDirectory = resolve("dist/client");
await mkdir(outputDirectory, { recursive: true });
await writeFile(
  resolve(outputDirectory, "index.html"),
  await response.text(),
  "utf8",
);
await copyFile(
  resolve(outputDirectory, "index.html"),
  resolve(outputDirectory, "404.html"),
);
await writeFile(resolve(outputDirectory, ".nojekyll"), "", "utf8");

console.log(`Static GitHub Pages output: ${outputDirectory}`);
