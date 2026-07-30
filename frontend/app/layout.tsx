import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:4173";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const siteBasePath = (
    process.env.NEXT_PUBLIC_SITE_BASE_PATH ?? "/"
  ).replace(/\/?$/, "/");
  const title = "MilvusTune · Vector Performance Lab";
  const description = process.env.NEXT_PUBLIC_READ_ONLY_DEMO === "true"
    ? "公开展示 VectorDBBench 的 Milvus CPU 索引实验结果，对比构建耗时、P99、Recall 与索引内存。"
    : "从本地 SQLite 汇总 VectorDBBench、QueryNode CPU 与 Vector Index 内存，快速对比 Milvus HNSW 实验。";

  return {
    title,
    description,
    openGraph: { title, description, images: [{ url: `${origin}${siteBasePath}og.png`, width: 1200, height: 630 }] },
    twitter: { card: "summary_large_image", title, description, images: [`${origin}${siteBasePath}og.png`] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
