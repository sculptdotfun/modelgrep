import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"], display: "swap" });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"], display: "swap" });
const display = Space_Grotesk({ variable: "--font-display", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  metadataBase: new URL("https://modelgrep.com"),
  title: {
    default: "modelgrep — Find & Understand Every LLM",
    template: "%s · modelgrep",
  },
  description:
    "The fastest way to find and understand AI models. Compare 300+ LLMs by intelligence benchmarks, speed, latency, price, context, and capabilities — updated continuously.",
  keywords: [
    "LLM comparison",
    "AI model benchmarks",
    "Artificial Analysis Intelligence Index",
    "GPQA",
    "model pricing",
    "fastest LLM",
    "cheapest LLM API",
    "tool calling models",
    "reasoning models",
    "context window",
    "OpenRouter models",
  ],
  authors: [{ name: "modelgrep" }],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: "modelgrep",
    url: "https://modelgrep.com",
    title: "modelgrep — Find & Understand Every LLM",
    description:
      "Compare 300+ LLMs by intelligence, speed, price, and capabilities. The default place to find and understand AI models.",
  },
  twitter: {
    card: "summary_large_image",
    title: "modelgrep — Find & Understand Every LLM",
    description: "Compare 300+ LLMs by intelligence, speed, price, and capabilities.",
  },
  robots: { index: true, follow: true },
  manifest: "/site.webmanifest",
  icons: { icon: "/favicon.svg" },
};

const websiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: "modelgrep",
  url: "https://modelgrep.com",
  description: "Find and understand every LLM — benchmarks, speed, price, and capabilities.",
  potentialAction: {
    "@type": "SearchAction",
    target: "https://modelgrep.com/?q={search_term_string}",
    "query-input": "required name=search_term_string",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} ${display.variable} h-full antialiased`} style={{ colorScheme: "light" }}>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
      </head>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
