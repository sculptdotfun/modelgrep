import { ImageResponse } from "next/og";
import { getCatalog } from "@/lib/catalog";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "modelgrep — the LLM leaderboard";
export const revalidate = 3600;

export default async function OG() {
  const models = await getCatalog().catch(() => []);
  const benchmarked = models.filter((m) => m.aa || m.da).length;
  const top = models
    .filter((m) => m.aa?.intelligence != null)
    .sort((a, b) => b.aa!.intelligence! - a.aa!.intelligence!)
    .slice(0, 5);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "#ffffff",
          padding: "64px 72px",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", fontSize: 30, fontWeight: 700, color: "#16161d" }}>
          model<span style={{ color: "#0a0a0a" }}>grep</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", marginTop: 28 }}>
          <div style={{ fontSize: 68, fontWeight: 800, color: "#16161d", letterSpacing: -2, lineHeight: 1.05 }}>
            Find &amp; understand
          </div>
          <div style={{ fontSize: 68, fontWeight: 800, color: "#16161d", letterSpacing: -2, lineHeight: 1.05 }}>
            every LLM.
          </div>
        </div>
        <div style={{ display: "flex", fontSize: 28, color: "#51535e", marginTop: 20 }}>
          {models.length} models · {benchmarked} benchmarked · ranked by intelligence, speed &amp; price
        </div>
        <div style={{ display: "flex", gap: 12, marginTop: "auto", flexWrap: "wrap" }}>
          {top.map((m) => (
            <div
              key={m.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                background: "#f2f3f6",
                borderRadius: 12,
                padding: "12px 18px",
                fontSize: 24,
                color: "#16161d",
              }}
            >
              <span style={{ fontWeight: 700 }}>{m.aa!.intelligence!.toFixed(1)}</span>
              <span style={{ color: "#51535e" }}>{m.id.split("/").slice(1).join("/")}</span>
            </div>
          ))}
        </div>
      </div>
    ),
    size,
  );
}
