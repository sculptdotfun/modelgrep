import { ImageResponse } from "next/og";
import { getModelDetail } from "@/lib/catalog";
import { fmtContext, fmtLatency, fmtPrice, fmtThroughput, modelOwner } from "@/lib/format";
import { ownerColor } from "@/lib/owners";

export const revalidate = 3600;

const SIZE = { width: 1200, height: 630 };

// Per-model social/OG image, referenced as /og?id=<model-id>.
export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get("id") ?? "";
  const m = id ? await getModelDetail(id).catch(() => undefined) : undefined;

  if (!m) {
    return new ImageResponse(
      (
        <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "#fff", fontSize: 64, fontWeight: 800, fontFamily: "sans-serif", color: "#16161d" }}>
          model<span style={{ color: "#0a0a0a" }}>grep</span>
        </div>
      ),
      SIZE,
    );
  }

  const owner = modelOwner(m.id);
  const oc = ownerColor(owner);
  const stats: { label: string; value: string }[] = [
    { label: "Intelligence", value: m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—" },
    { label: "Speed", value: m.throughput ? `${fmtThroughput(m.throughput)} t/s` : "—" },
    { label: "Latency", value: fmtLatency(m.latency) },
    { label: "Input", value: m.price_input != null ? `${fmtPrice(m.price_input)}/M` : "—" },
    { label: "Context", value: fmtContext(m.context_length) },
  ];

  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: "#ffffff", padding: "60px 68px", fontFamily: "sans-serif" }}>
        <div style={{ display: "flex", fontSize: 26, fontWeight: 700, color: "#16161d" }}>
          model<span style={{ color: "#0a0a0a" }}>grep</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 24, marginTop: 40 }}>
          <div style={{ display: "flex", width: 96, height: 96, borderRadius: 24, background: oc, color: "#fff", fontSize: 52, fontWeight: 800, alignItems: "center", justifyContent: "center" }}>
            {owner.charAt(0).toUpperCase()}
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 56, fontWeight: 800, color: "#16161d", letterSpacing: -1.5, lineHeight: 1.05 }}>{m.name}</div>
            <div style={{ fontSize: 26, color: "#8a8c97", marginTop: 6 }}>{m.id}</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 16, marginTop: "auto" }}>
          {stats.map((s) => (
            <div key={s.label} style={{ display: "flex", flexDirection: "column", flex: 1, background: "#f7f8fa", border: "1px solid #ebecf0", borderRadius: 16, padding: "20px 22px" }}>
              <div style={{ fontSize: 18, color: "#8a8c97", textTransform: "uppercase", letterSpacing: 1 }}>{s.label}</div>
              <div style={{ fontSize: 38, fontWeight: 800, color: "#16161d", marginTop: 8 }}>{s.value}</div>
            </div>
          ))}
        </div>
      </div>
    ),
    SIZE,
  );
}
