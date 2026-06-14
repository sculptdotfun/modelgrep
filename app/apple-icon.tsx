import { ImageResponse } from "next/og";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

// Apple touch icon — the gradient prompt mark.
export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #7d5cff, #5a32f0)",
          borderRadius: 40,
        }}
      >
        <svg width="120" height="120" viewBox="0 0 100 100">
          <path d="M31 33 L51 50 L31 67" fill="none" stroke="#ffffff" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
          <rect x="55" y="58" width="24" height="10" rx="5" fill="#ffffff" />
        </svg>
      </div>
    ),
    size,
  );
}
