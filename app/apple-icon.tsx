import { ImageResponse } from "next/og";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

// Apple touch icon — the >_ prompt mark on ink.
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
          background: "#101014",
          borderRadius: 36,
        }}
      >
        <svg width="120" height="120" viewBox="0 0 100 100">
          <path d="M28 32 L52 50 L28 68" fill="none" stroke="#ffffff" strokeWidth="9" strokeLinecap="square" />
          <rect x="56" y="61" width="20" height="9" fill="#5b3df5" />
        </svg>
      </div>
    ),
    size,
  );
}
