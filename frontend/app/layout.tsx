import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "ARQ-RAG TurboQuant | Research Dashboard",
  description:
    "Hệ thống nghiên cứu chuyên sâu về Vector Quantization trong RAG — so sánh Raw, PQ, SQ8, ARQ và Adaptive.",
  keywords: ["RAG", "Product Quantization", "Vector Search", "TurboQuant", "AI Research"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
