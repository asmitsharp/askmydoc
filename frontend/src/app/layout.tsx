import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  // Variable font — enables weight 100–900
  axes: ["opsz"],
});

export const metadata: Metadata = {
  title: "AskMyDocs — Your documents, answered",
  description:
    "A production-grade RAG system. Upload any document and ask questions grounded in your content.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} h-full`}>
      <body className="h-full">{children}</body>
    </html>
  );
}
