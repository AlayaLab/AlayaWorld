import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AlayaWorld — interactive world model",
  description:
    "Drive a generated world in real time: pick a starting image, write a prompt, and steer six camera axes from the keyboard.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
