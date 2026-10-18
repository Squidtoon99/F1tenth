import type { NextConfig } from "next";

const flask = process.env.FLASK_URL ?? "http://127.0.0.1:5000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${flask}/api/:path*` }];
  },
};

export default nextConfig;
