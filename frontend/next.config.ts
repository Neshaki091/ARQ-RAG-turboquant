import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8000/:path*',
      },
    ];
  },
  // Đưa ra cấp cao nhất theo yêu cầu của phiên bản Next.js mới
  // @ts-ignore - Bỏ qua cảnh báo type nếu có
  allowedDevOrigins: ['apiarqrag.evbtranding.site'],
  poweredByHeader: false,
};

export default nextConfig;
