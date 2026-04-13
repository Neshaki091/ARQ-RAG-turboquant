FROM node:20-alpine
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Ưu tiên copy package files trước để tận dụng Docker Cache
COPY package.json package-lock.json* ./

# Dùng npm install thay vì npm ci để tận dụng cache tốt hơn
RUN npm install

# Copy mã nguồn còn lại
COPY . .

EXPOSE 3000
ENV PORT 3000

# Chế độ phát triển sẽ dùng command trong docker-compose
CMD ["npm", "run", "dev"]
