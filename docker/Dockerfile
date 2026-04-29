FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y wget gcc ca-certificates && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then GOARCH="arm64"; else GOARCH="amd64"; fi && \
    wget https://go.dev/dl/go1.25.0.linux-${GOARCH}.tar.gz && \
    tar -C /usr/local -xzf go1.25.0.linux-${GOARCH}.tar.gz

ENV PATH=$PATH:/usr/local/go/bin

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o apiserver ./cmd/server/main.go 


FROM ubuntu:24.04

WORKDIR /app
COPY --from=builder /app/apiserver /app/apiserver

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Create a secure non-root user
RUN useradd -m nonroot
USER nonroot

EXPOSE 8080
CMD ["/app/apiserver"]
