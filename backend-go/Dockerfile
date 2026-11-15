FROM golang:1.23-bookworm AS builder

WORKDIR /app
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /svgml-go-gateway .

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /svgml-go-gateway /usr/local/bin/svgml-go-gateway
RUN mkdir -p /app/data

ENV GO_PORT=18080
ENV DATA_DIR=/app/data
EXPOSE 18080
CMD ["svgml-go-gateway"]
