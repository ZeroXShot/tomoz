# syntax=docker/dockerfile:1.7
# The Tomoz command line and S3 gateway, as a small non-root image.
#
#   docker build -t tomoz .
#   docker run --rm tomoz --help

FROM rust:1.98-slim-trixie AS build
# Parallel jobs of the build (a shared machine may want fewer).
ARG CARGO_BUILD_JOBS
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p tomoz-cli \
    && install -m 0755 target/release/tomoz /usr/local/bin/tomoz

FROM debian:trixie-slim
RUN useradd --uid 10001 --user-group --home-dir /var/lib/tomoz --create-home --shell /usr/sbin/nologin tomoz
COPY --from=build /usr/local/bin/tomoz /usr/local/bin/tomoz
USER 10001:10001
WORKDIR /var/lib/tomoz
# Inside a container the gateway must listen on all interfaces to be reached
# through a published port; requests are still authenticated (SigV4) and the
# credentials file has to be provided, e.g. as a secret:
#   -e TOMOZ__AUTH__CREDENTIALS_FILE=/run/secrets/tomoz-credentials
ENV TOMOZ__LISTEN=0.0.0.0:9100 \
    TOMOZ__DATA_DIR=/var/lib/tomoz/data
EXPOSE 9100
VOLUME ["/var/lib/tomoz"]
ENTRYPOINT ["tomoz"]
CMD ["serve"]
