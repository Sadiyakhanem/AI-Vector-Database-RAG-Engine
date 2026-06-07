# ── Stage 1: Build ───────────────────────────────────────────────────────────
# Use the full JDK to compile and package the application
FROM eclipse-temurin:17-jdk-alpine AS builder

WORKDIR /build

# Copy Maven wrapper and POM first (layer caching: dependencies won't
# re-download if only source code changed — speeds up rebuilds significantly)
COPY pom.xml .
COPY mvnw .
COPY .mvn .mvn

# Download dependencies in a separate layer
RUN ./mvnw dependency:go-offline -q

# Copy source and build
COPY src ./src
RUN ./mvnw package -DskipTests -q

# ── Stage 2: Run ──────────────────────────────────────────────────────────────
# Use JRE only (smaller image: ~200MB vs ~400MB with JDK)
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

# Create a non-root user — never run as root in production
RUN addgroup -S vectordb && adduser -S vectordb -G vectordb
USER vectordb

# Copy the built JAR from stage 1
COPY --from=builder /build/target/*.jar app.jar

# WAL file persists in this volume — mount it to survive container restarts
VOLUME ["/app/data"]

EXPOSE 8080

# JVM tuning for containers:
#   -XX:+UseContainerSupport      — respect Docker CPU/memory limits
#   -XX:MaxRAMPercentage=75.0     — use 75% of container memory for heap
#   -XX:+ExitOnOutOfMemoryError   — fail fast instead of degraded performance
ENV JAVA_OPTS="-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -XX:+ExitOnOutOfMemoryError"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
