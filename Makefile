.PHONY: pg rag app

# Build the pg_vector Docker image
pg:
	docker build -f Dockerfile.postgres -t postgres_vector .

# Build the rag Docker image
rag:
	@# $(grep -v '^#' .env | sed 's/^/--build-arg /')
	@# --build-arg TZ_INFO="$TZ_INFO" \
	docker build -f Dockerfile.rag -t rag .

# Run the "app" using Docker Compose
app:
	@# docker compose run app
	docker compose up -d

# Build both images and run "app"
all: pg rag app