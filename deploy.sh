#!/bin/sh
docker compose down || true
docker container rm -f "$APPLICATION_NAME" || true
docker compose up -d