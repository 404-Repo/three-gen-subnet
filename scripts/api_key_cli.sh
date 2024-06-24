#!/bin/bash

DB_FILE=$1

if [ -z "$DB_FILE" ]; then
  echo "Database file must be specified as the first argument."
  exit 1
fi

setup_database() {
  if [ ! -f "$DB_FILE" ]; then
    sqlite3 "$DB_FILE" <<EOF
CREATE TABLE api_keys (
    api_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    max_requests INTEGER NOT NULL,
    period INTEGER NOT NULL
);
EOF
  fi
}

list_keys() {
  setup_database
  sqlite3 "$DB_FILE" <<EOF
SELECT api_key, name, max_requests, period FROM api_keys;
EOF
}

gen_key() {
  API_KEY=$(openssl rand -base64 32 | tr -d '[:punct:]')
  echo "Generated API Key: $API_KEY"
}

add_key() {
  setup_database
  local API_KEY=$1
  local NAME=$2
  local MAX_REQUESTS=$3
  local PERIOD=$4
  sqlite3 "$DB_FILE" <<EOF
INSERT INTO api_keys (api_key, name, max_requests, period)
VALUES ('$API_KEY', '$NAME', $MAX_REQUESTS, $PERIOD);
EOF
  echo "Added API Key: $API_KEY"
}

remove_key() {
  setup_database
  local API_KEY=$1
  sqlite3 "$DB_FILE" <<EOF
DELETE FROM api_keys WHERE api_key = '$API_KEY';
EOF
  echo "Removed API Key: $API_KEY"
}

usage() {
  echo "Usage: $0 <db_file> {list|gen|add <api_key> <name> <max_requests> <period>|remove <api_key>}"
  exit 1
}

shift 1

case "$1" in
  list)
    list_keys
    ;;
  gen)
    gen_key
    ;;
  add)
    [ $# -eq 5 ] || usage
    add_key "$2" "$3" "$4" "$5"
    ;;
  remove)
    [ $# -eq 2 ] || usage
    remove_key "$2"
    ;;
  *)
    usage
    ;;
esac