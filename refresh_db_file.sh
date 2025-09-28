#!/bin/bash
# Copy and atomically replace energy_2.db

SRC="/volume1/docker/opt/nodered/node_red_data/energy_2.db"
DST_DIR="/volume1/docker/opt/inverter_opt"
TMP="$DST_DIR/energy_2.db.new"
DST="$DST_DIR/energy_2.db"

cp "$SRC" "$TMP"      # Step 1: copy to temp file
mv -f "$TMP" "$DST"   # Step 2: atomic rename to final name