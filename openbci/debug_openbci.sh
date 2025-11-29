#!/bin/bash
# debug_openbci.sh - Debug OpenBCI data collection

SHIELD_IP="192.168.4.1"
LOCAL_IP=$(ip addr show wlan1 | grep "inet " | awk '{print $2}' | cut -d/ -f1)
PORT=3000

echo "=== OpenBCI Debug Script ==="
echo "Shield IP: $SHIELD_IP"
echo "Local IP: $LOCAL_IP"
echo "Port: $PORT"
echo ""

# Step 1: Stop any existing stream
echo "1. Stopping any existing streams..."
curl -s -X DELETE http://$SHIELD_IP/tcp
echo ""
sleep 1

# Step 2: Check board connection
echo "2. Checking board connection..."
BOARD_INFO=$(curl -s http://$SHIELD_IP/board)
echo "$BOARD_INFO" | python3 -m json.tool
CONNECTED=$(echo "$BOARD_INFO" | grep -o '"board_connected":\s*true')
if [ -z "$CONNECTED" ]; then
  echo "ERROR: Board not connected!"
  exit 1
fi
echo ""

# Step 3: Start board streaming
echo "3. Starting board (sending 'b' command)..."
curl -s -X POST http://$SHIELD_IP/command \
  -H "Content-Type: application/json" \
  -d '{"command":"b"}'
echo ""
sleep 2

# Step 4: Start listener and stream
echo "4. Starting TCP listener on port $PORT..."
echo "   Listening for data..."

# Create a temporary file for data
TMP_FILE="/tmp/openbci_debug_$$.dat"

# Start netcat in background
timeout 10 nc -l $LOCAL_IP $PORT >"$TMP_FILE" &
NC_PID=$!

sleep 1

# Step 5: Request stream
echo "5. Requesting stream from shield..."
STREAM_CMD=$(
  cat <<EOF
{
  "ip": "$LOCAL_IP",
  "port": $PORT,
  "output": "json",
  "delimiter": true,
  "latency": 10000,
  "burst": false
}
EOF
)

echo "   Config: $STREAM_CMD"

STREAM_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST http://$SHIELD_IP/tcp \
  -H "Content-Type: application/json" \
  -d "$STREAM_CMD")

HTTP_STATUS=$(echo "$STREAM_RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$STREAM_RESPONSE" | grep -v "HTTP_STATUS")

echo "   HTTP Status: $HTTP_STATUS"
echo "   Response: $BODY"

if [ "$HTTP_STATUS" != "200" ]; then
  echo "   ERROR: Failed to start stream (HTTP $HTTP_STATUS)"
  kill $NC_PID 2>/dev/null
  exit 1
fi

echo ""
echo "6. Waiting 10 seconds for data..."
echo "   (Press Ctrl+C to stop early)"

# Wait for data
wait $NC_PID 2>/dev/null

# Check results
echo ""
if [ -f "$TMP_FILE" ] && [ -s "$TMP_FILE" ]; then
  BYTES=$(wc -c <"$TMP_FILE")
  LINES=$(wc -l <"$TMP_FILE")
  echo "SUCCESS! Received $BYTES bytes, $LINES lines"
  echo ""
  echo "First few lines:"
  head -n 3 "$TMP_FILE"
  echo ""
  echo "Data saved to: $TMP_FILE"

  # Try to parse JSON
  echo ""
  echo "Trying to parse JSON..."
  head -n 1 "$TMP_FILE" | python3 -m json.tool 2>/dev/null && echo "  ✓ Valid JSON" || echo "  ✗ Invalid JSON"

else
  echo "FAILED: No data received"
  echo "  File: $TMP_FILE"
  if [ -f "$TMP_FILE" ]; then
    echo "  Size: $(wc -c <"$TMP_FILE") bytes"
  else
    echo "  File does not exist"
  fi
fi

# Cleanup
echo ""
echo "7. Stopping stream..."
curl -s -X DELETE http://$SHIELD_IP/tcp
echo ""

echo "=== Debug Complete ==="
echo ""
echo "If you saw data above, the connection works!"
echo "If not, check:"
echo "  1. Is wlan1 connected to OpenBCI-E324?"
echo "  2. Can you ping $SHIELD_IP?"
echo "  3. Is your local IP correct? ($LOCAL_IP)"
echo "  4. Is port $PORT already in use?"
