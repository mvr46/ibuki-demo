#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Configure Reachy Mini Wireless for a hotspot demo.

Run this on the Reachy Mini, usually over SSH:

  bash scripts/configure_reachy_hotspot_demo.sh --ssid "My iPhone" --password "hotspot-password"

Options:
  --ssid SSID              Phone hotspot SSID. Also accepts REACHY_HOTSPOT_SSID.
  --password PASSWORD      Phone hotspot password. Also accepts REACHY_HOTSPOT_PASSWORD.
  --forget-other-wifi      Remove saved Wi-Fi connections other than this hotspot and the fallback AP.
  --no-restart-daemon      Configure Wi-Fi only; do not restart reachy-mini-daemon.
  -h, --help               Show this help.
EOF
}

ssid="${REACHY_HOTSPOT_SSID:-}"
password="${REACHY_HOTSPOT_PASSWORD:-}"
forget_other_wifi=0
restart_daemon=1

while (($#)); do
  case "$1" in
    --ssid)
      ssid="${2:-}"
      shift 2
      ;;
    --password)
      password="${2:-}"
      shift 2
      ;;
    --forget-other-wifi)
      forget_other_wifi=1
      shift
      ;;
    --no-restart-daemon)
      restart_daemon=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ssid" || -z "$password" ]]; then
  echo "Missing --ssid or --password." >&2
  usage >&2
  exit 2
fi

if ((${#password} < 8)); then
  echo "Hotspot password must be at least 8 characters for WPA/WPA2." >&2
  exit 2
fi

if ! command -v nmcli >/dev/null 2>&1; then
  echo "nmcli was not found. This script is intended for Reachy Mini Wireless / Linux." >&2
  exit 1
fi

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  sudo_cmd=()
else
  sudo_cmd=(sudo)
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_redacted() {
  local -a display=()
  local redact_next=0
  local arg
  for arg in "$@"; do
    if [[ "$redact_next" -eq 1 ]]; then
      display+=("<redacted>")
      redact_next=0
      continue
    fi
    display+=("$arg")
    case "$arg" in
      password|wifi-sec.psk)
        redact_next=1
        ;;
    esac
  done

  printf '+'
  printf ' %q' "${display[@]}"
  printf '\n'
  "$@"
}

connection_exists() {
  nmcli -t -f NAME connection show | sed 's/\\:/:/g' | grep -Fxq "$1"
}

wifi_device() {
  nmcli -t -f DEVICE,TYPE,STATE device status | awk -F: '$2 == "wifi" && $3 != "unavailable" { print $1; exit }'
}

echo "Configuring Reachy Mini Wi-Fi for hotspot: $ssid"
run "${sudo_cmd[@]}" rfkill unblock wifi || true
run "${sudo_cmd[@]}" nmcli radio wifi on
run "${sudo_cmd[@]}" nmcli device wifi rescan || true

if connection_exists "$ssid"; then
  echo "Updating saved Wi-Fi connection: $ssid"
  run_redacted "${sudo_cmd[@]}" nmcli connection modify "$ssid" \
    802-11-wireless.ssid "$ssid" \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$password"
else
  echo "Creating saved Wi-Fi connection: $ssid"
  run_redacted "${sudo_cmd[@]}" nmcli device wifi connect "$ssid" password "$password" name "$ssid"
fi

run "${sudo_cmd[@]}" nmcli connection modify "$ssid" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100

if connection_exists "Hotspot"; then
  run "${sudo_cmd[@]}" nmcli connection modify "Hotspot" \
    connection.autoconnect yes \
    connection.autoconnect-priority -20
fi

if [[ "$forget_other_wifi" -eq 1 ]]; then
  echo "Removing saved Wi-Fi connections other than '$ssid' and the fallback AP."
  while IFS= read -r name; do
    [[ -z "$name" || "$name" == "$ssid" || "$name" == "Hotspot" ]] && continue
    run "${sudo_cmd[@]}" nmcli connection delete "$name"
  done < <(nmcli -t -f NAME,TYPE connection show | awk -F: '$2 == "wifi" { gsub(/\\:/, ":", $1); print $1 }')
fi

echo "Activating hotspot connection..."
run "${sudo_cmd[@]}" nmcli connection up "$ssid"

if [[ "$restart_daemon" -eq 1 ]]; then
  service_name="reachy-mini-daemon.service"
  if ! systemctl list-unit-files "$service_name" >/dev/null 2>&1; then
    echo "Systemd service is missing; attempting to install the Reachy Mini wireless service."
    install_script="$(
      python3 - <<'PY' 2>/dev/null || true
from pathlib import Path
import reachy_mini

path = Path(reachy_mini.__file__).parent / "daemon" / "app" / "services" / "wireless" / "install_service.sh"
print(path if path.exists() else "")
PY
    )"
    if [[ -n "$install_script" ]]; then
      run bash "$install_script"
    else
      echo "Could not locate Reachy Mini wireless service installer." >&2
      exit 1
    fi
  fi

  echo "Enabling and restarting $service_name..."
  run "${sudo_cmd[@]}" systemctl daemon-reload
  run "${sudo_cmd[@]}" systemctl enable "$service_name"
  run "${sudo_cmd[@]}" systemctl restart "$service_name"
fi

device="$(wifi_device || true)"
ip_address=""
if [[ -n "$device" ]]; then
  ip_address="$(ip -4 -o addr show dev "$device" | awk '{ split($4, parts, "/"); print parts[1]; exit }')"
fi

echo
echo "Reachy Mini hotspot demo setup complete."
echo "Connected network: $(nmcli -t -f GENERAL.CONNECTION device show "${device:-wlan0}" 2>/dev/null | cut -d: -f2- || true)"
if [[ -n "$ip_address" ]]; then
  echo "Robot IP on hotspot: $ip_address"
  echo "Daemon status URL: http://$ip_address:8000/api/daemon/status"
else
  echo "Could not determine the robot IP yet. Check: nmcli device status"
fi
