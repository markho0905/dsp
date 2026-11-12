#!/bin/bash
# Builds RoomCorr.app and installs it to /Applications.  Needs internet (pip) and ~5 minutes.
cd "$(dirname "$0")" || exit 1
echo "=== RoomCorr installer ==="

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)'; then
    PY=$(command -v "$c"); break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ not found. Install it first:  brew install python@3.12   (or download from python.org)"
  read -p "Press Enter to close"; exit 1
fi
echo "Using $PY"

set -e
"$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy sounddevice PySide6 pyinstaller

ICON=""
if python make_icon.py 2>/dev/null && [ -f icon_1024.png ]; then
  rm -rf icon.iconset && mkdir icon.iconset
  for s in 16 32 128 256 512; do
    sips -z $s $s icon_1024.png --out icon.iconset/icon_${s}x${s}.png >/dev/null
    sips -z $((s*2)) $((s*2)) icon_1024.png --out icon.iconset/icon_${s}x${s}@2x.png >/dev/null
  done
  iconutil -c icns icon.iconset -o icon.icns && ICON="--icon icon.icns" || true
fi

rm -rf build dist RoomCorr.spec
pyinstaller --noconfirm --windowed --name RoomCorr --osx-bundle-identifier com.roomcorr.app \
  --collect-all sounddevice --collect-all _sounddevice_data $ICON roomcorr_app.py

PL="dist/RoomCorr.app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string 'RoomCorr reads the audio of your virtual input device (e.g. BlackHole).'" "$PL" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Set :NSMicrophoneUsageDescription 'RoomCorr reads the audio of your virtual input device (e.g. BlackHole).'" "$PL"
codesign --force --deep --sign - dist/RoomCorr.app

rm -rf /Applications/RoomCorr.app
cp -R dist/RoomCorr.app /Applications/
xattr -dr com.apple.quarantine /Applications/RoomCorr.app 2>/dev/null || true
set +e

if ! (command -v brew >/dev/null 2>&1 && brew list --cask blackhole-2ch >/dev/null 2>&1) && \
   ! ls /Library/Audio/Plug-Ins/HAL 2>/dev/null | grep -qi blackhole; then
  echo
  echo "BlackHole (virtual audio device) is not installed."
  if command -v brew >/dev/null 2>&1; then
    read -p "Install it now with Homebrew? [y/N] " yn
    [ "$yn" = "y" ] && brew install --cask blackhole-2ch
  else
    echo "Get it from https://existential.audio/blackhole/ (2ch version)."
  fi
fi

echo
echo "Done. RoomCorr is in /Applications. Opening it now (allow microphone access when asked)."
open /Applications/RoomCorr.app
sleep 7
if ! pgrep -x RoomCorr >/dev/null 2>&1; then
  echo
  echo "=== RoomCorr 沒有成功啟動，以下是錯誤訊息（請把這一段整個貼給我） ==="
  ( /Applications/RoomCorr.app/Contents/MacOS/RoomCorr > /tmp/roomcorr_run.log 2>&1 & )
  sleep 8
  echo "--- 啟動輸出 ---"
  tail -40 /tmp/roomcorr_run.log
  echo "--- ~/Library/Logs/RoomCorr.log ---"
  tail -40 "$HOME/Library/Logs/RoomCorr.log" 2>/dev/null
  echo "=== 結束 ==="
fi
read -p "Press Enter to close"
