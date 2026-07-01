#!/bin/bash
# Send a command string to the manimgl Zellij pane and return focus here
target=$(cat /tmp/manimgl-pane-id)
zellij action focus-pane-id "$target"
zellij action write-chars "$1"
zellij action write '10'   # byte 10 = newline / Enter
