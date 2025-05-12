#!/bin/bash

echo "Attempting to force close camera processes..."

# (Linux-GNU)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Find and kill processes using video devices
    fuser -k /dev/video0
    echo "Killed processes using /dev/video0"
    
    # Kill any processes that might be using the camera
    pkill -f "v4l2"
    pkill -f "camera"
    pkill -f "opencv"
    echo "Killed processes related to v4l2, camera, and opencv"

