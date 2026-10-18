#!/bin/bash

export LD_LIBRARY_PATH=$(find .venv -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)
