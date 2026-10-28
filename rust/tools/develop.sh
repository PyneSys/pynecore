#!/bin/sh
# Build the native extension and place it next to the Python sources, for an
# editable checkout. ``./develop.sh release`` builds optimized.
set -e
cd "$(dirname "$0")/.."
profile=debug
if [ "$1" = "release" ]; then
    profile=release
    cargo build -p pyne-transform-py --release
else
    cargo build -p pyne-transform-py
fi
for lib in target/$profile/lib_native.dylib target/$profile/lib_native.so; do
    if [ -f "$lib" ]; then
        # A new file, not an overwrite: macOS kills a process that maps a binary
        # rewritten in place
        cp "$lib" ../src/pynecore/transformers/_native.abi3.so.tmp
        mv -f ../src/pynecore/transformers/_native.abi3.so.tmp \
            ../src/pynecore/transformers/_native.abi3.so
        echo "installed $lib"
    fi
done
