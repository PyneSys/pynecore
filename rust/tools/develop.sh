#!/bin/sh
# Build the native extension and place it next to the Python sources, for an
# editable checkout. ``./develop.sh release`` builds optimized.
#
# The build lands as ``_native.abi3.so`` and over every interpreter-specific
# ``_native.cpython-*.so`` already there (a ``setup.py build_ext --inplace`` or a
# pip build writes one): Python loads such a file BEFORE the abi3 one, so a stale
# copy would hide the fresh build.
set -e
cd "$(dirname "$0")/.."
profile=debug
if [ "$1" = "release" ]; then
    profile=release
    cargo build -p pyne-transform-py --release
else
    cargo build -p pyne-transform-py
fi
dest=../src/pynecore/transformers
for lib in target/$profile/lib_native.dylib target/$profile/lib_native.so; do
    if [ -f "$lib" ]; then
        for target in "$dest/_native.abi3.so" "$dest"/_native.cpython-*.so; do
            # The glob stays literal when nothing matches it
            [ "$target" = "$dest/_native.abi3.so" ] || [ -f "$target" ] || continue
            # A new file, not an overwrite: macOS kills a process that maps a binary
            # rewritten in place
            cp "$lib" "$target.tmp"
            mv -f "$target.tmp" "$target"
            echo "installed $lib -> $target"
        done
    fi
done
