#!/bin/bash
# Replace ALL ROCm libraries in PyTorch with system versions
# This solves the "cannot enable executable stack" SELinux issue

TORCH_LIB="/home/fiqri/.local/lib/python3.10/site-packages/torch/lib"
SYSTEM_LIB="/usr/lib64"

echo "🔧 Replacing PyTorch ROCm libraries with system versions..."
echo ""

# List of known ROCm libraries that may need replacement
LIBS=(
    "libamdhip64.so"
    "libhiprtc.so"
    "libhipblas.so"
    "libhipsparse.so"
    "libhipsolver.so"
    "librocblas.so"
    "librocrand.so"
    "librocsparse.so"
    "librocsolver.so"
    "librocfft.so"
    "libMIOpen.so"
    "librccl.so"
    "libhsa-runtime64.so"
)

for lib in "${LIBS[@]}"; do
    TORCH_FILE="$TORCH_LIB/$lib"
    SYSTEM_FILE="$SYSTEM_LIB/$lib"
    
    # Check if PyTorch has this library
    if [ -e "$TORCH_FILE" ] || [ -L "$TORCH_FILE" ]; then
        # Check if system has this library
        if [ -e "$SYSTEM_FILE" ]; then
            # Skip if already a symlink to system
            if [ -L "$TORCH_FILE" ]; then
                LINK_TARGET=$(readlink "$TORCH_FILE")
                if [[ "$LINK_TARGET" == "$SYSTEM_FILE"* ]] || [[ "$LINK_TARGET" == "/usr/lib64/"* ]]; then
                    echo "✅ $lib -> already linked to system"
                    continue
                fi
            fi
            
            # Backup and replace
            echo "🔄 $lib -> replacing with system version"
            mv "$TORCH_FILE" "${TORCH_FILE}.bak" 2>/dev/null
            ln -sf "$SYSTEM_FILE" "$TORCH_FILE"
        else
            echo "⚠️  $lib -> no system version found, keeping original"
        fi
    fi
done

echo ""
echo "✅ Done! Testing PyTorch import..."
echo ""

cd /home/fiqri/Desktop/IDN/AI_v2
python3 test_gpu.py
