#!/bin/bash
# Test script that doesn't require pip install — just sets PYTHONPATH
# to use local code. Run from inside the ESK/ folder of the bundle.
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "======================================"
echo "  Sigkernel-MMD Tests (No Install)"
echo "======================================"
echo ""

# Set PYTHONPATH to include the project root
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "Using local code from: ${SCRIPT_DIR}"
echo ""

echo "Step 1: Verifying imports..."
python -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from sigkernel_mmd import SigMMD; print('✓ Imports successful')"
echo ""

echo "Step 2: Running tests..."
echo "======================================"
PYTHONPATH="${SCRIPT_DIR}" pytest tests/ -v -k "not long_sequences" --durations=10
TEST_EXIT_CODE=$?
echo "======================================"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed"
    exit 1
fi