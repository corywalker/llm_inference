set -e

OS="$(uname -s)"


TEXT_OUTPUT=false
while getopts "t" opt; do
  case $opt in
    t) TEXT_OUTPUT=true ;;
    *) echo "Usage: $0 [-t]" >&2; exit 1 ;;
  esac
done

bazel build -c opt --copt=-g --strip=never --fission=no //:llm_inference
if [ "$OS" = "Linux" ]; then
    sudo sysctl -w kernel.perf_event_paranoid=-1
    timeout -s INT 15s perf record -g -- bazel-bin/llm_inference --model ~/gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf --prompt="Write a poem" || true
    if [ "$TEXT_OUTPUT" = true ]; then
        perf report --stdio
    else
        perf script -F +pid | gzip > profile.perf.gz
        echo ""
        echo "Profiling complete!"
        echo "Visualize with 'perf report' or drag/drop 'profile.perf.gz' and analyze at https://profiler.firefox.com"
    fi
elif [ "$OS" = "Darwin" ]; then
    if [ "$TEXT_OUTPUT" = true ]; then
        bazel-bin/llm_inference --model ~/Documents/temp/gemma-3-1b-it-q4_0.gguf --prompt="Write a poem" --metal &
        PID=$!
        sample $PID 10 -f /tmp/sample.txt
        kill $PID || true
        # Extract the "Sort by top of stack" section and print the top 10 lines
        awk '/Sort by top of stack/ {flag=1; next} flag' /tmp/sample.txt | head -n 10
        rm /tmp/sample.txt
    else
        rm -rf mac_profile.trace
        xcrun xctrace record --template 'Metal System Trace' --time-limit 15s --launch --output mac_profile.trace -- bazel-bin/llm_inference --model ~/Documents/temp/gemma-3-1b-it-q4_0.gguf --prompt="Write a poem" --metal || true
        killall Instruments || true
        open mac_profile.trace
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi