#!/bin/bash
# Hourly alternating code review + implementation loop
# Alternates between Claude Opus and GPT-4o
# Runs until 8 AM, then generates final report

REPO="/home/andrew/Desktop/kinect-fbt"
REPORT="$REPO/REVIEW_REPORT.md"
LOG="$REPO/review_log.txt"
DEADLINE="08:00"  # Stop after generating report at 8 AM
ROUND=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

get_model() {
    if [ $((ROUND % 2)) -eq 0 ]; then
        echo "claude-opus-4-5"
    else
        echo "gpt-4o"
    fi
}

get_role() {
    if [ $((ROUND % 2)) -eq 0 ]; then
        echo "REVIEW"
    else
        echo "IMPLEMENT"
    fi
}

run_claude_review() {
    local round=$1
    local focus=$2
    log "Running Claude Opus review (round $round) — focus: $focus"
    
    local prompt="You are reviewing the kinect-fbt project at $REPO.
This is a Kinect v2 full-body tracking system for VRChat on Meta Quest 3.
Round $round code review — focus area: $focus

Review the codebase and:
1. Identify bugs, edge cases, or missing error handling
2. Check for correctness of OSC protocol implementation
3. Verify the 1-euro filter math
4. Check fusion algorithm correctness
5. Identify any race conditions or threading issues
6. Suggest specific improvements with code

Output your review as markdown. Be specific and actionable."
    
    cat > /tmp/review_prompt.txt << 'PROMPT_EOF'
Review the kinect-fbt codebase for bugs, missing error handling, edge cases, and improvements.
Files to review: kinect_server/*.py and quest_app/app/src/main/kotlin/com/fbt/quest/*.kt
Focus on correctness, robustness, and VRChat FBT compatibility.
Provide specific code fixes.
PROMPT_EOF
    
    # Call Claude API
    local response
    response=$(curl -s --max-time 120 \
        -H "x-api-key: $ANTHROPIC_API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -H "content-type: application/json" \
        -d "{
            \"model\": \"claude-opus-4-5\",
            \"max_tokens\": 4096,
            \"messages\": [{
                \"role\": \"user\",
                \"content\": $(cat /tmp/review_prompt.txt | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))') 
            }]
        }" \
        "https://api.anthropic.com/v1/messages" 2>/dev/null)
    
    echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'content' in data and data['content']:
    print(data['content'][0]['text'])
else:
    print('API error: ' + str(data))
" 2>/dev/null
}

run_gpt_implement() {
    local round=$1
    local context=$2
    log "Running GPT-4o implementation (round $round)"
    
    local prompt="You are implementing improvements to the kinect-fbt project.
Previous review found issues. Implement the top 3 most impactful fixes.
For each fix: provide the complete corrected file section and explain the change.
Focus on: correctness, robustness, VRChat FBT compatibility.
Output as markdown with code blocks."
    
    local response
    response=$(curl -s --max-time 120 \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-4o\",
            \"max_tokens\": 4096,
            \"messages\": [
                {\"role\": \"system\", \"content\": \"You are an expert in Python, Kotlin, VRChat OSC, and Kinect v2 programming. Implement fixes concisely and correctly.\"},
                {\"role\": \"user\", \"content\": $(echo "$context" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null)}
            ]
        }" \
        "https://api.openai.com/v1/chat/completions" 2>/dev/null)
    
    echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'choices' in data and data['choices']:
    print(data['choices'][0]['message']['content'])
else:
    print('API error: ' + str(data))
" 2>/dev/null
}

commit_changes() {
    local round=$1
    local model=$2
    cd "$REPO"
    git add -A
    if git diff --cached --quiet; then
        log "No changes to commit for round $round"
    else
        git commit -m "review($round): $model code review + implementation" 2>/dev/null
        git push 2>/dev/null && log "Pushed round $round changes"
    fi
}

generate_report() {
    log "Generating final 8 AM report..."
    local rounds_done=$1
    
    cat > "$REPORT" << EOF
# kinect-fbt Review Report
Generated: $(date)
GitHub: https://github.com/andrewjfiore/kinect-fbt

## Summary
- Total review rounds: $rounds_done
- Models used: claude-opus-4-5 (odd rounds), gpt-4o (even rounds)
- Alternating pattern: Review → Implement → Review → Implement...

## Implementation Status

### Component 1: Linux Kinect Fusion Server ✅
- \`server.py\`: Main loop, CLI, 20Hz OSC output
- \`camera.py\`: KinectCamera with depth-informed 3D landmark extraction
- \`filter.py\`: 1-euro filter (from scratch, ~80 lines)
- \`fusion.py\`: Weighted multi-camera fusion + 5 virtual trackers
- \`calibration.py\`: Stereo checkerboard calibration via cv2.stereoCalibrate
- \`osc_output.py\`: OSC bundle construction + UDP send
- \`debug_http.py\`: Flask debug server with MJPEG preview

### Component 2: Meta Quest Android APK ✅
- \`OscReceiver.kt\`: UDP OSC listener, coroutine-based
- \`VRChatOscForwarder.kt\`: 20Hz VRChat OSC forwarder, confidence threshold
- \`MainActivity.kt\`: Full Android Views UI (settings, status, calibration)

## Key Technical Decisions
1. **1-euro filter**: Implemented from scratch in filter.py (min_cutoff=1.0, beta=0.1)
2. **Frame buffer**: queue.Queue(maxsize=2) with put_nowait/catch Full (no threading.Lock)
3. **Pipeline fallback**: OpenGLPacketPipeline → CpuPacketPipeline with warning
4. **Depth clamping**: 500mm–4500mm (Kinect blind spot + accuracy limit)
5. **Confidence threshold**: 0.2 — below this, VRChat holds last position
6. **OSC addressing**: Custom /fbt/* namespace (server) → /tracking/trackers/* (VRChat)

## Review Log
$(cat "$LOG" 2>/dev/null | tail -100)
EOF
    
    git -C "$REPO" add "$REPORT"
    git -C "$REPO" commit -m "report: 8 AM review report ($rounds_done rounds)" 2>/dev/null
    git -C "$REPO" push 2>/dev/null
    log "Report generated: $REPORT"
}

# Main loop
log "Starting hourly review loop. Target: 8 AM report."
log "Repo: $REPO"

FOCUS_AREAS=(
    "depth extraction and depth confidence logic in camera.py"
    "1-euro filter math and edge cases in filter.py"
    "multi-camera fusion weighted average and staleness in fusion.py"
    "OSC bundle format and VRChat tracker compatibility"
    "Android OscReceiver parsing correctness"
    "VRChatOscForwarder confidence threshold and forwarding"
    "calibration matrix computation and application"
    "overall robustness: error handling, timeouts, reconnection"
)

while true; do
    CURRENT_HOUR=$(date +%H)
    CURRENT_MIN=$(date +%M)
    
    # Check if it's 8 AM
    if [ "$CURRENT_HOUR" -eq 8 ] && [ "$CURRENT_MIN" -lt 10 ]; then
        generate_report "$ROUND"
        log "Done. Report ready at $REPORT"
        exit 0
    fi
    
    ROUND=$((ROUND + 1))
    FOCUS="${FOCUS_AREAS[$((ROUND % ${#FOCUS_AREAS[@]}))]}"
    MODEL=$(get_model)
    ROLE=$(get_role)
    
    log "=== Round $ROUND: $MODEL ($ROLE) — $FOCUS ==="
    
    if [ $((ROUND % 2)) -eq 1 ]; then
        # Odd rounds: Claude Opus review
        REVIEW=$(run_claude_review "$ROUND" "$FOCUS")
        REVIEW_FILE="$REPO/reviews/round_${ROUND}_opus.md"
        mkdir -p "$REPO/reviews"
        echo "# Round $ROUND — Claude Opus Review" > "$REVIEW_FILE"
        echo "Focus: $FOCUS" >> "$REVIEW_FILE"
        echo "---" >> "$REVIEW_FILE"
        echo "$REVIEW" >> "$REVIEW_FILE"
        log "Review saved to $REVIEW_FILE"
    else
        # Even rounds: GPT-4o implementation
        PREV_REVIEW=""
        if [ -f "$REPO/reviews/round_$((ROUND-1))_opus.md" ]; then
            PREV_REVIEW=$(cat "$REPO/reviews/round_$((ROUND-1))_opus.md" 2>/dev/null | head -100)
        fi
        IMPL=$(run_gpt_implement "$ROUND" "Previous review:\n$PREV_REVIEW\n\nImplement the top fixes.")
        IMPL_FILE="$REPO/reviews/round_${ROUND}_gpt4o.md"
        echo "# Round $ROUND — GPT-4o Implementation" > "$IMPL_FILE"
        echo "---" >> "$IMPL_FILE"
        echo "$IMPL" >> "$IMPL_FILE"
        log "Implementation saved to $IMPL_FILE"
    fi
    
    commit_changes "$ROUND" "$MODEL"
    
    log "Round $ROUND complete. Sleeping 1 hour..."
    sleep 3600
done
