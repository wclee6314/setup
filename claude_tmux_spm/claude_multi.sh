#!/bin/bash
#
# claude_multi.sh - 여러 디렉토리에서 Claude Code를 tmux 세션으로 동시 실행
#
# 사용법:
#   ./claude_multi.sh <dir1> <dir2> [dir3] ...
#   ./claude_multi.sh -f dirs.txt           # 파일에서 디렉토리 목록 읽기
#   ./claude_multi.sh -s my-session <dirs>  # 세션 이름 지정
#   ./claude_multi.sh -k [session-name]     # 세션 종료
#   ./claude_multi.sh -l                    # 실행 중인 세션 목록
#
# 예시:
#   ./claude_multi.sh ~/project-a ~/project-b ~/project-c
#   ./claude_multi.sh -s dev -f dirs.txt
#

SESSION_NAME="claude-multi"
CLAUDE_CMD="" # 아래에서 OAuth 토큰 로드 후 설정
DIRS=()
FROM_FILE=""

usage() {
    cat <<'USAGE'
Usage:
  claude_multi.sh [options] <dir1> <dir2> [dir3 ...]

Options:
  -s NAME    tmux 세션 이름 (기본: claude-multi)
  -f FILE    디렉토리 목록 파일 (한 줄에 하나씩)
  -k [NAME]  세션 종료 (이름 생략 시 기본 세션)
  -l         실행 중인 claude tmux 세션 목록
  -h         도움말

Examples:
  claude_multi.sh ~/frontend ~/backend ~/infra
  claude_multi.sh -s my-work -f dirs.txt
  claude_multi.sh -k my-work
USAGE
    exit 0
}

# --- 옵션 파싱 ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s)
            SESSION_NAME="$2"
            shift 2
            ;;
        -f)
            FROM_FILE="$2"
            shift 2
            ;;
        -k)
            target="${2:-$SESSION_NAME}"
            if tmux has-session -t "$target" 2>/dev/null; then
                tmux kill-session -t "$target"
                echo "세션 '$target' 종료됨"
            else
                echo "세션 '$target'을 찾을 수 없음"
            fi
            exit 0
            ;;
        -l)
            echo "=== Claude tmux 세션 목록 ==="
            tmux list-sessions 2>/dev/null || echo "(실행 중인 세션 없음)"
            exit 0
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "알 수 없는 옵션: $1"
            usage
            ;;
        *)
            DIRS+=("$1")
            shift
            ;;
    esac
done

# --- 파일에서 디렉토리 읽기 ---
if [[ -n "$FROM_FILE" ]]; then
    if [[ ! -f "$FROM_FILE" ]]; then
        echo "오류: 파일을 찾을 수 없음: $FROM_FILE"
        exit 1
    fi
    while IFS= read -r line; do
        line="${line%%#*}"       # 주석 제거
        line="$(echo "$line" | xargs)"  # 앞뒤 공백 제거
        [[ -n "$line" ]] && DIRS+=("$line")
    done < "$FROM_FILE"
fi

# --- 디렉토리 검증 ---
if [[ ${#DIRS[@]} -lt 1 ]]; then
    echo "오류: 최소 1개의 디렉토리를 지정해야 합니다."
    echo ""
    usage
fi

VALID_DIRS=()
for dir in "${DIRS[@]}"; do
    expanded="$(eval echo "$dir")"  # ~/ 확장
    if [[ ! -d "$expanded" ]]; then
        echo "경고: 디렉토리가 존재하지 않아 건너뜀: $expanded"
    else
        VALID_DIRS+=("$expanded")
    fi
done

if [[ ${#VALID_DIRS[@]} -lt 1 ]]; then
    echo "오류: 유효한 디렉토리가 없습니다."
    exit 1
fi

# --- 기존 세션 확인 ---
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "세션 '$SESSION_NAME'이 이미 존재합니다."
    echo "  연결: tmux attach -t $SESSION_NAME"
    echo "  종료: $0 -k $SESSION_NAME"
    exit 1
fi

# --- tmux 중첩 실행 허용 (이미 tmux 안에서 실행할 경우) ---
unset TMUX

# --- 스크립트 디렉토리 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- OAuth 토큰 로드 (.oauth_token 파일에서) ---
OAUTH_TOKEN_FILE="$SCRIPT_DIR/.oauth_token"
if [[ -f "$OAUTH_TOKEN_FILE" ]]; then
    OAUTH_TOKEN="$(tr -d '[:space:]' < "$OAUTH_TOKEN_FILE")"
    if [[ -n "$OAUTH_TOKEN" ]]; then
        export CLAUDE_CODE_OAUTH_TOKEN="$OAUTH_TOKEN"
        echo "OAuth 토큰 로드됨: $OAUTH_TOKEN_FILE"
    else
        echo "경고: 토큰 파일이 비어있음: $OAUTH_TOKEN_FILE"
    fi
else
    echo "경고: 토큰 파일을 찾을 수 없음: $OAUTH_TOKEN_FILE"
fi

# --- Claude 실행 커맨드 구성 ---
if [[ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]]; then
    CLAUDE_CMD="export CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_CODE_OAUTH_TOKEN && IS_SANDBOX=1 claude --dangerously-skip-permissions"
else
    CLAUDE_CMD="IS_SANDBOX=1 claude --dangerously-skip-permissions"
fi

# --- tmux 설정 파일 로드 (스크립트와 같은 디렉토리의 tmux.conf) ---
TMUX_CONF="$SCRIPT_DIR/tmux.conf"
if [[ -f "$TMUX_CONF" ]]; then
    tmux start-server
    tmux source-file "$TMUX_CONF" 2>/dev/null
    TMUX_CONF_FLAG="-f $TMUX_CONF"
    echo "설정 로드: $TMUX_CONF"
else
    TMUX_CONF_FLAG=""
fi

# --- tmux 세션 생성 ---
echo "=== Claude Multi Session ==="
echo "세션: $SESSION_NAME"
echo "디렉토리 ${#VALID_DIRS[@]}개:"

# 첫 번째 디렉토리로 세션 생성
first_dir="${VALID_DIRS[0]}"
window_name="$(basename "$first_dir")"
tmux $TMUX_CONF_FLAG new-session -d -s "$SESSION_NAME" -n "$window_name" -c "$first_dir"
tmux send-keys -t "$SESSION_NAME:$window_name" "$CLAUDE_CMD" Enter
echo "  [0] $window_name -> $first_dir"

# 나머지 디렉토리를 별도 윈도우로 추가
for i in $(seq 1 $((${#VALID_DIRS[@]} - 1))); do
    dir="${VALID_DIRS[$i]}"
    window_name="$(basename "$dir")"
    tmux new-window -t "$SESSION_NAME" -n "$window_name" -c "$dir"
    tmux send-keys -t "$SESSION_NAME:$window_name" "$CLAUDE_CMD" Enter
    echo "  [$i] $window_name -> $dir"
done

echo ""
echo "=== 사용법 ==="
echo "  연결:       tmux attach -t $SESSION_NAME"
echo "  윈도우 전환: Ctrl+B, 번호(0,1,2...)"
echo "  윈도우 목록: Ctrl+B, W"
echo "  세션 분리:   Ctrl+B, D"
echo "  세션 종료:   $0 -k $SESSION_NAME"
