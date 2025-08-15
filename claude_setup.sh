#!/usr/bin/env bash
set -euo pipefail

# 1) 설치
curl -fsSL https://claude.ai/install.sh | bash

# 2) 현재 셸에 즉시 PATH 반영 (이 줄이 핵심)
export PATH="$HOME/.local/bin:$PATH"

# 3) 앞으로 열 셸에 영구 반영 (사용 중인 셸 감지)
if [ -n "${ZSH_VERSION-}" ]; then
  RC="$HOME/.zshrc"
elif [ -n "${BASH_VERSION-}" ]; then
  RC="$HOME/.bashrc"
else
  RC="$HOME/.profile"
fi

# 중복 없이 추가
if [ -f "$RC" ]; then
  grep -q 'HOME/.local/bin' "$RC" || printf '\n# Claude CLI\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$RC"
else
  printf '# Claude CLI\nexport PATH="$HOME/.local/bin:$PATH"\n' > "$RC"
fi

# 4) 설치 확인
command -v claude
claude --version
claude doctor
