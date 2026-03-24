#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script instead of executing it:"
  echo "  source ${BASH_SOURCE[0]}"
  exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  for distro in jazzy humble foxy iron galactic; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      export ROS_DISTRO="${distro}"
      break
    fi
  done
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "[setup_ros_env] Could not detect ROS_DISTRO under /opt/ros"
  return 1
fi

source "/opt/ros/${ROS_DISTRO}/setup.bash"

for ws in \
  "${UNITREE_ROS2_WS:-}" \
  "$HOME/unitree_ros2" \
  "$HOME/ros2_ws" \
  "$HOME/go2_ws"; do
  if [[ -n "$ws" && -f "$ws/install/setup.bash" ]]; then
    source "$ws/install/setup.bash"
    export UNITREE_ROS2_WS="$ws"
    break
  fi
done

export PYTHONPATH="${REPO_ROOT}/realworld:${REPO_ROOT}:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

cat <<EOF
[setup_ros_env] ROS_DISTRO=${ROS_DISTRO}
[setup_ros_env] UNITREE_ROS2_WS=${UNITREE_ROS2_WS:-<not found>}
[setup_ros_env] PYTHONPATH includes:
  - ${REPO_ROOT}
  - ${REPO_ROOT}/realworld

Example usage:
  source ${SCRIPT_DIR}/setup_ros_env.sh
  python ${SCRIPT_DIR}/go2_platform_sim.py
  STREAMVLN_SERVER_URL=http://127.0.0.1:5802/eval_vln \
  STREAMVLN_INSTRUCTION='走到门口。任务：检查门口是否关闭' \
  python ${SCRIPT_DIR}/go2_vln_client.py
EOF
