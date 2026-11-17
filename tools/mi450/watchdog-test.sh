# exercises the hang policy without touching a GPU kernel: gpurun on a command that outlives its limit must print HANG and return 137, then the payload stops
echo "GPU: ${IDLE_GPU:-none}"
gpurun sleep 30
rc=$?
echo "gpurun rc=$rc"
[ "$rc" = 137 ] && echo "WATCHDOG_OK" || echo "WATCHDOG_FAIL"
exit 0
