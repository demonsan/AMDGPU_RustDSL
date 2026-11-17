O="$W/hangscan"
cnt() { printf "%-34s %s\n" "$1" "$(cat $O/*.s | grep -cE "$2")"; }
cnt "tensor_load/store_to/from_lds" '\s(tensor_load_to_lds|tensor_store_from_lds)\s'
cnt "s_clause" '\ss_clause\s'
cnt "s_sleep (any)" '\ss_sleep\s'
cnt "s_monitor_sleep" '\ss_monitor_sleep'
cnt "s_barrier_signal" '\ss_barrier_signal'
cnt "s_barrier_wait" '\ss_barrier_wait'
cnt "s_barrier_join/leave/init/wakeup" '\ss_(barrier_join|barrier_leave|barrier_init|wakeup_barrier)'
cnt "DPP modifiers" '(quad_perm|row_shl|row_shr|row_ror|row_mirror|row_half_mirror|row_bcast|row_share|row_xmask|dpp8)'
cnt "v_permlane*" '\sv_permlane'
cnt "v_cmpx*" '\sv_cmpx'
cnt "s_setreg*" '\ss_setreg'
cnt "s_alloc_vgpr" '\ss_alloc_vgpr'
cnt "cluster_load" '\scluster_load'
cnt "s_set_vgpr_msb" '\ss_set_vgpr_msb'
echo "--- distinct barrier operands (signal/wait)"
cat $O/*.s | grep -E '\ss_barrier_(signal|wait)' | sed -E 's#//.*##' | awk '{print $1, $2}' | sort | uniq -c | sort -rn | head -12
echo "--- distinct setreg hwregs"
cat $O/*.s | grep -E '\ss_setreg' | sed -E 's#//.*##' | sed -E 's/^\s+//' | sort | uniq -c | sort -rn | head -12
echo "--- sleeps"
cat $O/*.s | grep -E '\ss_(sleep|monitor_sleep)' | sed -E 's#//.*##' | sed -E 's/^\s+//' | sort | uniq -c | sort -rn | head -6
