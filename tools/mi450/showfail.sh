grep -n -A14 "lite_probes_satisfy_the_contract_on_both_backends stdout" "$W/test.log" | head -30
grep -n -A30 "ui_cases stdout" "$W/test.log" | head -50
