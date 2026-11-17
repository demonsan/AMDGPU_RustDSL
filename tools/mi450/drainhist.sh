# for every pair of consecutive tensor ops (linear order, same file): the number of instructions between them and whether an s_wait_tensorcnt <= 10 sits between
O=$W/hangscan
for f in $O/*.s; do
  awk -v F=$(basename $f) '
    /^\t[a-z_0-9]+/ {
      mn=$1; n++
      if (mn ~ /^tensor_(load|store)/) {
        if (last>0) { gap=n-last-1; print gap, (waited?"WAIT":"NOWAIT"), F }
        last=n; waited=0
      } else if (mn=="s_wait_tensorcnt") {
        v=$2; sub(/^0x/,"",v); val=(substr($2,1,2)=="0x") ? strtonum($2) : $2+0; if (val<=10) waited=1
      } else if (mn=="s_endpgm") { last=0 }
    }' $f
done > $W/drainpairs.txt
echo "pairs: $(wc -l < $W/drainpairs.txt)"
echo "--- NOWAIT pairs: gap distribution (min / count)"
awk '$2=="NOWAIT" {print $1}' $W/drainpairs.txt | sort -n | uniq -c | sort -k2n | head -20
echo "--- WAIT pairs: gap distribution (smallest 8 and largest 8)"
awk '$2=="WAIT" {print $1}' $W/drainpairs.txt | sort -n | uniq -c | sort -k2n | head -8
awk '$2=="WAIT" {print $1}' $W/drainpairs.txt | sort -n | uniq -c | sort -k2n | tail -8
echo "--- adjacent pairs (gap < 10) without a wait:"
awk '$2=="NOWAIT" && $1<10' $W/drainpairs.txt | sort | uniq -c | head
