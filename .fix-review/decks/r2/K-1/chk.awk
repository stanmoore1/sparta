# counts N2 (type 1) particles whose vibmode level != round(evib/(k*3371))
/ITEM: ATOMS/ {go=1; next}
go && $2==1 {n++; lv=int($3/(1.380649e-23*3371)+0.5); if (lv!=$4) bad++; if ($4>0) nz++}
END {printf "N2 %d mismatch %d nonzero %d\n", n, bad+0, nz+0}
