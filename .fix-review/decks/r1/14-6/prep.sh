# writes rst.base / rst.lev2 with fin_serial and patches maxlevel -> 100 into rst.bad
source /home/user/sparta/.fix-review/tools/env.sh
$BIN/fin_serial -in in.write -echo none > log.write 2>&1
$BIN/fin_serial -in in.write2 -echo none > log.write2 2>&1
python3 patch.py rst.base rst.lev2 rst.bad ${PATCHVAL:-100} > log.patch
