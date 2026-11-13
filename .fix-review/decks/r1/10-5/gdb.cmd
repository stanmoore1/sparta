set pagination off
break Modify::post_run
run
print sparta->modify->fmask[0]
print sparta->modify->n_start_of_step
print sparta->modify->n_end_of_step
continue
quit
