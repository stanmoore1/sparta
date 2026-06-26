## 简要说明（中文摘要）

简单来说：**`read_particles` 本身不会导致"网格乱序"。**

- 你说的"乱序"，其实是指**粒子和网格单元在每个进程内部列表里的排列顺序**。这个顺序本来就不是按几何位置或 ID 排的，而是按插入/到达顺序排的。它只影响**可复现性**：顺序不同 → 随机数消耗顺序不同 → 两次"相同"的计算在统计上会有差异。
- `comm/sort yes`（你已经设了）就是用来解决这个问题的：在**相同进程数**下保证结果可复现。
- `read_particles` 读取是确定性的：proc 0 按文件顺序广播粒子，各进程按顺序把属于自己的粒子追加到列表末尾。固定文件 + 固定进程数 → 顺序固定。
- 真正会"打乱"顺序的是正常的并行操作：每步的粒子迁移、ghost 单元构建、`balance_grid` / `fix balance` / `adapt_grid` 等。
- 你的脚本里**没有 `collide` 命令**，所以按单元的粒子排序（`particle->sort()`）根本不会执行，列表不按单元分组是正常现象，不是数据损坏。
- **脚本里最可能的真正问题**：你在循环里每一步都重复读取**同一个快照（timestep 1）**且从不删除粒子，导致粒子数无限增长、粒子 ID 重复累积。建议在循环**之前**只读一次（或每次读不同的 `Nstep`）。

---

Thanks for the report. I think there's a terminology mix-up worth clearing up
first, because `read_particles` itself isn't the source of the "out of order"
behavior you're seeing.

## What "order" means in SPARTA

The thing you're calling **乱序** is the *order in which particles and grid cells
sit in each processor's internal lists*. That order is **not** geometric and
**not** sorted by ID — it reflects insertion/arrival order. It matters only for
reproducibility: as the `global comm/sort` docs note, if migration messages
arrive in a randomized order, the per-proc lists end up ordered differently, the
RNG is consumed in a different sequence, and two otherwise-identical runs can
diverge statistically. Setting `comm/sort yes` (which you've done) makes this
reproducible run-to-run **on the same number of MPI ranks**.

## `read_particles` is deterministic

Looking at the code: proc 0 reads the dump and broadcasts particles in file
order; every proc keeps the ones in cells it owns, appending them to the end of
its local list in that same order. For a fixed dump file and a fixed rank count,
the insertion order is fixed. **`read_particles` does not scramble anything.**

Where ordering actually gets reshuffled is normal parallel operation — particle
migration every timestep, ghost-cell setup, and `balance_grid` / `fix balance` /
`adapt_grid`. `comm/sort yes` is exactly the knob that makes those reproducible.

One detail specific to your setup: you have no `collide` command, so the
per-cell particle sort (`particle->sort()`) never runs during your timesteps.
That's fine for `move`, but it means the list is never grouped by cell — which
may be what you're observing and calling 乱序. That's expected, not corruption.

## The likely real issue in your script

```
variable count loop 10000
label check_loop
read_particles eztest1 1      # always re-reads timestep 1
run 1
next count
jump ezt.U check_loop
```

You re-read the **same snapshot (timestep 1) on every iteration** and never
delete particles, so:

- the particle count grows by `np` every step without bound, and
- the dump IDs collide every iteration → **duplicate particle IDs** accumulate
  (`read_particles` does no ID dedup).

That accumulation, plus normal per-proc list reordering from migration, is most
likely what looks like the mesh "going out of order." You probably want to
`read_particles` **once before the loop** (or read a different `Nstep` each
iteration), not on every step.

## Your three questions

1. **Avoiding scrambled order with `read_particles`:** it doesn't scramble it —
   keep `comm/sort yes` and a fixed rank count for reproducibility. Fix the loop
   so you aren't re-reading the same snapshot every step.
2. **Other commands that affect ordering:** normal timestepping (migration each
   step), `balance_grid`, `fix balance`, `adapt_grid`, and ghost-cell setup.
   `comm/sort yes` makes these reproducible on a fixed proc count.
3. **Checking for "disorder":** order being non-geometric isn't an error. To
   verify consistency, run twice identically with `comm/sort yes` — statistics
   should match on the same proc count. Comparing `comm/sort yes` vs `no` shows
   you the statistical-noise effect of ordering. If you actually want particles
   grouped by cell internally, that requires `collide` (or the KOKKOS
   `particle/reorder` option), which triggers the sort.

If you can share what specifically looks wrong (e.g. a dump diff between two
runs, or the symptom that made you suspect disorder), I'm happy to dig further.
