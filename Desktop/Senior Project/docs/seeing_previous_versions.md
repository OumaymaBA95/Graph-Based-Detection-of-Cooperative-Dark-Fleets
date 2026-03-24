# Seeing what things were like “before”

Your repo may not have many **git commits** for `docs/` and `scripts/` (lots of edits stayed local). Use the options below to **compare** or **recover** older looks.

---

## 1. Case study track plots (Figure 1)

### “Before” = sparse / fast (old default)

Earlier runs used **`--max-files-per-year 5`**, which only sampled **5 daily CSV files total** from the whole archive. That made plots **fast** but **wrong** for validation: few points, **0** or tiny `days_within_km`, not comparable to `docs/candidate_case_studies.md` (e.g. 102 days within 25 km).

**Regenerate that look** (saved to a separate folder so you don’t overwrite full-data plots):

```bash
cd "/Users/momoba/Desktop/Senior Project"
python3 scripts/compute_pair_overlap_from_daily.py \
  --pairs artifacts/case_study_pairs.csv \
  --daily-root "data/MMSI daily vessels " \
  --top-k 1 \
  --distance-km 25 \
  --day-window 1 \
  --max-files-per-year 5 \
  --out-dir artifacts/plots/case_study_pairs_SPARSE_sample \
  --out-summary artifacts/case_study_overlap_summary_SPARSE.csv
```

Open **`artifacts/plots/case_study_pairs_SPARSE_sample/pair_412000690_412325200.png`** and compare to the full-data version.

### “Now” = full daily files (intended for reporting)

```bash
python3 scripts/compute_pair_overlap_from_daily.py \
  --pairs artifacts/case_study_pairs.csv \
  --daily-root "data/MMSI daily vessels " \
  --top-k 1 \
  --distance-km 25 \
  --day-window 1 \
  --max-files-per-year 0 \
  --out-dir artifacts/plots/case_study_pairs \
  --out-summary artifacts/case_study_overlap_summary.csv
```

Default in the script is **`--max-files-per-year 0`** (all files) unless you override it.

---

## 2. Figure 1 with country · gear in title/legend (reverted feature)

That was **removed** from `scripts/compute_pair_overlap_from_daily.py` on purpose (see chat). There is **no separate PNG** in git for “with labels” unless you still have an old file on disk.

- **Cursor / VS Code:** right‑click `pair_412000690_412325200.png` → **Open Timeline** / **Local History** (if enabled) to recover an older PNG.
- **Time Machine:** restore an older `artifacts/plots/case_study_pairs/` from backup.

---

## 3. Git: why `git show HEAD:...final_report.md` fails

If you see:

`fatal: path 'Desktop/Senior Project/docs/final_report.md' exists on disk, but not in 'HEAD'`

that means the file **was never committed** (or is listed in `.gitignore`) in your home-directory repo. Git only knows about **tracked** files, so **`git show HEAD:...` cannot show old versions** of that path until you add and commit it at least once.

**Check:**

```bash
git -C /Users/momoba status "Desktop/Senior Project/docs/final_report.md"
git -C /Users/momoba ls-files "Desktop/Senior Project/docs/final_report.md"
```

If `ls-files` prints nothing, it’s **untracked** — there is no “previous version” in git history for that file.

**If you want git history from now on:**

```bash
cd /Users/momoba
git add "Desktop/Senior Project/docs/final_report.md"
git commit -m "Add final report"
```

After that, future edits will have **previous commits** to compare with `git show` / `git diff`.

---

## 4. Turn on Local History (best way to see “before” without git)

In **Cursor / VS Code**: Settings → search **local history** → enable.

Then open `docs/final_report.md` → **Timeline** (bottom of Explorer or Command Palette: “Local History”) → pick an earlier save. This works even when the file is **not** in git.
