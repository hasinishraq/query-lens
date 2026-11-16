SELECT SUM(CASE WHEN on_hand_units = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS v FROM inventory_snapshots WHERE snapshot_date >= DATE '2026-07-01' AND snapshot_date <= DATE '2026-09-30'
