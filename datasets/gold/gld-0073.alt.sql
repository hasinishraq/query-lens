SELECT COUNT(*) FILTER (WHERE on_hand_units = 0) * 1.0 / COUNT(*) AS v FROM inventory_snapshots WHERE snapshot_date >= DATE '2025-01-01' AND snapshot_date <= DATE '2025-03-31'
