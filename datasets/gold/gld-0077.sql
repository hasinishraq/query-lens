SELECT AVG(CASE WHEN s.on_hand_units = 0 THEN 1.0 ELSE 0.0 END) AS value FROM inventory_snapshots s WHERE s.snapshot_date BETWEEN DATE '2024-04-01' AND DATE '2024-04-30'
