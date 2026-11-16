SELECT SUM(s.shipping_fee_bdt) FROM shipments s WHERE s.shipped_at_utc >= TIMESTAMP '2025-06-01 00:00:00' - INTERVAL 6 HOUR AND s.shipped_at_utc < TIMESTAMP '2025-06-30 00:00:00' + INTERVAL 18 HOUR
